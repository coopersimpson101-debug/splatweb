/**
 * Headless render + GPU benchmark using Dawn (the `webgpu` npm package) — no browser needed.
 *
 *   npx vite-node scripts/headless.ts -- --out docs/images/demo.png [--width 1280 --height 720]
 *   npx vite-node scripts/headless.ts -- --bench [--counts 150000,500000,1000000] [--width 1920 --height 1080]
 *   npx vite-node scripts/headless.ts -- --scene path/to/scene.ply --out shot.png [--flip]
 *
 * The benchmark reports GPU timestamp-query times per stage (median over frames). It measures
 * the renderer's GPU work only; it is NOT a browser FPS number.
 */
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import { deflateSync } from 'node:zlib';
import { create, globals } from 'webgpu';
import * as m4 from '../src/core/math/mat4';
import { OrbitCamera, projectionMatrix } from '../src/camera/camera';
import { generateDemoScene } from '../src/io/procedural';
import { parsePly } from '../src/io/ply';
import { parseSplat } from '../src/io/splat';
import { computeBounds } from '../src/io/splatData';
import type { SplatData } from '../src/io/splatData';

Object.assign(globalThis, globals);

const args = process.argv.slice(2);
const flag = (name: string) => args.includes(`--${name}`);
const opt = (name: string, def: string) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && i + 1 < args.length ? args[i + 1] : def;
};

// ---- tiny hand-written PNG encoder (RGBA8, filter 0, zlib via node) ----
const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();
function crc32(buf: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
function chunk(type: string, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + data.length);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  dv.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
}
function encodePng(w: number, h: number, rgba: Uint8Array): Uint8Array {
  const ihdr = new Uint8Array(13);
  const dv = new DataView(ihdr.buffer);
  dv.setUint32(0, w);
  dv.setUint32(4, h);
  ihdr.set([8, 6, 0, 0, 0], 8); // 8-bit RGBA
  const raw = new Uint8Array(h * (w * 4 + 1));
  for (let y = 0; y < h; y++) raw.set(rgba.subarray(y * w * 4, (y + 1) * w * 4), y * (w * 4 + 1) + 1);
  const parts = [new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw, { level: 9 })), chunk('IEND', new Uint8Array(0))];
  const out = new Uint8Array(parts.reduce((s, p) => s + p.length, 0));
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
}

async function main() {
  const gpu = create([]);
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter (Dawn) available');
  const hasTs = adapter.features.has('timestamp-query');
  const device = await adapter.requestDevice({
    requiredFeatures: hasTs ? ['timestamp-query'] : [],
    requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize },
  });
  const { SplatRenderer } = await import('../src/gpu/renderer');
  const info = adapter.info;
  const gpuName = `${info.vendor} ${info.architecture} ${info.device} (${info.description})`.trim();
  const W = parseInt(opt('width', flag('bench') ? '1920' : '1280'), 10);
  const H = parseInt(opt('height', flag('bench') ? '1080' : '720'), 10);
  const format: GPUTextureFormat = 'rgba8unorm';
  const tex = device.createTexture({ size: [W, H], format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });

  const cameraFor = (d: SplatData, demo: boolean, flip: boolean) => {
    const cam = new OrbitCamera();
    if (demo) {
      cam.target = [0, 1.2, 0];
      cam.distance = 6.2;
      cam.yaw = 0.6;
      cam.pitch = 0.32;
    } else {
      const b = computeBounds(d);
      cam.frame(flip ? [b.center[0], -b.center[1], -b.center[2]] : b.center, b.radius);
      cam.pitch = 0.25;
    }
    return cam;
  };

  if (flag('bench')) {
    const counts = opt('counts', '150000,500000,1000000').split(',').map((s) => parseInt(s, 10));
    console.log(`GPU: ${gpuName}\nresolution ${W}x${H}, timestamp-query: ${hasTs}`);
    console.log('| splats | visible | preprocess ms | sort ms | raster ms | total GPU ms |');
    console.log('|---:|---:|---:|---:|---:|---:|');
    for (const n of counts) {
      const scene = generateDemoScene({ count: n });
      const r = new SplatRenderer({ device, colorFormat: format });
      r.setScene(scene);
      const cam = cameraFor(scene, true, false);
      const samples: number[][] = [];
      for (let f = 0; f < 60; f++) {
        cam.yaw += 0.01; // keep the order changing like a real orbit
        r.requestStats();
        r.render({ colorView: tex.createView(), clear: { r: 0, g: 0, b: 0, a: 0 } },
          { view: cam.viewMatrix(), proj: projectionMatrix(cam.lens, W / H), viewport: [W, H], near: cam.lens.near });
        await device.queue.onSubmittedWorkDone();
        await new Promise((res) => setTimeout(res, 1));
        const s = r.stats;
        if (f >= 10 && s.gpuPreprocessMs !== undefined) samples.push([s.gpuPreprocessMs, s.gpuSortMs ?? 0, s.gpuRenderMs ?? 0, s.visible]);
      }
      const med = (k: number) => {
        const v = samples.map((s) => s[k]).sort((a, b) => a - b);
        return v.length ? v[Math.floor(v.length / 2)] : NaN;
      };
      const [p, so, ra] = [med(0), med(1), med(2)];
      console.log(`| ${n.toLocaleString('en-US')} | ${Math.round(med(3)).toLocaleString('en-US')} | ${p.toFixed(2)} | ${so.toFixed(2)} | ${ra.toFixed(2)} | ${(p + so + ra).toFixed(2)} |`);
      r.destroy();
    }
  }

  const out = opt('out', '');
  if (out) {
    const scenePath = opt('scene', '');
    const flip = flag('flip');
    let scene: SplatData;
    if (scenePath) {
      const bytes = new Uint8Array(readFileSync(scenePath));
      scene = scenePath.toLowerCase().endsWith('.splat') ? parseSplat(bytes) : parsePly(bytes);
    } else scene = generateDemoScene({ count: parseInt(opt('count', '150000'), 10) });
    const r = new SplatRenderer({ device, colorFormat: format, timestamps: false });
    if (flip) r.modelMatrix = m4.fromScaling([1, -1, -1]);
    r.setScene(scene);
    const cam = cameraFor(scene, !scenePath, flip);
    r.render({ colorView: tex.createView(), clear: { r: 0, g: 0, b: 0, a: 0 } },
      { view: cam.viewMatrix(), proj: projectionMatrix(cam.lens, W / H), viewport: [W, H], near: cam.lens.near });
    const bpr = Math.ceil((W * 4) / 256) * 256;
    const buf = device.createBuffer({ size: bpr * H, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr }, [W, H]);
    device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(buf.getMappedRange());
    // Composite the premultiplied result over a dark background (like the web app's CSS).
    const bg = [18, 21, 30];
    const rgba = new Uint8Array(W * H * 4);
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        const s = y * bpr + x * 4, d = (y * W + x) * 4;
        const a = src[s + 3] / 255;
        for (let c = 0; c < 3; c++) rgba[d + c] = Math.min(255, Math.round(src[s + c] + (1 - a) * bg[c]));
        rgba[d + 3] = 255;
      }
    buf.unmap();
    mkdirSync(dirname(out) || '.', { recursive: true });
    writeFileSync(out, encodePng(W, H, rgba));
    console.log(`wrote ${out} (${W}x${H}, ${scene.loaded.toLocaleString('en-US')} splats) on ${gpuName}`);
    r.destroy();
  }
  // keep gpu/adapter alive until here (Dawn crashes if they are collected while the device lives)
  void gpu;
  device.destroy();
  process.exit(0);
}

// Top-level await: vite-node closes its module server once the entry module finishes evaluating.
try {
  await main();
} catch (e) {
  console.error(e);
  process.exit(1);
}
