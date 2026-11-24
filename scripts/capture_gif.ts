/**
 * Records the README hero GIF: a full orbit around the built-in procedural scene, rendered by the
 * real SplatRenderer (same WGSL as the browser demo) headless through Dawn, the `webgpu` package.
 * Needs a GPU. Every frame is a real render; nothing is drawn or retouched afterwards.
 *
 *   npm run capture:gif
 *   npx vite-node scripts/capture_gif.ts -- --out docs/images/orbit.gif --frames 60 --width 640 --height 360
 */
import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import { create, globals } from 'webgpu';
// gifenc ships no type declarations
// @ts-expect-error untyped module
import { GIFEncoder, quantize, applyPalette } from 'gifenc';
import { OrbitCamera, projectionMatrix } from '../src/camera/camera';
import { generateDemoScene } from '../src/io/procedural';

Object.assign(globalThis, globals);

const args = process.argv.slice(2);
const opt = (name: string, def: string) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && i + 1 < args.length ? args[i + 1] : def;
};

async function main() {
  const out = opt('out', 'docs/images/orbit.gif');
  const W = parseInt(opt('width', '640'), 10);
  const H = parseInt(opt('height', '360'), 10);
  const frames = parseInt(opt('frames', '60'), 10);
  const count = parseInt(opt('count', '150000'), 10);
  const delay = parseInt(opt('delay', '60'), 10); // ms per frame

  const gpu = create([]);
  (globalThis as { __gpu?: unknown }).__gpu = gpu; // keep Dawn's instance alive (it crashes if collected)
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter (Dawn) available');
  const device = await adapter.requestDevice({
    requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize },
  });
  const { SplatRenderer } = await import('../src/gpu/renderer');
  const info = adapter.info;
  const format: GPUTextureFormat = 'rgba8unorm';
  const tex = device.createTexture({ size: [W, H], format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  const bpr = Math.ceil((W * 4) / 256) * 256;
  const buf = device.createBuffer({ size: bpr * H, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

  const scene = generateDemoScene({ count });
  const r = new SplatRenderer({ device, colorFormat: format, timestamps: false });
  r.setScene(scene);
  const cam = new OrbitCamera();
  cam.target = [0, 1.2, 0];
  cam.distance = 6.2;
  cam.pitch = 0.32;

  const gif = GIFEncoder();
  const bg = [13, 17, 23]; // #0d1117, GitHub dark background
  for (let f = 0; f < frames; f++) {
    cam.yaw = 0.6 + (2 * Math.PI * f) / frames;
    r.render({ colorView: tex.createView(), clear: { r: 0, g: 0, b: 0, a: 0 } },
      { view: cam.viewMatrix(), proj: projectionMatrix(cam.lens, W / H), viewport: [W, H], near: cam.lens.near });
    const enc = device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr }, [W, H]);
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    await new Promise((res) => setTimeout(res, 1));
    await buf.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(buf.getMappedRange());
    // Composite the premultiplied output over the background (as the PNG path in headless.ts does).
    const rgba = new Uint8Array(W * H * 4);
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        const s = y * bpr + x * 4, d = (y * W + x) * 4;
        const a = src[s + 3] / 255;
        for (let c = 0; c < 3; c++) rgba[d + c] = Math.min(255, Math.round(src[s + c] + (1 - a) * bg[c]));
        rgba[d + 3] = 255;
      }
    buf.unmap();
    const palette = quantize(rgba, 256);
    gif.writeFrame(applyPalette(rgba, palette), W, H, { palette, delay, repeat: 0 });
    process.stdout.write(`\rframe ${f + 1}/${frames}`);
  }
  gif.finish();
  const bytes = gif.bytes() as Uint8Array;
  mkdirSync(dirname(out) || '.', { recursive: true });
  writeFileSync(out, bytes);
  console.log(`\nwrote ${out}: ${frames} frames ${W}x${H}, ${(bytes.length / 1e6).toFixed(2)} MB, ` +
    `${scene.loaded.toLocaleString('en-US')} splats on ${info.vendor} ${info.description}`.trim());
  r.destroy();
  void gpu;
  device.destroy();
  process.exit(0);
}

try {
  await main();
} catch (e) {
  console.error(e);
  process.exit(1);
}
