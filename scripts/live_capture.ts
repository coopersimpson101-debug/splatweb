/**
 * Headless live-stream client: connects to a splatscan WebSocket stream, applies every message
 * through the same LiveScene + flushLiveScene path the browser uses, and renders a frame with
 * Dawn each time a FRAME marker arrives — from a fixed third-person "dollhouse" camera.
 *
 *   npx vite-node scripts/live_capture.ts -- --url ws://127.0.0.1:8765 --out live_frames \
 *       [--width 480 --height 360] [--eye 0,-7.5,7.5] [--target 0,0,0.6] [--fov 50] [--clip x0,y0,z0,x1,y1,z1]
 *
 * --eye/--target are in *stream* coordinates (z-up for splatscan). Writes <out>/live_<frame>.rgba
 * (raw RGBA8, composited over a dark background) and <out>/frames.json with, per frame, the
 * splat count, GPU capacity/grow count, bytes received and the device camera frustum projected
 * into the image (so the compositor can draw it). Exits when the server closes the socket or
 * after --idle seconds without messages.
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { create, globals } from 'webgpu';
import * as m4 from '../src/core/math/mat4';
import { LiveScene, flushLiveScene, upAxisModelMatrix } from '../src/io/live';
import type { LiveFrame } from '../src/io/live';
import type { SplatData } from '../src/io/splatData';

Object.assign(globalThis, globals);
const args = process.argv.slice(2);
const opt = (name: string, def: string) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && i + 1 < args.length ? args[i + 1] : def;
};
const vec = (s: string) => s.split(',').map(Number) as [number, number, number];

async function main() {
  const url = opt('url', 'ws://127.0.0.1:8765');
  const out = opt('out', 'live_frames');
  const W = parseInt(opt('width', '480'), 10);
  const H = parseInt(opt('height', '360'), 10);
  const idle = parseFloat(opt('idle', '20'));
  const clipS = opt('clip', '');
  const clipV = clipS ? clipS.split(',').map(Number) : null;
  mkdirSync(out, { recursive: true });

  const gpu = create([]);
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter (Dawn)');
  // Dawn crashes if the instance/adapter are garbage-collected while the device lives.
  (globalThis as { __keepAlive?: unknown[] }).__keepAlive = [gpu, adapter];
  const device = await adapter.requestDevice({
    requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize },
  });
  const { SplatRenderer } = await import('../src/gpu/renderer');
  const renderer = new SplatRenderer({ device, colorFormat: 'rgba8unorm', timestamps: false });
  const tex = device.createTexture({ size: [W, H], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  const bpr = Math.ceil((W * 4) / 256) * 256;
  const readBuf = device.createBuffer({ size: bpr * H, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

  const model = upAxisModelMatrix(2);
  const toY = (p: [number, number, number]): [number, number, number] => [p[0], p[2], -p[1]];
  const eye = toY(vec(opt('eye', '0,-7.5,7.5')));
  const target = toY(vec(opt('target', '0,0,0.6')));
  const view = m4.lookAt(eye, target, [0, 1, 0]);
  const proj = m4.perspectiveZO((parseFloat(opt('fov', '50')) * Math.PI) / 180, W / H, 0.05, 100);
  const vp = m4.multiply(proj, m4.multiply(view, model));
  const project = (p: number[]) => {
    const c = m4.transformPoint4(vp, p as [number, number, number]);
    return c[3] > 0 ? [((c[0] / c[3]) * 0.5 + 0.5) * W, (0.5 - (c[1] / c[3]) * 0.5) * H] : null;
  };
  const frustum = (f: LiveFrame) => {
    const d = 0.45;
    const corners = [[0, 0], [f.width - 1, 0], [f.width - 1, f.height - 1], [0, f.height - 1]].map(([u, v]) => {
      const c = [((u - f.cx) / f.fx) * d, ((v - f.cy) / f.fy) * d, d];
      const R = f.R_wc;
      return [0, 1, 2].map((r) => R[r * 3] * c[0] + R[r * 3 + 1] * c[1] + R[r * 3 + 2] * c[2] + f.p_wc[r]);
    });
    return [project(f.p_wc), ...corners.map(project)];
  };

  const state: { bound?: SplatData | null } = {};
  const log: unknown[] = [];
  // Messages are queued and applied strictly in order, rendering at each FRAME marker, so the
  // capture shows exactly the map state the device had at that frame even if the socket
  // delivers faster than we render.
  const queue: ArrayBuffer[] = [];
  let lastMsg = Date.now();
  let closed = false;
  const scene = new LiveScene({
    clip: clipV && clipV.length === 6 ? { min: clipV.slice(0, 3) as [number, number, number], max: clipV.slice(3) as [number, number, number] } : undefined,
  });
  const socket = new WebSocket(url);
  socket.binaryType = 'arraybuffer';
  socket.addEventListener('message', (ev: MessageEvent) => { lastMsg = Date.now(); queue.push(ev.data as ArrayBuffer); });
  socket.addEventListener('close', () => (closed = true));
  await new Promise<void>((res, rej) => { socket.addEventListener('open', () => res()); socket.addEventListener('error', rej); });
  renderer.modelMatrix = model;
  console.log(`connected to ${url}`);
  const bg = [13, 17, 23];
  const t0 = Date.now();
  while (queue.length > 0 || (!closed && (Date.now() - lastMsg) / 1000 < idle)) {
    if (queue.length === 0) {
      await new Promise((r) => setTimeout(r, 5));
      continue;
    }
    const m = scene.applyBuffer(queue.shift()!);
    if (m.type !== 'frame') continue;
    {
      const f = m.frame;
      const tf = performance.now();
      const uploaded = flushLiveScene(scene, renderer, state);
      const flushMs = performance.now() - tf;
      renderer.render({ colorView: tex.createView(), clear: { r: 0, g: 0, b: 0, a: 0 } }, { view, proj, viewport: [W, H], near: 0.05 });
      const enc = device.createCommandEncoder();
      enc.copyTextureToBuffer({ texture: tex }, { buffer: readBuf, bytesPerRow: bpr }, [W, H]);
      device.queue.submit([enc.finish()]);
      await readBuf.mapAsync(GPUMapMode.READ);
      const src = new Uint8Array(readBuf.getMappedRange());
      const rgba = new Uint8Array(W * H * 4);
      for (let y = 0; y < H; y++)
        for (let x = 0; x < W; x++) {
          const s = y * bpr + x * 4, d = (y * W + x) * 4;
          const a = src[s + 3] / 255;
          for (let c = 0; c < 3; c++) rgba[d + c] = Math.min(255, Math.round(src[s + c] + (1 - a) * bg[c]));
          rgba[d + 3] = 255;
        }
      readBuf.unmap();
      writeFileSync(join(out, `live_${String(f.frameIndex).padStart(4, '0')}.rgba`), rgba);
      log.push({
        frame: f.frameIndex, t: f.t, splats: scene.size, streamSplatCount: f.splatCount, gpuCapacity: renderer.gpuCapacity,
        grows: renderer.stats.grows ?? 0, uploaded, flushMs: +flushMs.toFixed(3), bytes: scene.counts.bytes,
        messages: scene.counts.messages, frustum: frustum(f),
      });
    }
  }
  writeFileSync(join(out, 'frames.json'), JSON.stringify({ width: W, height: H, eye, target, frames: log, counts: scene.counts }, null, 1));
  console.log(`rendered ${log.length} frames, ${scene.size} splats, ${scene.counts.messages} messages, ` +
    `${(scene.counts.bytes / 2 ** 20).toFixed(1)} MiB in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
  socket.close();
  void gpu;
  renderer.destroy();
  device.destroy();
  process.exit(0);
}

try {
  await main();
} catch (e) {
  console.error(e);
  process.exit(1);
}
