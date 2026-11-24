/**
 * Runs the REAL WGSL shaders on a real WebGPU implementation (Google's Dawn, via the `webgpu`
 * npm package) inside Node. If no adapter is available (e.g. a CI runner without Vulkan/D3D12 or
 * a software fallback) the whole suite is skipped and says so — it never fakes a pass.
 */
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import type { SplatRenderer as SplatRendererT } from '../src/gpu/renderer';
import type { GpuRadixSorter as GpuRadixSorterT } from '../src/gpu/radixSort';
import { radixSortPairs } from '../src/core/sort';
import { projectSplat } from '../src/core/covariance';
import * as m4 from '../src/core/math/mat4';
import { generateDemoScene, mulberry32 } from '../src/io/procedural';
import { createSplatData } from '../src/io/splatData';
import type { SplatData } from '../src/io/splatData';
import { colorToDC, coeffCountForDegree } from '../src/core/sh';
import { unpackHalf2 } from '../src/core/half';

let device: GPUDevice | null = null;
let skipReason = '';
const uncaptured: string[] = [];
// Keep the Dawn instance + adapter reachable: if they are garbage-collected while the device is
// alive, the native addon crashes the worker.
let keepAlive: unknown[] = [];
let SplatRenderer: typeof SplatRendererT;
let GpuRadixSorter: typeof GpuRadixSorterT;

beforeAll(async () => {
  // Hosted CI runners have no GPU and the native Dawn addon is not guaranteed to load there, so
  // CI skips this suite unless SPLATWEB_GPU=1 is set (e.g. on a self-hosted GPU runner).
  if (process.env.CI && process.env.SPLATWEB_GPU !== '1') {
    skipReason = 'CI without SPLATWEB_GPU=1';
    return;
  }
  try {
    const webgpu = await import('webgpu');
    Object.assign(globalThis, webgpu.globals);
    const gpu = webgpu.create([]);
    const adapter = await gpu.requestAdapter();
    keepAlive = [gpu, adapter];
    if (!adapter) {
      skipReason = 'no WebGPU adapter';
      return;
    }
    device = await adapter.requestDevice();
    device.addEventListener('uncapturederror', (e) => {
      uncaptured.push((e as GPUUncapturedErrorEvent).error.message);
    });
    ({ SplatRenderer } = await import('../src/gpu/renderer'));
    ({ GpuRadixSorter } = await import('../src/gpu/radixSort'));
  } catch (e) {
    skipReason = `webgpu (Dawn) unavailable: ${(e as Error).message}`;
    device = null;
  }
});

afterAll(() => {
  void keepAlive;
  expect(uncaptured, 'uncaptured WebGPU errors').toEqual([]);
});

async function readBuffer(src: GPUBuffer, size: number, offset = 0): Promise<ArrayBuffer> {
  const d = device!;
  const rb = d.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = d.createCommandEncoder();
  enc.copyBufferToBuffer(src, offset, rb, 0, size);
  d.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const out = rb.getMappedRange().slice(0);
  rb.unmap();
  rb.destroy();
  return out;
}

async function checkErrors<T>(fn: () => Promise<T>): Promise<T> {
  device!.pushErrorScope('validation');
  const r = await fn();
  const err = await device!.popErrorScope();
  if (err) throw new Error(`validation error: ${err.message}`);
  return r;
}

const gpuIt = (name: string, fn: () => Promise<void>) =>
  it(name, async (ctx) => {
    if (!device) {
      console.warn(`[gpu.test] skipped "${name}": ${skipReason}`);
      ctx.skip();
      return;
    }
    await fn();
  });

describe('GPU (Dawn) — real WGSL', () => {
  gpuIt('radix sort matches the CPU reference for many sizes', async () => {
    for (const n of [1, 100, 2047, 2048, 2049, 70_001, 600_000]) {
      const r = mulberry32(n);
      const keys = new Uint32Array(n);
      for (let i = 0; i < n; i++) keys[i] = n % 2 ? Math.floor(r() * 2 ** 32) >>> 0 : Math.floor(r() * 64); // ties for even n
      const vals = Uint32Array.from(keys.keys());
      const sorter = await checkErrors(async () => new GpuRadixSorter(device!, n));
      device!.queue.writeBuffer(sorter.keysA, 0, keys);
      device!.queue.writeBuffer(sorter.valsA, 0, vals);
      const enc = device!.createCommandEncoder();
      const pass = enc.beginComputePass();
      sorter.encode(pass);
      pass.end();
      device!.queue.submit([enc.finish()]);
      const gotK = new Uint32Array(await readBuffer(sorter.keysA, n * 4));
      const gotV = new Uint32Array(await readBuffer(sorter.valsA, n * 4));
      const ref = radixSortPairs(keys, vals);
      expect(gotK, `keys n=${n}`).toEqual(ref.keys);
      expect(gotV, `vals n=${n} (stability)`).toEqual(ref.vals);
      sorter.destroy();
    }
  });

  gpuIt('sorts fewer keys than capacity after setCount (progressive loading)', async () => {
    const cap = 10_000, n = 3_333;
    const sorter = new GpuRadixSorter(device!, cap);
    sorter.setCount(n);
    const r = mulberry32(77);
    const keys = Uint32Array.from({ length: n }, () => Math.floor(r() * 2 ** 32) >>> 0);
    device!.queue.writeBuffer(sorter.keysA, 0, keys);
    device!.queue.writeBuffer(sorter.valsA, 0, Uint32Array.from(keys.keys()));
    const enc = device!.createCommandEncoder();
    const pass = enc.beginComputePass();
    sorter.encode(pass);
    pass.end();
    device!.queue.submit([enc.finish()]);
    const got = new Uint32Array(await readBuffer(sorter.keysA, n * 4));
    expect(got).toEqual(radixSortPairs(keys, Uint32Array.from(keys.keys())).keys);
    sorter.destroy();
  });

  gpuIt('preprocess.wgsl matches the CPU reference projectSplat', async () => {
    const scene = generateDemoScene({ count: 20_000, seed: 5 });
    const W = 640, H = 480;
    const view = m4.lookAt([3, 2.5, 5], [0, 1, 0], [0, 1, 0]);
    const proj = m4.perspectiveZO(Math.PI / 3, W / H, 0.05, 100);
    const r = new SplatRenderer({ device: device!, colorFormat: 'rgba8unorm', timestamps: false });
    r.setScene(scene);
    const tex = device!.createTexture({ size: [W, H], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT });
    await checkErrors(async () => {
      r.render({ colorView: tex.createView(), clear: { r: 0, g: 0, b: 0, a: 0 } }, { view, proj, viewport: [W, H], near: 0.05 });
    });
    const out = await readBuffer(r.buffers.out2d!, scene.loaded * 40);
    const f = new Float32Array(out);
    const u = new Uint32Array(out);
    const args = new Uint32Array(await readBuffer(r.buffers.drawArgs, 16));
    const coeffs = coeffCountForDegree(scene.shDegree) * 3;
    let visibleCpu = 0, mismatchedCull = 0, maxCenterErr = 0, maxAxisRel = 0, maxColorErr = 0;
    for (let i = 0; i < scene.loaded; i++) {
      const p = projectSplat({
        position: [scene.positions[i * 3], scene.positions[i * 3 + 1], scene.positions[i * 3 + 2]],
        scale: [scene.scales[i * 3], scene.scales[i * 3 + 1], scene.scales[i * 3 + 2]],
        rotation: [scene.rotations[i * 4], scene.rotations[i * 4 + 1], scene.rotations[i * 4 + 2], scene.rotations[i * 4 + 3]],
        opacity: scene.opacities[i],
        sh: scene.sh.subarray(i * coeffs, (i + 1) * coeffs),
        shDegree: scene.shDegree,
      }, { view, proj, viewport: [W, H], camPos: [3, 2.5, 5], near: 0.05 });
      const o = i * 10;
      const gpuCulled = f[o + 2] === 0 && f[o + 3] === 0;
      if (p) visibleCpu++;
      if (!p !== gpuCulled) { mismatchedCull++; continue; }
      if (!p) continue;
      maxCenterErr = Math.max(maxCenterErr, Math.abs(f[o] - p.center[0]), Math.abs(f[o + 1] - p.center[1]));
      const a1 = Math.hypot(f[o + 2], f[o + 3]), r1 = Math.hypot(...p.axis1);
      maxAxisRel = Math.max(maxAxisRel, Math.abs(a1 - r1) / Math.max(r1, 1));
      const [cr, cg] = unpackHalf2(u[o + 6]);
      const [cb, op] = unpackHalf2(u[o + 7]);
      maxColorErr = Math.max(maxColorErr, Math.abs(cr - p.color[0]), Math.abs(cg - p.color[1]), Math.abs(cb - p.color[2]), Math.abs(op - p.opacity));
    }
    console.info(`[gpu.test] preprocess: visible gpu=${args[1]} cpu=${visibleCpu} cullMismatch=${mismatchedCull} ` +
      `maxCenterErr=${maxCenterErr.toExponential(2)}px maxAxisRel=${maxAxisRel.toExponential(2)} maxColorErr=${maxColorErr.toExponential(2)}`);
    expect(visibleCpu).toBeGreaterThan(10_000);
    // f16 inputs on the GPU vs f32 on the CPU can flip a handful of borderline culls.
    expect(mismatchedCull).toBeLessThan(scene.loaded * 0.001);
    expect(Math.abs(args[1] - visibleCpu)).toBeLessThan(scene.loaded * 0.001);
    expect(maxCenterErr).toBeLessThan(1e-2);
    expect(maxAxisRel).toBeLessThan(5e-3);
    expect(maxColorErr).toBeLessThan(5e-3);
    const check = await r.validateGpuSort({ view, proj, viewport: [W, H], near: 0.05 });
    expect(check.ok, check.message).toBe(true);
    tex.destroy();
    r.destroy();
  });

  /** Renders `scene` into a WxH rgba8unorm texture and returns RGBA floats in [0,1] (row 0 = top). */
  async function renderToPixels(scene: SplatData, W: number, H: number, blend: 'back-to-front' | 'front-to-back', sort: 'gpu' | 'cpu' = 'gpu') {
    const r = new SplatRenderer({ device: device!, colorFormat: 'rgba8unorm', blend, sort, timestamps: false });
    r.setScene(scene);
    const tex = device!.createTexture({ size: [W, H], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const view = m4.lookAt([0, 0, 5], [0, 0, 0], [0, 1, 0]);
    const proj = m4.perspectiveZO(Math.PI / 4, W / H, 0.05, 100);
    await checkErrors(async () => r.render({ colorView: tex.createView(), clear: { r: 0, g: 0, b: 0, a: 0 } }, { view, proj, viewport: [W, H] }));
    const bpr = Math.ceil((W * 4) / 256) * 256;
    const buf = device!.createBuffer({ size: bpr * H, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device!.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr }, [W, H]);
    device!.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const bytes = new Uint8Array(buf.getMappedRange().slice(0));
    buf.unmap();
    const px = new Float32Array(W * H * 4);
    for (let y = 0; y < H; y++) for (let x = 0; x < W * 4; x++) px[y * W * 4 + x] = bytes[y * bpr + x] / 255;
    buf.destroy();
    tex.destroy();
    r.destroy();
    return { px, view, proj };
  }

  function splatScene(items: { p: [number, number, number]; s: number; c: [number, number, number]; a: number }[]): SplatData {
    const d = createSplatData(items.length, 0);
    items.forEach((it, i) => {
      d.positions.set(it.p, i * 3);
      d.scales.set([it.s, it.s, it.s], i * 3);
      d.rotations.set([1, 0, 0, 0], i * 4);
      d.opacities[i] = it.a;
      d.sh.set(it.c.map(colorToDC), i * 3);
    });
    d.loaded = items.length;
    return d;
  }

  gpuIt('rasterizes a single Gaussian with the analytic falloff (premultiplied)', async () => {
    const W = 64, H = 64;
    const scene = splatScene([{ p: [0, 0, 0], s: 0.12, c: [1, 0.5, 0.25], a: 0.9 }]);
    const { px, view, proj } = await renderToPixels(scene, W, H, 'back-to-front');
    const ref = projectSplat({ position: [0, 0, 0], scale: [0.12, 0.12, 0.12], rotation: [1, 0, 0, 0], opacity: 0.9, sh: scene.sh, shDegree: 0 },
      { view, proj, viewport: [W, H], camPos: [0, 0, 5], near: 0.05 })!;
    let maxErr = 0;
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        const dx = x + 0.5 - ref.center[0];
        const dy = H - (y + 0.5) - ref.center[1]; // texture rows go down, our pixel y goes up
        const m2 = ref.conic[0] * dx * dx + 2 * ref.conic[1] * dx * dy + ref.conic[2] * dy * dy;
        let a = m2 > 9 ? 0 : Math.min(0.99, 0.9 * Math.exp(-0.5 * m2));
        if (a < 1 / 255) a = 0;
        const o = (y * W + x) * 4;
        maxErr = Math.max(maxErr, Math.abs(px[o + 3] - a), Math.abs(px[o] - a * 1), Math.abs(px[o + 1] - a * 0.5), Math.abs(px[o + 2] - a * 0.25));
      }
    console.info(`[gpu.test] single-splat raster max abs error = ${(maxErr * 255).toFixed(2)}/255`);
    expect(maxErr).toBeLessThan(2.5 / 255);
  });

  gpuIt('back-to-front and front-to-back blending agree with a CPU compositing reference', async () => {
    const W = 96, H = 64;
    const items = [
      { p: [-0.2, 0, -1] as [number, number, number], s: 0.35, c: [1, 0.1, 0.1] as [number, number, number], a: 0.8 },
      { p: [0.15, 0.05, 0] as [number, number, number], s: 0.3, c: [0.1, 1, 0.2] as [number, number, number], a: 0.6 },
      { p: [0.3, -0.1, 1] as [number, number, number], s: 0.2, c: [0.2, 0.3, 1] as [number, number, number], a: 0.7 },
    ];
    const scene = splatScene(items);
    const b2f = await renderToPixels(scene, W, H, 'back-to-front');
    const f2b = await renderToPixels(scene, W, H, 'front-to-back');
    const cpu = await renderToPixels(scene, W, H, 'back-to-front', 'cpu');
    const proj = b2f.proj, view = b2f.view;
    const refs = items.map((it) => projectSplat({ position: it.p, scale: [it.s, it.s, it.s], rotation: [1, 0, 0, 0], opacity: it.a, sh: it.c.map(colorToDC), shDegree: 0 },
      { view, proj, viewport: [W, H], camPos: [0, 0, 5], near: 0.05 })!);
    let errB = 0, errF = 0, errC = 0;
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        let C = [0, 0, 0], A = 0;
        for (let k = 0; k < items.length; k++) { // items are listed far -> near: "over" operator
          const r = refs[k];
          const dx = x + 0.5 - r.center[0], dy = H - (y + 0.5) - r.center[1];
          const m2 = r.conic[0] * dx * dx + 2 * r.conic[1] * dx * dy + r.conic[2] * dy * dy;
          let a = m2 > 9 ? 0 : Math.min(0.99, items[k].a * Math.exp(-0.5 * m2));
          if (a < 1 / 255) a = 0;
          C = C.map((v, c) => items[k].c[c] * a + (1 - a) * v);
          A = a + (1 - a) * A;
        }
        const o = (y * W + x) * 4;
        for (let c = 0; c < 4; c++) {
          const want = c < 3 ? C[c] : A;
          errB = Math.max(errB, Math.abs(b2f.px[o + c] - want));
          errF = Math.max(errF, Math.abs(f2b.px[o + c] - want));
          errC = Math.max(errC, Math.abs(cpu.px[o + c] - want));
        }
      }
    console.info(`[gpu.test] blending max abs error: b2f=${(errB * 255).toFixed(2)}/255 f2b=${(errF * 255).toFixed(2)}/255 cpu-sort=${(errC * 255).toFixed(2)}/255`);
    expect(errB).toBeLessThan(4 / 255);
    expect(errF).toBeLessThan(4 / 255);
    expect(errC).toBeLessThan(4 / 255);
  });
});
