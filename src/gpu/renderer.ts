/**
 * SplatRenderer — the embeddable core. It owns GPU buffers and pipelines for one scene and
 * records preprocess -> sort -> draw into a command encoder the *host* provides, targeting a
 * texture view the host provides. It never touches canvases, DOM or input; see
 * docs/vanta-integration.md for the contract a host engine must satisfy.
 */
import preprocessWGSL from './shaders/preprocess.wgsl?raw';
import renderWGSL from './shaders/render.wgsl?raw';
import type { Mat4 } from '../core/math/mat4';
import { create as mat4Create, invert, multiply } from '../core/math/mat4';
import { CULLED_KEY, cpuDepthSort } from '../core/sort';
import type { SplatData } from '../io/splatData';
import { GpuRadixSorter } from './radixSort';
import {
  PREPROCESS_UNIFORM_BYTES, SPLAT2D_BYTES, SPLAT_BYTES, packSH, packSplats, shWordsPerSplat, writePreprocessUniforms,
} from './packing';

export type SortBackend = 'gpu' | 'cpu';
export type BlendOrder = 'back-to-front' | 'front-to-back';

export interface SplatRendererOptions {
  device: GPUDevice;
  /** Format of the colour target(s) you will pass to encode(). */
  colorFormat: GPUTextureFormat;
  /** If set, splats are depth-tested (read-only) against a host depth buffer of this format. */
  depthFormat?: GPUTextureFormat;
  /** Depth compare function for the host depth buffer (use 'greater' for reversed-Z). */
  depthCompare?: GPUCompareFunction;
  sort?: SortBackend;
  blend?: BlendOrder;
  /** Request GPU timestamps if the device has 'timestamp-query'. */
  timestamps?: boolean;
}

/** Camera for one view. Matrices are column-major Float32Array(16). */
export interface SplatView {
  /** World -> view. Right-handed, camera looks down -Z, +Y up. */
  view: Mat4;
  /** View -> clip. x,y in [-1,1] (y up); z convention only matters when depth testing. */
  proj: Mat4;
  /** Size in pixels of the region being rendered (the viewport, not necessarily the texture). */
  viewport: [number, number];
  /** View-space near distance below which splats are culled. Default 0.05. */
  near?: number;
}

export interface SplatTarget {
  colorView: GPUTextureView;
  depthView?: GPUTextureView;
  /** Sub-rectangle [x, y, w, h] in pixels (e.g. one eye of an XR layer). Defaults to full target. */
  viewportRect?: [number, number, number, number];
  /** Clear colour (premultiplied). null/undefined = load existing contents. */
  clear?: GPUColor | null;
}

export interface RendererStats {
  splats: number;
  visible: number;
  /** GPU milliseconds (only when timestamps are available). */
  gpuPreprocessMs?: number;
  gpuSortMs?: number;
  gpuRenderMs?: number;
  cpuSortMs?: number;
}

const U = () => GPUBufferUsage;

interface SlotResources {
  uniforms: GPUBuffer;
  renderParams: GPUBuffer;
  preprocessBG: GPUBindGroup;
  renderGpuBG: GPUBindGroup;
  cpuOrder?: GPUBuffer;
  renderCpuBG?: GPUBindGroup;
  lastCpuView?: Float32Array;
  lastCpuCount?: number;
}

export class SplatRenderer {
  readonly device: GPUDevice;
  sort: SortBackend;
  blend: BlendOrder;
  /** Multiplies every splat's scale (debug / artistic). */
  scaleModifier = 1;
  /** Maximum SH degree to evaluate (clamped to the scene's degree). */
  maxShDegree = 3;
  /** Model -> world transform applied to the scene (e.g. flip COLMAP's Y-down to Y-up). */
  modelMatrix: Mat4 = mat4Create();
  readonly stats: RendererStats = { splats: 0, visible: 0 };

  private readonly colorFormat: GPUTextureFormat;
  private readonly depthFormat?: GPUTextureFormat;
  private readonly depthCompare: GPUCompareFunction;
  private readonly preprocessLayout: GPUBindGroupLayout;
  private readonly renderLayout: GPUBindGroupLayout;
  private readonly preprocessPipeline: GPUComputePipeline;
  private readonly renderPipelines = new Map<string, GPURenderPipeline>();
  private readonly renderModule: GPUShaderModule;

  private capacity = 0;
  private count = 0;
  private shDegree = 0;
  private positions: Float32Array | null = null;
  private splatBuf: GPUBuffer | null = null;
  private shBuf: GPUBuffer | null = null;
  private out2d: GPUBuffer | null = null;
  private sorter: GpuRadixSorter | null = null;
  private readonly drawArgs: GPUBuffer;
  private slots: SlotResources[] = [];
  private uploaded = 0;

  // Timestamps / stats readback.
  private readonly querySet: GPUQuerySet | null = null;
  private readonly queryResolve: GPUBuffer | null = null;
  private readonly readback: GPUBuffer;
  private readbackBusy = false;
  private readbackRequested = false;
  private frame = 0;

  constructor(opts: SplatRendererOptions) {
    const device = opts.device;
    this.device = device;
    this.colorFormat = opts.colorFormat;
    this.depthFormat = opts.depthFormat;
    this.depthCompare = opts.depthCompare ?? 'less';
    this.sort = opts.sort ?? 'gpu';
    this.blend = opts.blend ?? 'back-to-front';
    const S = GPUShaderStage;
    this.preprocessLayout = device.createBindGroupLayout({
      label: 'preprocess layout',
      entries: [
        { binding: 0, visibility: S.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: S.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: S.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: S.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: S.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: S.COMPUTE, buffer: { type: 'storage' } },
        { binding: 6, visibility: S.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    this.renderLayout = device.createBindGroupLayout({
      label: 'render layout',
      entries: [
        { binding: 0, visibility: S.VERTEX, buffer: { type: 'uniform' } },
        { binding: 1, visibility: S.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: S.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });
    this.preprocessPipeline = device.createComputePipeline({
      label: 'splat preprocess',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.preprocessLayout] }),
      compute: { module: device.createShaderModule({ label: 'preprocess.wgsl', code: preprocessWGSL }), entryPoint: 'main' },
    });
    this.renderModule = device.createShaderModule({ label: 'render.wgsl', code: renderWGSL });
    this.drawArgs = device.createBuffer({
      label: 'splat draw args',
      size: 16,
      usage: U().STORAGE | U().INDIRECT | U().COPY_DST | U().COPY_SRC,
    });
    device.queue.writeBuffer(this.drawArgs, 0, new Uint32Array([4, 0, 0, 0]));
    if ((opts.timestamps ?? true) && device.features.has('timestamp-query')) {
      this.querySet = device.createQuerySet({ type: 'timestamp', count: 4 });
      this.queryResolve = device.createBuffer({ size: 32, usage: U().QUERY_RESOLVE | U().COPY_SRC });
    }
    this.readback = device.createBuffer({ label: 'stats readback', size: 48, usage: U().MAP_READ | U().COPY_DST });
  }

  get splatCount(): number {
    return this.count;
  }

  get sceneShDegree(): number {
    return this.shDegree;
  }

  /** Internal buffers, exposed for tests and advanced hosts. */
  get buffers() {
    return { splats: this.splatBuf, sh: this.shBuf, out2d: this.out2d, keys: this.sorter?.keysA, vals: this.sorter?.valsA, drawArgs: this.drawArgs };
  }

  /**
   * Allocates GPU storage for `data.count` splats and uploads the `data.loaded` that are ready.
   * Call `updateScene(data)` as more splats stream in.
   */
  setScene(data: SplatData): void {
    this.releaseScene();
    const d = this.device;
    const cap = Math.max(1, data.count);
    const limit = d.limits.maxStorageBufferBindingSize;
    const shBytes = cap * shWordsPerSplat(data.shDegree) * 4;
    if (shBytes > limit || cap * SPLAT2D_BYTES > limit)
      throw new Error(
        `Scene too large for this device: ${data.count} splats need ${(Math.max(shBytes, cap * SPLAT2D_BYTES) / 2 ** 20).toFixed(0)} MiB ` +
        `per storage binding, limit is ${(limit / 2 ** 20).toFixed(0)} MiB. Lower the SH degree or split the scene.`,
      );
    this.capacity = cap;
    this.shDegree = data.shDegree;
    this.positions = data.positions;
    this.splatBuf = d.createBuffer({ label: 'splats', size: cap * SPLAT_BYTES, usage: U().STORAGE | U().COPY_DST });
    this.shBuf = d.createBuffer({ label: 'splat SH', size: shBytes, usage: U().STORAGE | U().COPY_DST });
    this.out2d = d.createBuffer({ label: 'splats 2D', size: cap * SPLAT2D_BYTES, usage: U().STORAGE | U().COPY_SRC });
    this.sorter = new GpuRadixSorter(d, cap);
    this.uploaded = 0;
    this.count = 0;
    this.updateScene(data);
  }

  /** Uploads splats in [previously uploaded, data.loaded). */
  updateScene(data: SplatData): void {
    if (!this.splatBuf || !this.shBuf || !this.sorter) throw new Error('setScene() first');
    const start = this.uploaded;
    const end = Math.min(data.loaded, this.capacity);
    if (end <= start) return;
    this.device.queue.writeBuffer(this.splatBuf, start * SPLAT_BYTES, packSplats(data, start, end));
    const words = shWordsPerSplat(data.shDegree);
    this.device.queue.writeBuffer(this.shBuf, start * words * 4, packSH(data, start, end));
    this.uploaded = end;
    this.count = end;
    this.stats.splats = end;
    this.sorter.setCount(end);
  }

  private slot(i: number): SlotResources {
    let s = this.slots[i];
    if (s) return s;
    const d = this.device;
    const uniforms = d.createBuffer({ label: `preprocess uniforms ${i}`, size: PREPROCESS_UNIFORM_BYTES, usage: U().UNIFORM | U().COPY_DST });
    const renderParams = d.createBuffer({ label: `render params ${i}`, size: 16, usage: U().UNIFORM | U().COPY_DST });
    const sorter = this.sorter!;
    const preprocessBG = d.createBindGroup({
      layout: this.preprocessLayout,
      entries: [uniforms, this.splatBuf!, this.shBuf!, this.out2d!, sorter.keysA, sorter.valsA, this.drawArgs].map(
        (buffer, binding) => ({ binding, resource: { buffer } }),
      ),
    });
    const renderGpuBG = d.createBindGroup({
      layout: this.renderLayout,
      entries: [renderParams, this.out2d!, sorter.valsA].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    s = { uniforms, renderParams, preprocessBG, renderGpuBG };
    this.slots[i] = s;
    return s;
  }

  private pipeline(): GPURenderPipeline {
    const key = `${this.blend}`;
    let p = this.renderPipelines.get(key);
    if (p) return p;
    const blend: GPUBlendState =
      this.blend === 'back-to-front'
        ? {
            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          }
        : {
            color: { srcFactor: 'one-minus-dst-alpha', dstFactor: 'one', operation: 'add' },
            alpha: { srcFactor: 'one-minus-dst-alpha', dstFactor: 'one', operation: 'add' },
          };
    p = this.device.createRenderPipeline({
      label: `splat render (${key})`,
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.renderLayout] }),
      vertex: { module: this.renderModule, entryPoint: 'vs_main' },
      fragment: { module: this.renderModule, entryPoint: 'fs_main', targets: [{ format: this.colorFormat, blend }] },
      primitive: { topology: 'triangle-strip', cullMode: 'none' },
      depthStencil: this.depthFormat
        ? { format: this.depthFormat, depthWriteEnabled: false, depthCompare: this.depthCompare }
        : undefined,
    });
    this.renderPipelines.set(key, p);
    return p;
  }

  /**
   * Records preprocess + sort + draw for one view into `encoder`.
   * `slot` must be distinct for every view encoded before the same queue.submit (e.g. 0 and 1
   * for the two XR eyes) because uniforms are uploaded with queue.writeBuffer.
   */
  encode(encoder: GPUCommandEncoder, target: SplatTarget, view: SplatView, slot = 0): void {
    const clearOnly = this.count === 0 || !this.out2d;
    const [vx, vy, vw, vh] = target.viewportRect ?? [0, 0, view.viewport[0], view.viewport[1]];
    if (!clearOnly) {
      const s = this.slot(slot);
      const modelView = multiply(view.view, this.modelMatrix);
      const inv = invert(modelView) ?? mat4Create();
      const frontToBack = this.blend === 'front-to-back';
      const near = view.near ?? 0.05;
      this.device.queue.writeBuffer(
        s.uniforms, 0,
        writePreprocessUniforms({
          view: modelView, proj: view.proj, camPos: [inv[12], inv[13], inv[14]], viewport: [vw, vh],
          splatCount: this.count, shDegree: Math.min(this.maxShDegree, this.shDegree),
          shStride: shWordsPerSplat(this.shDegree), frontToBack, scaleModifier: this.scaleModifier, near,
        }),
      );
      this.device.queue.writeBuffer(s.renderParams, 0, new Float32Array([vw, vh, 0, 0]));
      const timing = slot === 0 && this.querySet !== null;

      encoder.clearBuffer(this.drawArgs, 4, 4); // instanceCount = 0
      const pre = encoder.beginComputePass({
        label: 'splat preprocess',
        timestampWrites: timing ? { querySet: this.querySet!, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } : undefined,
      });
      pre.setPipeline(this.preprocessPipeline);
      pre.setBindGroup(0, s.preprocessBG);
      const wgs = Math.ceil(this.count / 256);
      pre.dispatchWorkgroups(Math.min(wgs, 65535), Math.ceil(wgs / 65535));
      pre.end();

      if (this.sort === 'gpu') {
        const sp = encoder.beginComputePass({
          label: 'splat radix sort',
          timestampWrites: timing ? { querySet: this.querySet!, endOfPassWriteIndex: 2 } : undefined,
        });
        this.sorter!.encode(sp);
        sp.end();
      } else {
        this.updateCpuOrder(s, modelView, frontToBack, near);
      }

      const rp = this.beginRenderPass(encoder, target, timing);
      rp.setViewport(vx, vy, vw, vh, 0, 1);
      rp.setPipeline(this.pipeline());
      if (this.sort === 'gpu') {
        rp.setBindGroup(0, s.renderGpuBG);
        rp.drawIndirect(this.drawArgs, 0);
      } else {
        rp.setBindGroup(0, s.renderCpuBG!);
        rp.draw(4, this.count);
      }
      rp.end();

      if (slot === 0 && this.readbackRequested && !this.readbackBusy) {
        if (timing) {
          encoder.resolveQuerySet(this.querySet!, 0, 4, this.queryResolve!, 0);
          encoder.copyBufferToBuffer(this.queryResolve!, 0, this.readback, 0, 32);
        }
        encoder.copyBufferToBuffer(this.drawArgs, 0, this.readback, 32, 16);
        this.readbackBusy = true;
        this.readbackRequested = false;
        this.pendingMap = true;
      }
    } else if (target.clear) {
      this.beginRenderPass(encoder, target, false).end();
    }
  }

  private pendingMap = false;

  private beginRenderPass(encoder: GPUCommandEncoder, target: SplatTarget, timing: boolean): GPURenderPassEncoder {
    const hasStencil = this.depthFormat?.includes('stencil') ?? false;
    return encoder.beginRenderPass({
      label: 'splat render',
      colorAttachments: [{
        view: target.colorView,
        loadOp: target.clear ? 'clear' : 'load',
        clearValue: target.clear ?? undefined,
        storeOp: 'store',
      }],
      depthStencilAttachment: target.depthView && this.depthFormat
        ? { view: target.depthView, depthReadOnly: true, stencilReadOnly: hasStencil ? true : undefined }
        : undefined,
      timestampWrites: timing ? { querySet: this.querySet!, endOfPassWriteIndex: 3 } : undefined,
    });
  }

  private updateCpuOrder(s: SlotResources, modelView: Mat4, frontToBack: boolean, near: number) {
    const d = this.device;
    if (!s.cpuOrder) {
      s.cpuOrder = d.createBuffer({ label: 'cpu sort order', size: this.capacity * 4, usage: U().STORAGE | U().COPY_DST });
      s.renderCpuBG = d.createBindGroup({
        layout: this.renderLayout,
        entries: [s.renderParams, this.out2d!, s.cpuOrder].map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
    }
    // Re-sort only when the view (or the number of splats / ordering) actually changed.
    const key = new Float32Array([...modelView, frontToBack ? 1 : 0]);
    const same = s.lastCpuView && s.lastCpuCount === this.count && s.lastCpuView.every((v, k) => Math.abs(v - key[k]) < 1e-6);
    if (same) return;
    const t0 = performance.now();
    const order = cpuDepthSort(this.positions!, this.count, modelView, frontToBack, near);
    this.stats.cpuSortMs = performance.now() - t0;
    d.queue.writeBuffer(s.cpuOrder, 0, order);
    s.lastCpuView = key;
    s.lastCpuCount = this.count;
  }

  /** Requests a stats readback (visible count + GPU timings) on the next encode(). */
  requestStats(): void {
    this.readbackRequested = true;
  }

  /**
   * Must be called after the queue.submit() that contained encode() if you use requestStats().
   * `render()` does this for you.
   */
  afterSubmit(): void {
    this.frame++;
    if (!this.pendingMap) return;
    this.pendingMap = false;
    this.readback.mapAsync(GPUMapMode.READ).then(
      () => {
        const buf = this.readback.getMappedRange();
        const ts = new BigUint64Array(buf, 0, 4);
        const args = new Uint32Array(buf, 32, 4);
        this.stats.visible = args[1];
        if (this.querySet && ts[0] > 0n) {
          const ms = (a: bigint, b: bigint) => (b > a ? Number(b - a) / 1e6 : 0);
          this.stats.gpuPreprocessMs = ms(ts[0], ts[1]);
          this.stats.gpuSortMs = this.sort === 'gpu' ? ms(ts[1], ts[2]) : 0;
          this.stats.gpuRenderMs = ms(this.sort === 'gpu' ? ts[2] : ts[1], ts[3]);
        }
        this.readback.unmap();
        this.readbackBusy = false;
      },
      () => {
        this.readbackBusy = false;
      },
    );
  }

  /** Convenience: encode one view, submit, and handle stats readback. */
  render(target: SplatTarget, view: SplatView): void {
    const enc = this.device.createCommandEncoder({ label: 'splatweb frame' });
    this.encode(enc, target, view, 0);
    this.device.queue.submit([enc.finish()]);
    this.afterSubmit();
  }

  /**
   * Runtime self-test of the GPU sort on the *current* scene and view: re-runs preprocess+sort,
   * reads keys back and checks they are non-decreasing, that exactly `visible` keys are valid and
   * that the values are a permutation of distinct splat indices.
   */
  async validateGpuSort(view: SplatView): Promise<{ ok: boolean; visible: number; message: string }> {
    if (!this.sorter || this.count === 0) return { ok: false, visible: 0, message: 'no scene loaded' };
    const prevSort = this.sort;
    this.sort = 'gpu';
    const d = this.device;
    const tex = d.createTexture({ size: [Math.max(1, view.viewport[0]), Math.max(1, view.viewport[1])], format: this.colorFormat, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    const enc = d.createCommandEncoder();
    this.encode(enc, { colorView: tex.createView(), clear: { r: 0, g: 0, b: 0, a: 0 } }, view, 7);
    const n = this.count;
    const keysRead = d.createBuffer({ size: n * 4, usage: U().MAP_READ | U().COPY_DST });
    const valsRead = d.createBuffer({ size: n * 4, usage: U().MAP_READ | U().COPY_DST });
    const argsRead = d.createBuffer({ size: 16, usage: U().MAP_READ | U().COPY_DST });
    enc.copyBufferToBuffer(this.sorter.keysA, 0, keysRead, 0, n * 4);
    enc.copyBufferToBuffer(this.sorter.valsA, 0, valsRead, 0, n * 4);
    enc.copyBufferToBuffer(this.drawArgs, 0, argsRead, 0, 16);
    d.queue.submit([enc.finish()]);
    this.sort = prevSort;
    await Promise.all([keysRead.mapAsync(GPUMapMode.READ), valsRead.mapAsync(GPUMapMode.READ), argsRead.mapAsync(GPUMapMode.READ)]);
    const keys = new Uint32Array(keysRead.getMappedRange().slice(0));
    const vals = new Uint32Array(valsRead.getMappedRange().slice(0));
    const visible = new Uint32Array(argsRead.getMappedRange().slice(0))[1];
    for (const b of [keysRead, valsRead, argsRead]) { b.unmap(); b.destroy(); }
    tex.destroy();
    for (let i = 1; i < n; i++)
      if (keys[i - 1] > keys[i]) return { ok: false, visible, message: `keys out of order at ${i}: ${keys[i - 1]} > ${keys[i]}` };
    let valid = 0;
    for (let i = 0; i < n; i++) if (keys[i] !== CULLED_KEY) valid++;
    if (valid !== visible) return { ok: false, visible, message: `visible count ${visible} != valid keys ${valid}` };
    const seen = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      if (vals[i] >= n || seen[vals[i]]) return { ok: false, visible, message: `values are not a permutation (index ${i})` };
      seen[vals[i]] = 1;
    }
    return { ok: true, visible, message: `GPU sort OK: ${n} keys sorted, ${visible} visible` };
  }

  private releaseScene() {
    this.splatBuf?.destroy();
    this.shBuf?.destroy();
    this.out2d?.destroy();
    this.sorter?.destroy();
    this.slots.forEach((s) => {
      s.uniforms.destroy();
      s.renderParams.destroy();
      s.cpuOrder?.destroy();
    });
    this.slots = [];
    this.splatBuf = this.shBuf = this.out2d = null;
    this.sorter = null;
    this.count = this.capacity = this.uploaded = 0;
  }

  destroy(): void {
    this.releaseScene();
    this.drawArgs.destroy();
    this.readback.destroy();
    this.queryResolve?.destroy();
    this.querySet?.destroy();
  }
}
