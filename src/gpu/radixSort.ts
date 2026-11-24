/**
 * Host-side orchestration of the hand-written GPU LSD radix sort (32-bit keys + 32-bit values,
 * 4 passes x 8 bits). Each pass = histogram -> multi-level exclusive scan -> stable scatter.
 * Keys/values are ping-ponged between buffer pairs A and B; after 4 passes the sorted result is
 * back in A.
 */
import histogramWGSL from './shaders/radix_histogram.wgsl?raw';
import scanWGSL from './shaders/radix_scan.wgsl?raw';
import scatterWGSL from './shaders/radix_scatter.wgsl?raw';
import { RADIX, RADIX_PASSES, RADIX_BITS, SCAN_BLOCK, scanLevels, sortBlockCount } from '../core/sort';

const U = GPUBufferUsage;
const S = GPUShaderStage;

function layout(device: GPUDevice, label: string, types: GPUBufferBindingType[]): GPUBindGroupLayout {
  return device.createBindGroupLayout({
    label,
    entries: types.map((type, binding) => ({ binding, visibility: S.COMPUTE, buffer: { type } })),
  });
}

export class GpuRadixSorter {
  readonly keysA: GPUBuffer;
  readonly valsA: GPUBuffer;
  readonly keysB: GPUBuffer;
  readonly valsB: GPUBuffer;
  private readonly histogram: GPUBuffer;
  private readonly levelBuffers: GPUBuffer[]; // [histogram, sums1, sums2, ...] at capacity
  private readonly dummy: GPUBuffer;
  private readonly passParams: GPUBuffer[] = [];
  private readonly scanParams: GPUBuffer[] = [];
  private readonly histPipeline: GPUComputePipeline;
  private readonly scanPipeline: GPUComputePipeline;
  private readonly addPipeline: GPUComputePipeline;
  private readonly scatterPipeline: GPUComputePipeline;
  private readonly histBG: GPUBindGroup[] = [];
  private readonly scatterBG: GPUBindGroup[] = [];
  private readonly scanBG: GPUBindGroup[] = [];
  private count = 0;
  private numBlocks = 1;
  private levels: number[] = [RADIX];

  constructor(private readonly device: GPUDevice, readonly capacity: number, extraKeyUsage = 0) {
    const n = Math.max(1, capacity);
    const bytes = n * 4;
    const mk = (label: string, size: number, usage: number) => device.createBuffer({ label, size, usage });
    const kvUsage = U.STORAGE | U.COPY_SRC | U.COPY_DST | extraKeyUsage;
    this.keysA = mk('sort keys A', bytes, kvUsage);
    this.valsA = mk('sort vals A', bytes, kvUsage);
    this.keysB = mk('sort keys B', bytes, kvUsage);
    this.valsB = mk('sort vals B', bytes, kvUsage);
    const capLevels = scanLevels(RADIX * sortBlockCount(n));
    this.histogram = mk('radix histogram', capLevels[0] * 4, U.STORAGE | U.COPY_SRC);
    this.levelBuffers = [this.histogram];
    for (let l = 1; l < capLevels.length; l++) this.levelBuffers.push(mk(`scan sums L${l}`, capLevels[l] * 4, U.STORAGE));
    this.dummy = mk('scan dummy sums', 16, U.STORAGE);

    const histLayout = layout(device, 'radix histogram layout', ['uniform', 'read-only-storage', 'storage']);
    const scanLayout = layout(device, 'radix scan layout', ['uniform', 'storage', 'storage']);
    const scatterLayout = layout(device, 'radix scatter layout', [
      'uniform', 'read-only-storage', 'read-only-storage', 'storage', 'storage', 'read-only-storage',
    ]);
    const pl = (l: GPUBindGroupLayout) => device.createPipelineLayout({ bindGroupLayouts: [l] });
    const histModule = device.createShaderModule({ label: 'radix_histogram.wgsl', code: histogramWGSL });
    const scanModule = device.createShaderModule({ label: 'radix_scan.wgsl', code: scanWGSL });
    const scatterModule = device.createShaderModule({ label: 'radix_scatter.wgsl', code: scatterWGSL });
    this.histPipeline = device.createComputePipeline({
      label: 'radix histogram', layout: pl(histLayout), compute: { module: histModule, entryPoint: 'main' },
    });
    this.scanPipeline = device.createComputePipeline({
      label: 'radix scan', layout: pl(scanLayout), compute: { module: scanModule, entryPoint: 'scan_block' },
    });
    this.addPipeline = device.createComputePipeline({
      label: 'radix add sums', layout: pl(scanLayout), compute: { module: scanModule, entryPoint: 'add_sums' },
    });
    this.scatterPipeline = device.createComputePipeline({
      label: 'radix scatter', layout: pl(scatterLayout), compute: { module: scatterModule, entryPoint: 'main' },
    });

    for (let p = 0; p < RADIX_PASSES; p++) {
      const params = mk(`radix pass ${p} params`, 16, U.UNIFORM | U.COPY_DST);
      this.passParams.push(params);
      const [kin, vin, kout, vout] = p % 2 === 0
        ? [this.keysA, this.valsA, this.keysB, this.valsB]
        : [this.keysB, this.valsB, this.keysA, this.valsA];
      this.histBG.push(device.createBindGroup({
        layout: histLayout,
        entries: [params, kin, this.histogram].map((buffer, binding) => ({ binding, resource: { buffer } })),
      }));
      this.scatterBG.push(device.createBindGroup({
        layout: scatterLayout,
        entries: [params, kin, vin, kout, vout, this.histogram].map((buffer, binding) => ({ binding, resource: { buffer } })),
      }));
    }
    for (let l = 0; l < this.levelBuffers.length; l++) {
      const params = mk(`scan level ${l} params`, 16, U.UNIFORM | U.COPY_DST);
      this.scanParams.push(params);
      const sums = this.levelBuffers[l + 1] ?? this.dummy;
      this.scanBG.push(device.createBindGroup({
        layout: scanLayout,
        entries: [params, this.levelBuffers[l], sums].map((buffer, binding) => ({ binding, resource: { buffer } })),
      }));
    }
    this.setCount(capacity);
  }

  /** Number of keys to sort (<= capacity). Cheap: only rewrites a few uniforms. */
  setCount(count: number): void {
    if (count > this.capacity) throw new Error(`sort count ${count} exceeds capacity ${this.capacity}`);
    if (count === this.count) return;
    this.count = count;
    this.numBlocks = sortBlockCount(count);
    this.levels = scanLevels(RADIX * this.numBlocks);
    for (let p = 0; p < RADIX_PASSES; p++)
      this.device.queue.writeBuffer(this.passParams[p], 0, new Uint32Array([count, p * RADIX_BITS, this.numBlocks, 0]));
    for (let l = 0; l < this.levels.length; l++)
      this.device.queue.writeBuffer(this.scanParams[l], 0, new Uint32Array([this.levels[l], 0, 0, 0]));
  }

  get sortCount(): number {
    return this.count;
  }

  /** Records the full 4-pass sort of keysA/valsA (first `count` entries) into a compute pass. */
  encode(pass: GPUComputePassEncoder): void {
    if (this.count === 0) return;
    for (let p = 0; p < RADIX_PASSES; p++) {
      pass.setPipeline(this.histPipeline);
      pass.setBindGroup(0, this.histBG[p]);
      pass.dispatchWorkgroups(this.numBlocks);

      pass.setPipeline(this.scanPipeline);
      for (let l = 0; l < this.levels.length; l++) {
        pass.setBindGroup(0, this.scanBG[l]);
        pass.dispatchWorkgroups(Math.ceil(this.levels[l] / SCAN_BLOCK));
      }
      pass.setPipeline(this.addPipeline);
      for (let l = this.levels.length - 2; l >= 0; l--) {
        pass.setBindGroup(0, this.scanBG[l]);
        pass.dispatchWorkgroups(Math.ceil(this.levels[l] / SCAN_BLOCK));
      }

      pass.setPipeline(this.scatterPipeline);
      pass.setBindGroup(0, this.scatterBG[p]);
      pass.dispatchWorkgroups(this.numBlocks);
    }
  }

  destroy(): void {
    for (const b of [this.keysA, this.valsA, this.keysB, this.valsB, this.dummy, ...this.levelBuffers, ...this.passParams, ...this.scanParams])
      b.destroy();
  }
}
