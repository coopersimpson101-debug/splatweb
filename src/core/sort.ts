/**
 * Sorting utilities: depth-key encoding, a CPU LSD radix sort (the fallback path), and a
 * step-by-step CPU *simulation* of the GPU radix sort kernels (histogram -> scan -> stable
 * scatter) that uses the same block sizes and memory layout as `radix_*.wgsl`.
 */
import type { Mat4 } from './math/mat4';

export const RADIX_BITS = 8;
export const RADIX = 1 << RADIX_BITS; // 256 buckets
export const RADIX_PASSES = 32 / RADIX_BITS; // 4
export const SORT_WG_SIZE = 256; // threads per workgroup
export const SORT_ITEMS_PER_THREAD = 8; // chunks of 256 processed sequentially per block
export const SORT_BLOCK = SORT_WG_SIZE * SORT_ITEMS_PER_THREAD; // 2048 keys per block
export const SCAN_BLOCK = 512; // elements per scan workgroup (256 threads x 2)
export const CULLED_KEY = 0xffffffff;

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

/** Raw IEEE bits of a float. For positive floats the bit pattern is monotonic in the value. */
export function floatBits(x: number): number {
  f32[0] = x;
  return u32[0];
}

/** Order-preserving map from any float (including negatives) to u32. */
export function floatToSortableU32(x: number): number {
  const b = floatBits(x);
  return (b & 0x80000000 ? ~b : b | 0x80000000) >>> 0;
}

/**
 * Sort key for a splat at positive view depth `d`, matching preprocess.wgsl:
 *  back-to-front -> ~bits(d) (ascending key = farthest first)
 *  front-to-back ->  bits(d) (ascending key = nearest first)
 */
export function depthKey(d: number, frontToBack: boolean): number {
  const b = floatBits(d);
  return frontToBack ? b : ~b >>> 0;
}

/** Stable LSD radix sort of (key, value) pairs, 4 passes of 8 bits. Returns new arrays. */
export function radixSortPairs(keysIn: Uint32Array, valsIn: Uint32Array): { keys: Uint32Array<ArrayBuffer>; vals: Uint32Array<ArrayBuffer> } {
  const n = keysIn.length;
  let keys = Uint32Array.from(keysIn);
  let vals = Uint32Array.from(valsIn);
  let k2 = new Uint32Array(n);
  let v2 = new Uint32Array(n);
  const counts = new Uint32Array(RADIX);
  for (let pass = 0; pass < RADIX_PASSES; pass++) {
    const shift = pass * RADIX_BITS;
    counts.fill(0);
    for (let i = 0; i < n; i++) counts[(keys[i] >>> shift) & 0xff]++;
    // Skip passes where every key has the same digit (common for the high byte).
    let trivial = false;
    for (let b = 0; b < RADIX; b++) if (counts[b] === n) trivial = true;
    if (trivial) continue;
    let sum = 0;
    for (let b = 0; b < RADIX; b++) {
      const c = counts[b];
      counts[b] = sum;
      sum += c;
    }
    for (let i = 0; i < n; i++) {
      const dst = counts[(keys[i] >>> shift) & 0xff]++;
      k2[dst] = keys[i];
      v2[dst] = vals[i];
    }
    [keys, k2] = [k2, keys];
    [vals, v2] = [v2, vals];
  }
  return { keys, vals };
}

/**
 * CPU fallback: computes view depths for all splats, culls points behind `near`, and returns
 * splat indices in draw order. Culled splats are appended at the end (the GPU preprocess pass
 * still zeroes their quads, so drawing them is harmless).
 */
export function cpuDepthSort(
  positions: Float32Array,
  count: number,
  view: Mat4,
  frontToBack: boolean,
  near = 0.01,
): Uint32Array<ArrayBuffer> {
  const keys = new Uint32Array(count);
  const vals = new Uint32Array(count);
  const v2 = view[2], v6 = view[6], v10 = view[10], v14 = view[14];
  for (let i = 0; i < count; i++) {
    const d = -(v2 * positions[i * 3] + v6 * positions[i * 3 + 1] + v10 * positions[i * 3 + 2] + v14);
    keys[i] = d < near ? CULLED_KEY : depthKey(d, frontToBack);
    vals[i] = i;
  }
  return radixSortPairs(keys, vals).vals;
}

export const sortBlockCount = (n: number): number => Math.max(1, Math.ceil(n / SORT_BLOCK));

/** Sizes of the recursive scan levels for a histogram of `256 * blocks` counters. */
export function scanLevels(histogramSize: number): number[] {
  const levels = [histogramSize];
  let s = histogramSize;
  while (s > SCAN_BLOCK) {
    s = Math.ceil(s / SCAN_BLOCK);
    levels.push(s);
  }
  return levels;
}

/** Work-efficient (Blelloch) exclusive scan of one SCAN_BLOCK, exactly like `radix_scan.wgsl`. */
function blellochBlock(temp: Uint32Array): number {
  let offset = 1;
  for (let d = SCAN_BLOCK >> 1; d > 0; d >>= 1) {
    for (let t = 0; t < d; t++) {
      const ai = offset * (2 * t + 1) - 1;
      const bi = offset * (2 * t + 2) - 1;
      temp[bi] = (temp[bi] + temp[ai]) >>> 0;
    }
    offset *= 2;
  }
  const total = temp[SCAN_BLOCK - 1];
  temp[SCAN_BLOCK - 1] = 0;
  for (let d = 1; d < SCAN_BLOCK; d *= 2) {
    offset >>= 1;
    for (let t = 0; t < d; t++) {
      const ai = offset * (2 * t + 1) - 1;
      const bi = offset * (2 * t + 2) - 1;
      const x = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] = (temp[bi] + x) >>> 0;
    }
  }
  return total;
}

/** Multi-level exclusive scan (scan blocks -> scan block sums -> add back), in place. */
export function simulateMultiLevelScan(data: Uint32Array): void {
  const levels: Uint32Array[] = [data];
  for (const size of scanLevels(data.length).slice(1)) levels.push(new Uint32Array(size));
  const temp = new Uint32Array(SCAN_BLOCK);
  for (let l = 0; l < levels.length; l++) {
    const buf = levels[l];
    const blocks = Math.ceil(buf.length / SCAN_BLOCK);
    for (let b = 0; b < blocks; b++) {
      for (let i = 0; i < SCAN_BLOCK; i++) {
        const idx = b * SCAN_BLOCK + i;
        temp[i] = idx < buf.length ? buf[idx] : 0;
      }
      const total = blellochBlock(temp);
      for (let i = 0; i < SCAN_BLOCK; i++) {
        const idx = b * SCAN_BLOCK + i;
        if (idx < buf.length) buf[idx] = temp[i];
      }
      if (l + 1 < levels.length) levels[l + 1][b] = total;
    }
  }
  for (let l = levels.length - 2; l >= 0; l--) {
    const buf = levels[l];
    const sums = levels[l + 1];
    for (let i = 0; i < buf.length; i++) buf[i] = (buf[i] + sums[Math.floor(i / SCAN_BLOCK)]) >>> 0;
  }
}

/**
 * CPU simulation of the GPU sort: same block decomposition, the same column-major global
 * histogram layout `hist[digit * numBlocks + block]`, the same multi-level scan and the same
 * stable in-block ranking. Used to validate the algorithm independently of a GPU.
 */
export function simulateGpuRadixSort(keysIn: Uint32Array, valsIn: Uint32Array): { keys: Uint32Array<ArrayBuffer>; vals: Uint32Array<ArrayBuffer> } {
  const n = keysIn.length;
  const numBlocks = sortBlockCount(n);
  let kA = Uint32Array.from(keysIn), vA = Uint32Array.from(valsIn);
  let kB = new Uint32Array(n), vB = new Uint32Array(n);
  const hist = new Uint32Array(RADIX * numBlocks);
  for (let pass = 0; pass < RADIX_PASSES; pass++) {
    const shift = pass * RADIX_BITS;
    // 1) per-block histograms
    hist.fill(0);
    for (let b = 0; b < numBlocks; b++) {
      for (let i = 0; i < SORT_BLOCK; i++) {
        const idx = b * SORT_BLOCK + i;
        if (idx < n) hist[((kA[idx] >>> shift) & 0xff) * numBlocks + b]++;
      }
    }
    // 2) exclusive scan over (digit-major, block-minor) layout
    simulateMultiLevelScan(hist);
    // 3) stable scatter, chunk by chunk, like the workgroup does
    for (let b = 0; b < numBlocks; b++) {
      const offsets = new Uint32Array(RADIX);
      for (let d = 0; d < RADIX; d++) offsets[d] = hist[d * numBlocks + b];
      for (let c = 0; c < SORT_ITEMS_PER_THREAD; c++) {
        const digits = new Uint32Array(SORT_WG_SIZE).fill(RADIX);
        const chunkHist = new Uint32Array(RADIX);
        for (let t = 0; t < SORT_WG_SIZE; t++) {
          const idx = b * SORT_BLOCK + c * SORT_WG_SIZE + t;
          if (idx < n) {
            digits[t] = (kA[idx] >>> shift) & 0xff;
            chunkHist[digits[t]]++;
          }
        }
        for (let t = 0; t < SORT_WG_SIZE; t++) {
          const idx = b * SORT_BLOCK + c * SORT_WG_SIZE + t;
          if (idx >= n) continue;
          let rank = 0;
          for (let j = 0; j < t; j++) if (digits[j] === digits[t]) rank++;
          const dst = offsets[digits[t]] + rank;
          kB[dst] = kA[idx];
          vB[dst] = vA[idx];
        }
        for (let d = 0; d < RADIX; d++) offsets[d] += chunkHist[d];
      }
    }
    [kA, kB] = [kB, kA];
    [vA, vB] = [vB, vA];
  }
  return { keys: kA, vals: vA };
}
