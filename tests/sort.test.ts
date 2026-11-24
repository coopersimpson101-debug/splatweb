import { describe, expect, it } from 'vitest';
import {
  CULLED_KEY, SCAN_BLOCK, SORT_BLOCK, cpuDepthSort, depthKey, floatBits, floatToSortableU32, radixSortPairs, scanLevels,
  simulateGpuRadixSort, simulateMultiLevelScan,
} from '../src/core/sort';
import * as m4 from '../src/core/math/mat4';
import { mulberry32 } from '../src/io/procedural';

function randomKeys(n: number, seed: number, range = 2 ** 32) {
  const r = mulberry32(seed);
  const keys = new Uint32Array(n);
  for (let i = 0; i < n; i++) keys[i] = Math.floor(r() * range) >>> 0;
  return keys;
}

function referenceSort(keys: Uint32Array) {
  const idx = Array.from(keys.keys());
  idx.sort((a, b) => keys[a] - keys[b] || a - b); // stable reference
  return idx;
}

describe('depth keys', () => {
  it('positive float bits are monotonic', () => {
    const vals = [1e-6, 0.01, 0.5, 1, 1.0000001, 3, 1000, 1e20];
    for (let i = 1; i < vals.length; i++) expect(floatBits(vals[i])).toBeGreaterThan(floatBits(vals[i - 1]));
  });

  it('sortable u32 handles negatives', () => {
    const vals = [-1e10, -3, -0.5, 0, 0.5, 3, 1e10];
    for (let i = 1; i < vals.length; i++) expect(floatToSortableU32(vals[i])).toBeGreaterThan(floatToSortableU32(vals[i - 1]));
  });

  it('back-to-front keys sort farthest first; culled key sorts last', () => {
    expect(depthKey(10, false)).toBeLessThan(depthKey(1, false));
    expect(depthKey(1, true)).toBeLessThan(depthKey(10, true));
    expect(depthKey(1e30, false)).toBeLessThan(CULLED_KEY);
    expect(depthKey(1e30, true)).toBeLessThan(CULLED_KEY);
  });
});

describe('CPU radix sort', () => {
  it('matches a stable reference sort', () => {
    for (const [n, range] of [[1, 10], [17, 4], [1000, 2 ** 32], [5000, 300], [65536, 2 ** 20]] as const) {
      const keys = randomKeys(n, n, range);
      const vals = Uint32Array.from(keys.keys());
      const out = radixSortPairs(keys, vals);
      expect(Array.from(out.vals)).toEqual(referenceSort(keys));
    }
  });

  it('cpuDepthSort orders splats back-to-front and front-to-back', () => {
    const r = mulberry32(2);
    const n = 3000;
    const pos = new Float32Array(n * 3);
    for (let i = 0; i < n * 3; i++) pos[i] = (r() - 0.5) * 10;
    const V = m4.lookAt([0, 0, 20], [0, 0, 0], [0, 1, 0]);
    const depth = (i: number) => -(V[2] * pos[i * 3] + V[6] * pos[i * 3 + 1] + V[10] * pos[i * 3 + 2] + V[14]);
    const b2f = cpuDepthSort(pos, n, V, false);
    for (let k = 1; k < n; k++) expect(depth(b2f[k - 1])).toBeGreaterThanOrEqual(depth(b2f[k]) - 1e-5); // keys are f32
    const f2b = cpuDepthSort(pos, n, V, true);
    for (let k = 1; k < n; k++) expect(depth(f2b[k - 1])).toBeLessThanOrEqual(depth(f2b[k]) + 1e-5);
  });
});

describe('GPU sort algorithm (CPU simulation of the WGSL kernels)', () => {
  it('multi-level Blelloch scan equals a serial exclusive scan', () => {
    for (const n of [1, 511, 512, 513, 256 * 300, SCAN_BLOCK * SCAN_BLOCK + 7]) {
      const data = randomKeys(n, n, 1000);
      const ref = new Uint32Array(n);
      let s = 0;
      for (let i = 0; i < n; i++) {
        ref[i] = s;
        s += data[i];
      }
      simulateMultiLevelScan(data);
      expect(data).toEqual(ref);
    }
    expect(scanLevels(256)).toEqual([256]);
    expect(scanLevels(512 * 512 + 1)).toEqual([262145, 513, 2]);
  });

  it('histogram/scan/scatter sort matches the stable reference across block boundaries', () => {
    for (const n of [1, 255, 256, SORT_BLOCK - 1, SORT_BLOCK, SORT_BLOCK + 1, 3 * SORT_BLOCK + 123]) {
      const keys = randomKeys(n, 100 + n, n < 300 ? 16 : 2 ** 32); // small range -> many ties
      keys[0] = CULLED_KEY;
      const vals = Uint32Array.from(keys.keys());
      const out = simulateGpuRadixSort(keys, vals);
      expect(Array.from(out.vals)).toEqual(referenceSort(keys));
    }
  });
});
