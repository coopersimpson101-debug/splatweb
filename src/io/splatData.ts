/**
 * In-memory CPU representation of a Gaussian scene with all activations already applied:
 * linear scales (not log), unit quaternions, opacity in [0,1] (not logit).
 */
import { coeffCountForDegree } from '../core/sh';

export interface SplatData {
  /** Allocated capacity in splats. */
  count: number;
  /** Number of splats that are fully populated (grows while streaming). */
  loaded: number;
  shDegree: number;
  /** xyz per splat. */
  positions: Float32Array;
  /** Linear scale per axis (exp(scale_i) of the .ply). */
  scales: Float32Array;
  /** Unit quaternion [w, x, y, z] per splat. */
  rotations: Float32Array;
  /** sigmoid(opacity) per splat. */
  opacities: Float32Array;
  /** SH coefficients, layout [splat][k][rgb] with k < (shDegree+1)². k = 0 is the DC term. */
  sh: Float32Array;
}

export function createSplatData(count: number, shDegree: number): SplatData {
  const coeffs = coeffCountForDegree(shDegree);
  return {
    count,
    loaded: 0,
    shDegree,
    positions: new Float32Array(count * 3),
    scales: new Float32Array(count * 3),
    rotations: new Float32Array(count * 4),
    opacities: new Float32Array(count),
    sh: new Float32Array(count * coeffs * 3),
  };
}

export const shFloatsPerSplat = (d: SplatData): number => coeffCountForDegree(d.shDegree) * 3;

export interface Bounds {
  min: [number, number, number];
  max: [number, number, number];
  center: [number, number, number];
  radius: number;
}

/**
 * Robust bounds: percentile-trimmed so a handful of floaters far away from the scene do not
 * make the initial camera frame a tiny speck.
 */
export function computeBounds(d: SplatData, trim = 0.02): Bounds {
  const n = d.loaded;
  if (n === 0) return { min: [-1, -1, -1], max: [1, 1, 1], center: [0, 0, 0], radius: 1 };
  const step = Math.max(1, Math.floor(n / 20000));
  const min: [number, number, number] = [0, 0, 0];
  const max: [number, number, number] = [0, 0, 0];
  for (let a = 0; a < 3; a++) {
    const vals: number[] = [];
    for (let i = 0; i < n; i += step) vals.push(d.positions[i * 3 + a]);
    vals.sort((x, y) => x - y);
    min[a] = vals[Math.floor(trim * (vals.length - 1))];
    max[a] = vals[Math.ceil((1 - trim) * (vals.length - 1))];
  }
  const center: [number, number, number] = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
  const radius = Math.max(1e-3, Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) / 2);
  return { min, max, center, radius };
}
