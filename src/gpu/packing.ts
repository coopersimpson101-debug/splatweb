/**
 * GPU buffer layouts. These must match `preprocess.wgsl` exactly.
 *
 * Splat buffer, 8 x u32 = 32 bytes per splat:
 *   [0..2] position xyz            f32 bits
 *   [3]    scale.x | scale.y       2 x f16
 *   [4]    scale.z | opacity       2 x f16
 *   [5]    rot.w   | rot.x         2 x f16
 *   [6]    rot.y   | rot.z         2 x f16
 *   [7]    reserved (0)
 *
 * SH buffer: (degree+1)² x 3 halves per splat, interleaved [k][rgb], padded to whole u32s.
 *   degree 0 -> 2 words, 1 -> 6, 2 -> 14, 3 -> 24 (96 bytes vs 192 bytes as f32).
 *
 * Splat2D (preprocess output, read by the vertex shader), 40 bytes:
 *   center: vec2f, axis1: vec2f, axis2: vec2f, color: vec2u (rgba f16), depth: f32, pad: f32
 *
 * Positions stay f32 because f16 has only ~3 decimal digits — not enough for room-scale scenes.
 */
import { packHalf2 } from '../core/half';
import { coeffCountForDegree } from '../core/sh';
import type { SplatData } from '../io/splatData';

export const SPLAT_WORDS = 8;
export const SPLAT_BYTES = SPLAT_WORDS * 4;
export const SPLAT2D_BYTES = 40;

export const shWordsPerSplat = (degree: number): number => Math.ceil((coeffCountForDegree(degree) * 3) / 2);

export function packSplats(d: SplatData, start = 0, end = d.loaded): Uint32Array<ArrayBuffer> {
  const n = end - start;
  const out = new Uint32Array(n * SPLAT_WORDS);
  const f = new Float32Array(out.buffer);
  for (let j = 0; j < n; j++) {
    const i = start + j;
    const o = j * SPLAT_WORDS;
    f[o] = d.positions[i * 3];
    f[o + 1] = d.positions[i * 3 + 1];
    f[o + 2] = d.positions[i * 3 + 2];
    out[o + 3] = packHalf2(d.scales[i * 3], d.scales[i * 3 + 1]);
    out[o + 4] = packHalf2(d.scales[i * 3 + 2], d.opacities[i]);
    out[o + 5] = packHalf2(d.rotations[i * 4], d.rotations[i * 4 + 1]);
    out[o + 6] = packHalf2(d.rotations[i * 4 + 2], d.rotations[i * 4 + 3]);
    out[o + 7] = 0;
  }
  return out;
}

export function packSH(d: SplatData, start = 0, end = d.loaded): Uint32Array<ArrayBuffer> {
  const floats = coeffCountForDegree(d.shDegree) * 3;
  const words = shWordsPerSplat(d.shDegree);
  const n = end - start;
  const out = new Uint32Array(n * words);
  for (let j = 0; j < n; j++) {
    const src = (start + j) * floats;
    const o = j * words;
    for (let w = 0; w < words; w++) {
      const a = d.sh[src + 2 * w];
      const b = 2 * w + 1 < floats ? d.sh[src + 2 * w + 1] : 0;
      out[o + w] = packHalf2(a, b);
    }
  }
  return out;
}

/** Byte size of the 192-byte preprocess uniform block (see preprocess.wgsl `Uniforms`). */
export const PREPROCESS_UNIFORM_BYTES = 192;

export interface PreprocessUniformValues {
  view: Float32Array;
  proj: Float32Array;
  camPos: [number, number, number];
  viewport: [number, number];
  splatCount: number;
  shDegree: number;
  shStride: number;
  frontToBack: boolean;
  scaleModifier: number;
  near: number;
}

export function writePreprocessUniforms(v: PreprocessUniformValues, out = new ArrayBuffer(PREPROCESS_UNIFORM_BYTES)): ArrayBuffer {
  const f = new Float32Array(out);
  const u = new Uint32Array(out);
  f.set(v.view, 0);
  f.set(v.proj, 16);
  f[32] = v.camPos[0];
  f[33] = v.camPos[1];
  f[34] = v.camPos[2];
  f[35] = 1;
  f[36] = v.viewport[0];
  f[37] = v.viewport[1];
  f[38] = v.proj[0] * v.viewport[0] * 0.5; // fx in pixels
  f[39] = v.proj[5] * v.viewport[1] * 0.5; // fy in pixels
  u[40] = v.splatCount;
  u[41] = v.shDegree;
  u[42] = v.shStride;
  u[43] = v.frontToBack ? 1 : 0;
  f[44] = v.scaleModifier;
  f[45] = v.near;
  f[46] = 0;
  f[47] = 0;
  return out;
}
