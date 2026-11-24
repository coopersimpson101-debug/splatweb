/**
 * Deterministic procedural demo scene so the live demo works with zero downloads.
 * It deliberately exercises every part of the pipeline:
 *  - flat, strongly anisotropic splats (floor, cube faces) -> covariance + rotation handling;
 *  - tangent-aligned needles (torus knot) -> quaternion-from-direction;
 *  - non-zero SH bands 1..3 (iridescent knot, galaxy) -> view-dependent colour;
 *  - semi-transparent overlapping layers -> sorting + blending order.
 * World is Y-up, roughly centred at the origin, ~8 units across.
 */
import type { Quat } from '../core/math/quat';
import { fromUnitVectors, multiply as qmul, fromAxisAngle } from '../core/math/quat';
import type { Vec3 } from '../core/math/vec3';
import { colorToDC, coeffCountForDegree } from '../core/sh';
import type { SplatData } from './splatData';
import { createSplatData } from './splatData';

/** Small, fast, seedable PRNG (mulberry32). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hsv(h: number, s: number, v: number): Vec3 {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
  switch (((i % 6) + 6) % 6) {
    case 0: return [v, t, p];
    case 1: return [q, v, p];
    case 2: return [p, v, t];
    case 3: return [p, q, v];
    case 4: return [t, p, v];
    default: return [v, p, q];
  }
}

export interface DemoSceneOptions {
  count?: number;
  seed?: number;
}

export const DEMO_SH_DEGREE = 3;

export function generateDemoScene(opts: DemoSceneOptions = {}): SplatData {
  const total = Math.max(1000, Math.floor(opts.count ?? 150_000));
  const rand = mulberry32(opts.seed ?? 1234);
  const gauss = () => {
    // Box-Muller
    const u = Math.max(rand(), 1e-12);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
  };
  const data = createSplatData(total, DEMO_SH_DEGREE);
  const coeffs = coeffCountForDegree(DEMO_SH_DEGREE);
  const shStride = coeffs * 3;
  let i = 0;

  const put = (p: Vec3, s: Vec3, q: Quat, color: Vec3, opacity: number, sh?: (k: number, c: number) => number) => {
    if (i >= total) return;
    data.positions.set(p, i * 3);
    data.scales.set(s, i * 3);
    data.rotations.set(q, i * 4);
    data.opacities[i] = opacity;
    const base = i * shStride;
    for (let c = 0; c < 3; c++) data.sh[base + c] = colorToDC(color[c]);
    if (sh) for (let k = 1; k < coeffs; k++) for (let c = 0; c < 3; c++) data.sh[base + k * 3 + c] = sh(k, c);
    i++;
  };

  const nFloor = Math.floor(total * 0.34);
  const nKnot = Math.floor(total * 0.3);
  const nCube = Math.floor(total * 0.14);
  const nSphere = Math.floor(total * 0.08);
  const nGalaxy = total - nFloor - nKnot - nCube - nSphere;

  // 1) Floor: a checkerboard disc of flat splats (thin along Y).
  for (let k = 0; k < nFloor; k++) {
    const r = 4.2 * Math.sqrt(rand());
    const a = rand() * Math.PI * 2;
    const x = r * Math.cos(a), z = r * Math.sin(a);
    const checker = (Math.floor(x * 1.25 + 100) + Math.floor(z * 1.25 + 100)) % 2 === 0;
    const base: Vec3 = checker ? [0.82, 0.8, 0.76] : [0.2, 0.22, 0.27];
    const fade = Math.min(1, Math.max(0, (4.2 - r) / 1.0));
    const s = 0.018 + 0.01 * rand();
    put([x, -0.002 * rand(), z], [s, 0.002, s * (0.8 + 0.4 * rand())], fromAxisAngle([0, 1, 0], rand() * Math.PI),
      base, 0.9 * fade + 0.02);
  }

  // 2) Iridescent (2,3) torus knot made of tangent-aligned needles with SH bands 1-2.
  const knot = (t: number): Vec3 => {
    const p = 2, q = 3, R = 0.7, rr = 0.28;
    const c = Math.cos(q * t);
    return [(R + rr * c) * Math.cos(p * t), 1.35 + rr * Math.sin(q * t) * 1.6, (R + rr * c) * Math.sin(p * t)];
  };
  for (let k = 0; k < nKnot; k++) {
    const t = rand() * Math.PI * 2;
    const c0 = knot(t), c1 = knot(t + 1e-3);
    const tan = [c1[0] - c0[0], c1[1] - c0[1], c1[2] - c0[2]] as Vec3;
    const tube = 0.07;
    const p: Vec3 = [c0[0] + gauss() * tube * 0.5, c0[1] + gauss() * tube * 0.5, c0[2] + gauss() * tube * 0.5];
    const q = fromUnitVectors([1, 0, 0], tan);
    const col = hsv(t / (Math.PI * 2), 0.65, 0.95);
    put(p, [0.045, 0.01, 0.01], q, col, 0.75, (kk, ch) => {
      // Band 1: colour swings strongly with view direction; band 2: subtle sheen.
      if (kk === 1) return [0.0, 0.5, -0.6][ch];
      if (kk === 3) return [0.7, -0.2, 0.0][ch];
      if (kk === 2) return [-0.3, 0.0, 0.5][ch];
      if (kk >= 4 && kk <= 8) return 0.08 * Math.sin(kk * 1.7 + ch);
      return 0;
    });
  }

  // 3) A "lab crate": cube surface of flat splats oriented per face normal.
  const cubeC: Vec3 = [2.1, 0.45, -1.3];
  const half = 0.45;
  const faces: { n: Vec3; col: Vec3 }[] = [
    { n: [1, 0, 0], col: [0.9, 0.35, 0.2] }, { n: [-1, 0, 0], col: [0.9, 0.55, 0.2] },
    { n: [0, 1, 0], col: [0.95, 0.85, 0.6] }, { n: [0, -1, 0], col: [0.3, 0.3, 0.3] },
    { n: [0, 0, 1], col: [0.25, 0.6, 0.9] }, { n: [0, 0, -1], col: [0.3, 0.8, 0.5] },
  ];
  const yaw = fromAxisAngle([0, 1, 0], 0.5);
  for (let k = 0; k < nCube; k++) {
    const f = faces[k % 6];
    const u = (rand() * 2 - 1) * half, v = (rand() * 2 - 1) * half;
    let local: Vec3;
    if (f.n[0] !== 0) local = [f.n[0] * half, u, v];
    else if (f.n[1] !== 0) local = [u, f.n[1] * half, v];
    else local = [u, v, f.n[2] * half];
    // rotate by yaw
    const cy = Math.cos(0.5), sy = Math.sin(0.5);
    const rp: Vec3 = [cy * local[0] + sy * local[2], local[1], -sy * local[0] + cy * local[2]];
    const edge = Math.max(Math.abs(u), Math.abs(v)) > half * 0.9 ? 0.55 : 1;
    const q = qmul(yaw, fromUnitVectors([0, 0, 1], f.n));
    put([cubeC[0] + rp[0], cubeC[1] + rp[1], cubeC[2] + rp[2]], [0.03, 0.03, 0.002], q,
      [f.col[0] * edge, f.col[1] * edge, f.col[2] * edge], 0.95);
  }

  // 4) Glass-like translucent sphere with latitude stripes (tests blending of many layers).
  const sc: Vec3 = [-1.9, 0.75, 1.1];
  for (let k = 0; k < nSphere; k++) {
    const z = rand() * 2 - 1;
    const a = rand() * Math.PI * 2;
    const r = Math.sqrt(1 - z * z);
    const n: Vec3 = [r * Math.cos(a), z, r * Math.sin(a)];
    const stripe = Math.floor((z + 1) * 6) % 2 === 0;
    const col: Vec3 = stripe ? [0.95, 0.3, 0.45] : [0.95, 0.95, 1.0];
    put([sc[0] + n[0] * 0.7, sc[1] + n[1] * 0.7, sc[2] + n[2] * 0.7], [0.045, 0.045, 0.004],
      fromUnitVectors([0, 0, 1], n), col, stripe ? 0.8 : 0.25);
  }

  // 5) A small spiral galaxy floating above, with random band-3 SH twinkle.
  const tiltC = Math.cos(0.6), tiltS = Math.sin(0.6);
  for (let k = 0; k < nGalaxy; k++) {
    const arm = k % 3;
    const r = Math.pow(rand(), 0.7) * 1.6;
    const a = arm * (Math.PI * 2 / 3) + r * 2.4 + gauss() * 0.18;
    const gx = Math.cos(a) * r + gauss() * 0.03;
    const gy = gauss() * 0.05 * (1.7 - r);
    const gz = Math.sin(a) * r + gauss() * 0.03;
    // tilt the disc ~35 degrees towards the default camera
    const p: Vec3 = [gx, 3.0 + gy * tiltC - gz * tiltS, gy * tiltS + gz * tiltC];
    const core = Math.exp(-r * 2);
    const col = hsv(0.6 + 0.15 * rand() - 0.1 * core, 0.6 * (1 - core), 0.6 + 0.4 * core);
    const s = 0.008 + 0.02 * rand() + 0.05 * core;
    put(p, [s, s * 0.6, s], fromAxisAngle([0, 1, 0], a), col, 0.6 + 0.3 * rand(), (kk) =>
      kk >= 9 ? 0.15 * (rand() - 0.5) : 0);
  }

  data.loaded = i;
  return data;
}
