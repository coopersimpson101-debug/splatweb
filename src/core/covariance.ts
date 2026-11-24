/**
 * CPU reference implementation of the per-splat math that `preprocess.wgsl` runs on the GPU.
 * Every function here is mirrored line-for-line in WGSL, and the unit tests (plus the optional
 * Dawn-in-Node GPU test) check the two against each other.
 *
 * Notation (see README "How it works"):
 *   Σ   = R S Sᵀ Rᵀ                         3D covariance from scale S and rotation R
 *   t   = W p + w                           view-space centre (camera looks down -Z)
 *   d   = -t.z                              positive depth
 *   J   = [[fx/d, 0, fx·tx/d²],             Jacobian of the perspective projection
 *          [0, fy/d, fy·ty/d²]]             (pixels per view-space unit)
 *   Σ'  = J W Σ Wᵀ Jᵀ + 0.3·I               EWA screen-space covariance + low-pass filter
 *   conic = Σ'⁻¹,  radius = 3·sqrt(λ_max)
 */
import type { Mat4 } from './math/mat4';
import type { Quat } from './math/quat';
import { toMat3 } from './math/quat';
import type { Vec3 } from './math/vec3';
import { evalSH } from './sh';

/** Symmetric 3x3 stored as [xx, xy, xz, yy, yz, zz]. */
export type Sym3 = [number, number, number, number, number, number];
/** Symmetric 2x2 stored as [a, b, c] meaning [[a, b], [b, c]]. */
export type Sym2 = [number, number, number];

/** Low-pass filter added to the 2D covariance diagonal (pixels²), as in the 3DGS reference. */
export const LOW_PASS = 0.3;
/** Splats are drawn out to this many standard deviations. */
export const SIGMA_EXTENT = 3;
/** Smallest alpha that contributes (1/255). */
export const ALPHA_CUTOFF = 1 / 255;

export function computeCov3D(scale: Vec3, rot: Quat): Sym3 {
  const R = toMat3(rot); // row-major
  // M = R * S
  const M = [
    R[0] * scale[0], R[1] * scale[1], R[2] * scale[2],
    R[3] * scale[0], R[4] * scale[1], R[5] * scale[2],
    R[6] * scale[0], R[7] * scale[1], R[8] * scale[2],
  ];
  // Σ = M Mᵀ
  const row = (i: number, j: number) =>
    M[i * 3] * M[j * 3] + M[i * 3 + 1] * M[j * 3 + 1] + M[i * 3 + 2] * M[j * 3 + 2];
  return [row(0, 0), row(0, 1), row(0, 2), row(1, 1), row(1, 2), row(2, 2)];
}

export function sym3ToMat(s: Sym3): number[] {
  return [s[0], s[1], s[2], s[1], s[3], s[4], s[2], s[4], s[5]];
}

export interface ProjectionParams {
  /** World(model)->view matrix, column-major. */
  view: Mat4;
  fx: number;
  fy: number;
  /** tan(fov/2) per axis, used to clamp the Jacobian like the reference implementation. */
  tanFovX: number;
  tanFovY: number;
}

/** Row-major 2x3 Jacobian of pixel coordinates (y up) w.r.t. the view-space point t. */
export function projectionJacobian(t: Vec3, fx: number, fy: number): number[] {
  const d = -t[2];
  return [fx / d, 0, (fx * t[0]) / (d * d), 0, fy / d, (fy * t[1]) / (d * d)];
}

/**
 * EWA projection of a 3D covariance into a 2D screen-space covariance (pixels², y up),
 * including the 0.3 px² low-pass term.
 */
export function projectCov3D(cov: Sym3, t: Vec3, p: ProjectionParams, lowPass = LOW_PASS): Sym2 {
  const d = -t[2];
  // Clamp the point used for the Jacobian to 1.3x the frustum, like the 3DGS rasterizer,
  // to keep huge off-screen splats from producing unstable covariances.
  const limX = 1.3 * p.tanFovX;
  const limY = 1.3 * p.tanFovY;
  const tx = Math.min(limX, Math.max(-limX, t[0] / d)) * d;
  const ty = Math.min(limY, Math.max(-limY, t[1] / d)) * d;
  const J = projectionJacobian([tx, ty, t[2]], p.fx, p.fy);
  const v = p.view;
  // W: upper-left 3x3 of view, row-major.
  const W = [v[0], v[4], v[8], v[1], v[5], v[9], v[2], v[6], v[10]];
  // T = J W (2x3)
  const T = new Array<number>(6);
  for (let r = 0; r < 2; r++)
    for (let c = 0; c < 3; c++)
      T[r * 3 + c] = J[r * 3] * W[c] + J[r * 3 + 1] * W[3 + c] + J[r * 3 + 2] * W[6 + c];
  const S = sym3ToMat(cov);
  // TS = T Σ (2x3)
  const TS = new Array<number>(6);
  for (let r = 0; r < 2; r++)
    for (let c = 0; c < 3; c++)
      TS[r * 3 + c] = T[r * 3] * S[c] + T[r * 3 + 1] * S[3 + c] + T[r * 3 + 2] * S[6 + c];
  const dot3 = (a: number[], ai: number, b: number[], bi: number) =>
    a[ai] * b[bi] + a[ai + 1] * b[bi + 1] + a[ai + 2] * b[bi + 2];
  return [dot3(TS, 0, T, 0) + lowPass, dot3(TS, 0, T, 3), dot3(TS, 3, T, 3) + lowPass];
}

/** Inverse of a symmetric 2x2 ("conic"), or null when not positive definite. */
export function conicFromCov2D(c: Sym2): Sym2 | null {
  const det = c[0] * c[2] - c[1] * c[1];
  if (!(det > 0)) return null;
  const inv = 1 / det;
  return [c[2] * inv, -c[1] * inv, c[0] * inv];
}

/** Eigenvalues (λ1 ≥ λ2) of a symmetric 2x2. */
export function eigenvalues2(c: Sym2): [number, number] {
  const mid = 0.5 * (c[0] + c[2]);
  const det = c[0] * c[2] - c[1] * c[1];
  // The 3DGS rasterizer floors this at 0.1 because it only needs a conservative radius; we use
  // the exact value because the eigenvectors also define the oriented quad.
  const disc = Math.sqrt(Math.max(0, mid * mid - det));
  return [mid + disc, Math.max(mid - disc, 0)];
}

/**
 * Oriented quad axes for a 2D Gaussian: axis1 is the major eigenvector scaled by 3·sqrt(λ1),
 * axis2 the minor one scaled by 3·sqrt(λ2). A fragment at quad coordinate uv ∈ [-1,1]² then has
 * Mahalanobis distance² = 9·|uv|² — no conic needed in the fragment shader.
 */
export function quadAxes(c: Sym2): { axis1: [number, number]; axis2: [number, number]; radius: number } {
  const [l1, l2] = eigenvalues2(c);
  let v: [number, number];
  if (Math.abs(c[1]) < 1e-9) v = c[0] >= c[2] ? [1, 0] : [0, 1];
  else {
    const vx = c[1];
    const vy = l1 - c[0];
    const n = Math.hypot(vx, vy);
    v = [vx / n, vy / n];
  }
  const r1 = SIGMA_EXTENT * Math.sqrt(l1);
  const r2 = SIGMA_EXTENT * Math.sqrt(l2);
  return { axis1: [v[0] * r1, v[1] * r1], axis2: [-v[1] * r2, v[0] * r2], radius: Math.ceil(r1) };
}

export interface ProjectInput {
  position: Vec3;
  scale: Vec3;
  rotation: Quat;
  opacity: number;
  /** SH coefficients laid out [k*3 + channel]. */
  sh: ArrayLike<number>;
  shDegree: number;
}

export interface CameraInput {
  view: Mat4;
  proj: Mat4;
  viewport: [number, number];
  /** Camera position in the same (model) space as the splat positions. */
  camPos: Vec3;
  near: number;
  scaleModifier?: number;
}

export interface Projected {
  center: [number, number];
  axis1: [number, number];
  axis2: [number, number];
  color: Vec3;
  opacity: number;
  /** Positive view-space depth. */
  depth: number;
  /** clip.z / clip.w (WebGPU depth). */
  ndcZ: number;
  cov2d: Sym2;
  conic: Sym2;
  radius: number;
}

/** Full CPU reference for one splat; mirrors `preprocess.wgsl`. Returns null when culled. */
export function projectSplat(s: ProjectInput, cam: CameraInput): Projected | null {
  if (s.opacity < ALPHA_CUTOFF) return null;
  const v = cam.view;
  const [x, y, z] = s.position;
  const t: Vec3 = [
    v[0] * x + v[4] * y + v[8] * z + v[12],
    v[1] * x + v[5] * y + v[9] * z + v[13],
    v[2] * x + v[6] * y + v[10] * z + v[14],
  ];
  const d = -t[2];
  if (d < cam.near) return null;
  const P = cam.proj;
  const cx = P[0] * t[0] + P[4] * t[1] + P[8] * t[2] + P[12];
  const cy = P[1] * t[0] + P[5] * t[1] + P[9] * t[2] + P[13];
  const cz = P[2] * t[0] + P[6] * t[1] + P[10] * t[2] + P[14];
  const cw = P[3] * t[0] + P[7] * t[1] + P[11] * t[2] + P[15];
  const [W, H] = cam.viewport;
  const center: [number, number] = [((cx / cw) * 0.5 + 0.5) * W, ((cy / cw) * 0.5 + 0.5) * H];
  const fx = P[0] * W * 0.5;
  const fy = P[5] * H * 0.5;
  const sm = cam.scaleModifier ?? 1;
  const cov3 = computeCov3D([s.scale[0] * sm, s.scale[1] * sm, s.scale[2] * sm], s.rotation);
  const cov2d = projectCov3D(cov3, t, { view: v, fx, fy, tanFovX: W / (2 * fx), tanFovY: H / (2 * fy) });
  const conic = conicFromCov2D(cov2d);
  if (!conic) return null;
  const { axis1, axis2, radius } = quadAxes(cov2d);
  if (center[0] + radius < 0 || center[0] - radius > W || center[1] + radius < 0 || center[1] - radius > H)
    return null;
  const dir: Vec3 = [x - cam.camPos[0], y - cam.camPos[1], z - cam.camPos[2]];
  const color = evalSH(s.shDegree, s.sh, dir);
  return { center, axis1, axis2, color, opacity: s.opacity, depth: d, ndcZ: cz / cw, cov2d, conic, radius };
}

/** Gaussian weight at pixel offset (dx, dy) from the centre using the conic form. */
export function conicWeight(conic: Sym2, dx: number, dy: number): number {
  const power = -0.5 * (conic[0] * dx * dx + conic[2] * dy * dy) - conic[1] * dx * dy;
  return Math.exp(power);
}
