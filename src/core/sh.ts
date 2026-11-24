/**
 * Real spherical harmonics up to degree 3, with the constants and sign convention of the
 * original 3D Gaussian Splatting code, so trained .ply files evaluate to the same colours.
 */
import type { Vec3 } from './math/vec3';

export const SH_C0 = 0.28209479177387814;
export const SH_C1 = 0.4886025119029199;
export const SH_C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396];
export const SH_C3 = [
  -0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
  1.445305721320277, -0.5900435899266435,
];

export const coeffCountForDegree = (deg: number): number => (deg + 1) * (deg + 1);

export function degreeForRestCount(restCount: number): number {
  // restCount = 3 * ((deg+1)^2 - 1)
  const perChannel = restCount / 3;
  for (let d = 0; d <= 3; d++) if (coeffCountForDegree(d) - 1 === perChannel) return d;
  throw new Error(`Unsupported number of f_rest_* properties: ${restCount}`);
}

/** The 16 signed basis values for a unit direction (entries above `degree` are 0). */
export function shBasis(degree: number, dirIn: Vec3): number[] {
  const l = Math.hypot(dirIn[0], dirIn[1], dirIn[2]) || 1;
  const x = dirIn[0] / l;
  const y = dirIn[1] / l;
  const z = dirIn[2] / l;
  const b = new Array<number>(16).fill(0);
  b[0] = SH_C0;
  if (degree > 0) {
    b[1] = -SH_C1 * y;
    b[2] = SH_C1 * z;
    b[3] = -SH_C1 * x;
  }
  if (degree > 1) {
    const xx = x * x, yy = y * y, zz = z * z;
    b[4] = SH_C2[0] * x * y;
    b[5] = SH_C2[1] * y * z;
    b[6] = SH_C2[2] * (2 * zz - xx - yy);
    b[7] = SH_C2[3] * x * z;
    b[8] = SH_C2[4] * (xx - yy);
    if (degree > 2) {
      b[9] = SH_C3[0] * y * (3 * xx - yy);
      b[10] = SH_C3[1] * x * y * z;
      b[11] = SH_C3[2] * y * (4 * zz - xx - yy);
      b[12] = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
      b[13] = SH_C3[4] * x * (4 * zz - xx - yy);
      b[14] = SH_C3[5] * z * (xx - yy);
      b[15] = SH_C3[6] * x * (xx - 3 * yy);
    }
  }
  return b;
}

/**
 * View-dependent colour. `coeffs` is laid out [k*3 + channel] for k < (degree+1)².
 * Returns max(0, Σ_k basis_k · c_k + 0.5), exactly like the 3DGS reference.
 */
export function evalSH(degree: number, coeffs: ArrayLike<number>, dir: Vec3): Vec3 {
  const b = shBasis(degree, dir);
  const n = coeffCountForDegree(degree);
  const out: Vec3 = [0.5, 0.5, 0.5];
  for (let k = 0; k < n; k++) {
    out[0] += b[k] * coeffs[k * 3];
    out[1] += b[k] * coeffs[k * 3 + 1];
    out[2] += b[k] * coeffs[k * 3 + 2];
  }
  return [Math.max(0, out[0]), Math.max(0, out[1]), Math.max(0, out[2])];
}

/** DC coefficient that produces the given base colour (0..1). */
export const colorToDC = (c: number): number => (c - 0.5) / SH_C0;
export const dcToColor = (dc: number): number => 0.5 + SH_C0 * dc;
