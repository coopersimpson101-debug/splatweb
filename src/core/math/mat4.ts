/**
 * Column-major 4x4 matrices (same memory layout as WGSL `mat4x4<f32>`).
 * Element (row r, column c) lives at index `c * 4 + r`.
 *
 * Conventions used throughout splatweb:
 *  - right-handed world, camera looks down -Z in view space, +Y is up;
 *  - projection maps view-space depth to clip z in [0, 1] (WebGPU convention).
 */
import type { Vec3 } from './vec3';
import { cross, normalize, sub } from './vec3';
import type { Quat } from './quat';

export type Mat4 = Float32Array<ArrayBuffer>;

export function create(): Mat4 {
  const m = new Float32Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

export function fromValues(colMajor: ArrayLike<number>): Mat4 {
  const m = new Float32Array(16);
  for (let i = 0; i < 16; i++) m[i] = colMajor[i];
  return m;
}

export const get = (m: Mat4, row: number, col: number): number => m[col * 4 + row];

/** out = a * b (apply b first, then a). Safe when out aliases a or b. */
export function multiply(a: Mat4, b: Mat4, out: Mat4 = new Float32Array(16)): Mat4 {
  const r = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let row = 0; row < 4; row++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + row] * b[c * 4 + k];
      r[c * 4 + row] = s;
    }
  }
  out.set(r);
  return out;
}

export function transpose(a: Mat4): Mat4 {
  const o = new Float32Array(16);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) o[c * 4 + r] = a[r * 4 + c];
  return o;
}

/** General 4x4 inverse via cofactors. Returns null for singular matrices. */
export function invert(a: Mat4): Mat4 | null {
  const [a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33] = a;
  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;
  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) return null;
  det = 1 / det;
  return fromValues([
    (a11 * b11 - a12 * b10 + a13 * b09) * det,
    (a02 * b10 - a01 * b11 - a03 * b09) * det,
    (a31 * b05 - a32 * b04 + a33 * b03) * det,
    (a22 * b04 - a21 * b05 - a23 * b03) * det,
    (a12 * b08 - a10 * b11 - a13 * b07) * det,
    (a00 * b11 - a02 * b08 + a03 * b07) * det,
    (a32 * b02 - a30 * b05 - a33 * b01) * det,
    (a20 * b05 - a22 * b02 + a23 * b01) * det,
    (a10 * b10 - a11 * b08 + a13 * b06) * det,
    (a01 * b08 - a00 * b10 - a03 * b06) * det,
    (a30 * b04 - a31 * b02 + a33 * b00) * det,
    (a21 * b02 - a20 * b04 - a23 * b00) * det,
    (a11 * b07 - a10 * b09 - a12 * b06) * det,
    (a00 * b09 - a01 * b07 + a02 * b06) * det,
    (a31 * b01 - a30 * b03 - a32 * b00) * det,
    (a20 * b03 - a21 * b01 + a22 * b00) * det,
  ]);
}

/**
 * Right-handed perspective projection with WebGPU depth range [0, 1].
 * near maps to z_ndc = 0 and far maps to z_ndc = 1.
 */
export function perspectiveZO(fovY: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovY / 2);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[11] = -1;
  if (Number.isFinite(far)) {
    const nf = 1 / (near - far);
    m[10] = far * nf;
    m[14] = far * near * nf;
  } else {
    m[10] = -1;
    m[14] = -near;
  }
  return m;
}

/** Right-handed look-at view matrix (world -> view). */
export function lookAt(eye: Vec3, target: Vec3, up: Vec3): Mat4 {
  const z = normalize(sub(eye, target)); // camera backward axis
  let x = normalize(cross(up, z));
  if (x[0] === 0 && x[1] === 0 && x[2] === 0) x = normalize(cross([0, 0, 1], z));
  const y = cross(z, x);
  return fromValues([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -(x[0] * eye[0] + x[1] * eye[1] + x[2] * eye[2]),
    -(y[0] * eye[0] + y[1] * eye[1] + y[2] * eye[2]),
    -(z[0] * eye[0] + z[1] * eye[1] + z[2] * eye[2]),
    1,
  ]);
}

export function fromTranslation(t: Vec3): Mat4 {
  const m = create();
  m[12] = t[0];
  m[13] = t[1];
  m[14] = t[2];
  return m;
}

export function fromScaling(s: Vec3): Mat4 {
  const m = create();
  m[0] = s[0];
  m[5] = s[1];
  m[10] = s[2];
  return m;
}

/** Rotation matrix from a unit quaternion stored as [w, x, y, z]. */
export function fromQuat(q: Quat): Mat4 {
  const [w, x, y, z] = q;
  return fromValues([
    1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y), 0,
    2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x), 0,
    2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y), 0,
    0, 0, 0, 1,
  ]);
}

/** Transforms a point (w = 1) and returns the homogeneous result [x, y, z, w]. */
export function transformPoint4(m: Mat4, p: Vec3): [number, number, number, number] {
  const [x, y, z] = p;
  return [
    m[0] * x + m[4] * y + m[8] * z + m[12],
    m[1] * x + m[5] * y + m[9] * z + m[13],
    m[2] * x + m[6] * y + m[10] * z + m[14],
    m[3] * x + m[7] * y + m[11] * z + m[15],
  ];
}

/** Transforms a point and divides by w. */
export function transformPoint(m: Mat4, p: Vec3): Vec3 {
  const r = transformPoint4(m, p);
  return [r[0] / r[3], r[1] / r[3], r[2] / r[3]];
}

/** Upper-left 3x3 as a row-major array [m00, m01, m02, m10, ...]. */
export function upper3x3RowMajor(m: Mat4): number[] {
  return [m[0], m[4], m[8], m[1], m[5], m[9], m[2], m[6], m[10]];
}

export function equalsApprox(a: Mat4, b: Mat4, eps = 1e-5): boolean {
  for (let i = 0; i < 16; i++) if (Math.abs(a[i] - b[i]) > eps) return false;
  return true;
}
