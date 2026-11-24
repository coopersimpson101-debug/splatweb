/**
 * Quaternions stored as [w, x, y, z] — the same order 3DGS uses for rot_0..rot_3.
 */
import type { Vec3 } from './vec3';
import { cross, dot, normalize as normalize3 } from './vec3';

export type Quat = [number, number, number, number];

export const identity = (): Quat => [1, 0, 0, 0];

export function normalize(q: Quat): Quat {
  const l = Math.hypot(q[0], q[1], q[2], q[3]);
  return l > 0 ? [q[0] / l, q[1] / l, q[2] / l, q[3] / l] : [1, 0, 0, 0];
}

/** Hamilton product a * b (rotate by b, then by a). */
export function multiply(a: Quat, b: Quat): Quat {
  const [aw, ax, ay, az] = a;
  const [bw, bx, by, bz] = b;
  return [
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ];
}

export function fromAxisAngle(axis: Vec3, angle: number): Quat {
  const a = normalize3(axis);
  const s = Math.sin(angle / 2);
  return [Math.cos(angle / 2), a[0] * s, a[1] * s, a[2] * s];
}

/** Shortest-arc rotation taking unit vector `from` onto unit vector `to`. */
export function fromUnitVectors(from: Vec3, to: Vec3): Quat {
  const f = normalize3(from);
  const t = normalize3(to);
  const d = dot(f, t);
  if (d < -0.999999) {
    // 180 degrees: pick any orthogonal axis.
    let axis = cross([1, 0, 0], f);
    if (Math.hypot(...axis) < 1e-6) axis = cross([0, 1, 0], f);
    return fromAxisAngle(axis, Math.PI);
  }
  const c = cross(f, t);
  return normalize([1 + d, c[0], c[1], c[2]]);
}

/** Row-major 3x3 rotation matrix of a (not necessarily unit) quaternion; normalizes first. */
export function toMat3(qIn: Quat): number[] {
  const [w, x, y, z] = normalize(qIn);
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  ];
}

export function rotateVec3(q: Quat, v: Vec3): Vec3 {
  const m = toMat3(q);
  return [
    m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
    m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
    m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
  ];
}
