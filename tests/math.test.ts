import { describe, expect, it } from 'vitest';
import * as m4 from '../src/core/math/mat4';
import * as q from '../src/core/math/quat';
import type { Vec3 } from '../src/core/math/vec3';
import { mulberry32 } from '../src/io/procedural';

const close = (a: ArrayLike<number>, b: ArrayLike<number>, eps = 1e-5) => {
  expect(a.length).toBe(b.length);
  for (let i = 0; i < a.length; i++) expect(Math.abs(a[i] - b[i])).toBeLessThan(eps);
};

describe('mat4', () => {
  it('multiplies with identity and is column-major', () => {
    const t = m4.fromTranslation([1, 2, 3]);
    close(m4.multiply(m4.create(), t), t);
    close(m4.transformPoint(t, [1, 1, 1]), [2, 3, 4]);
    expect(m4.get(t, 0, 3)).toBe(1); // row 0, column 3 = tx
  });

  it('composes in the right order (a * b applies b first)', () => {
    const T = m4.fromTranslation([10, 0, 0]);
    const S = m4.fromScaling([2, 2, 2]);
    close(m4.transformPoint(m4.multiply(T, S), [1, 0, 0]), [12, 0, 0]);
    close(m4.transformPoint(m4.multiply(S, T), [1, 0, 0]), [22, 0, 0]);
  });

  it('inverts random affine matrices', () => {
    const r = mulberry32(7);
    for (let k = 0; k < 20; k++) {
      const rot = m4.fromQuat(q.normalize([r() - 0.5, r() - 0.5, r() - 0.5, r() - 0.5]));
      const M = m4.multiply(m4.fromTranslation([r() * 10, r() * -5, r()]), m4.multiply(rot, m4.fromScaling([1 + r(), 0.5 + r(), 2])));
      const inv = m4.invert(M)!;
      close(m4.multiply(M, inv), m4.create(), 1e-4);
    }
    expect(m4.invert(new Float32Array(16))).toBeNull();
  });

  it('perspectiveZO maps near to depth 0 and far to depth 1', () => {
    const P = m4.perspectiveZO(Math.PI / 3, 1.5, 0.1, 100);
    expect(m4.transformPoint(P, [0, 0, -0.1])[2]).toBeCloseTo(0, 5);
    expect(m4.transformPoint(P, [0, 0, -100])[2]).toBeCloseTo(1, 5);
    // A point on the top frustum plane maps to y_ndc = 1.
    const z = -5;
    const y = Math.tan(Math.PI / 6) * 5;
    expect(m4.transformPoint(P, [0, y, z])[1]).toBeCloseTo(1, 5);
    expect(m4.transformPoint(P, [y * 1.5, 0, z])[0]).toBeCloseTo(1, 5);
  });

  it('lookAt puts the eye at the origin looking down -Z', () => {
    const eye: Vec3 = [3, 4, 5];
    const target: Vec3 = [0, 1, 0];
    const V = m4.lookAt(eye, target, [0, 1, 0]);
    close(m4.transformPoint(V, eye), [0, 0, 0], 1e-5);
    const t = m4.transformPoint(V, target);
    expect(Math.abs(t[0])).toBeLessThan(1e-5);
    expect(Math.abs(t[1])).toBeLessThan(1e-5);
    expect(t[2]).toBeCloseTo(-Math.hypot(3, 3, 5), 4);
    // Rotation part is orthonormal.
    const R = m4.upper3x3RowMajor(V);
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) {
        const d = R[i * 3] * R[j * 3] + R[i * 3 + 1] * R[j * 3 + 1] + R[i * 3 + 2] * R[j * 3 + 2];
        expect(d).toBeCloseTo(i === j ? 1 : 0, 5);
      }
  });
});

describe('quat', () => {
  it('toMat3 of a 90 deg rotation about Z maps x to y', () => {
    const R = q.toMat3(q.fromAxisAngle([0, 0, 1], Math.PI / 2));
    close([R[0], R[3], R[6]], [0, 1, 0]);
  });

  it('matches mat4.fromQuat', () => {
    const qq = q.normalize([0.3, -0.2, 0.8, 0.1]);
    const R = q.toMat3(qq);
    const M = m4.fromQuat(qq);
    close(R, m4.upper3x3RowMajor(M));
  });

  it('fromUnitVectors rotates a onto b (including antiparallel)', () => {
    const r = mulberry32(3);
    for (let k = 0; k < 50; k++) {
      const a: Vec3 = [r() - 0.5, r() - 0.5, r() - 0.5];
      const b: Vec3 = k === 0 ? [-a[0], -a[1], -a[2]] : [r() - 0.5, r() - 0.5, r() - 0.5];
      const rq = q.fromUnitVectors(a, b);
      const ra = q.rotateVec3(rq, a);
      const la = Math.hypot(...a), lb = Math.hypot(...b);
      close(ra.map((v) => v / la), b.map((v) => v / lb), 1e-5);
    }
  });

  it('multiply composes rotations', () => {
    const a = q.fromAxisAngle([0, 1, 0], 0.4);
    const b = q.fromAxisAngle([1, 0, 0], -1.1);
    const v: Vec3 = [0.3, -0.7, 2];
    close(q.rotateVec3(q.multiply(a, b), v), q.rotateVec3(a, q.rotateVec3(b, v)), 1e-6);
  });
});
