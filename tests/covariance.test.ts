import { describe, expect, it } from 'vitest';
import {
  computeCov3D, conicFromCov2D, conicWeight, eigenvalues2, projectCov3D, projectSplat, projectionJacobian, quadAxes, sym3ToMat,
} from '../src/core/covariance';
import type { Sym2, Sym3 } from '../src/core/covariance';
import * as m4 from '../src/core/math/mat4';
import * as q from '../src/core/math/quat';
import type { Vec3 } from '../src/core/math/vec3';
import { mulberry32 } from '../src/io/procedural';
import { colorToDC } from '../src/core/sh';

const W = 800, H = 600;
const proj = m4.perspectiveZO(Math.PI / 3, W / H, 0.05, 100);
const fx = proj[0] * W / 2, fy = proj[5] * H / 2;

/** Exact (non-linearised) pixel projection of a view-space point, y up. */
function projectPix(V: m4.Mat4, p: Vec3): [number, number] {
  const c = m4.transformPoint(m4.multiply(proj, V), p);
  return [(c[0] * 0.5 + 0.5) * W, (c[1] * 0.5 + 0.5) * H];
}

describe('3D covariance', () => {
  it('is diag(s^2) for identity rotation', () => {
    const c = computeCov3D([1, 2, 3], [1, 0, 0, 0]);
    expect(c).toEqual([1, 0, 0, 4, 0, 9]);
  });

  it('swaps axes under a 90 degree rotation about Z', () => {
    const c = computeCov3D([1, 2, 3], q.fromAxisAngle([0, 0, 1], Math.PI / 2));
    const e = [4, 0, 0, 1, 0, 9];
    c.forEach((v, i) => expect(v).toBeCloseTo(e[i], 6));
  });

  it('equals R S S^T R^T for random inputs and is symmetric positive definite', () => {
    const r = mulberry32(11);
    for (let k = 0; k < 20; k++) {
      const rot = q.normalize([r() - 0.5, r() - 0.5, r() - 0.5, r() - 0.5]);
      const s: Vec3 = [0.1 + r(), 0.1 + r(), 0.1 + r()];
      const R = q.toMat3(rot);
      const ref = new Array(9).fill(0);
      for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++) for (let a = 0; a < 3; a++) ref[i * 3 + j] += R[i * 3 + a] * s[a] * s[a] * R[j * 3 + a];
      const got = sym3ToMat(computeCov3D(s, rot));
      got.forEach((v, i) => expect(v).toBeCloseTo(ref[i], 9));
      // trace is rotation invariant
      expect(got[0] + got[4] + got[8]).toBeCloseTo(s[0] ** 2 + s[1] ** 2 + s[2] ** 2, 9);
    }
  });
});

describe('EWA projection', () => {
  it('Jacobian matches finite differences of the exact projection', () => {
    const V = m4.create();
    const t: Vec3 = [0.4, -0.3, -3];
    const J = projectionJacobian(t, fx, fy);
    const h = 1e-4;
    for (let c = 0; c < 3; c++) {
      const tp: Vec3 = [...t];
      const tm: Vec3 = [...t];
      tp[c] += h;
      tm[c] -= h;
      const a = projectPix(V, tp), b = projectPix(V, tm);
      expect(J[c]).toBeCloseTo((a[0] - b[0]) / (2 * h), 0);
      expect(J[3 + c]).toBeCloseTo((a[1] - b[1]) / (2 * h), 0);
      expect(Math.abs(J[c] - (a[0] - b[0]) / (2 * h))).toBeLessThan(1e-2 * (1 + Math.abs(J[c])));
    }
  });

  it('isotropic splat at distance d projects to (f*s/d)^2 + 0.3 at the image centre', () => {
    const V = m4.create();
    const s = 0.05, d = 4;
    const c = projectCov3D(computeCov3D([s, s, s], [1, 0, 0, 0]), [0, 0, -d], { view: V, fx, fy, tanFovX: W / (2 * fx), tanFovY: H / (2 * fy) });
    expect(c[0]).toBeCloseTo((fx * s / d) ** 2 + 0.3, 6);
    expect(c[2]).toBeCloseTo((fy * s / d) ** 2 + 0.3, 6);
    expect(c[1]).toBeCloseTo(0, 9);
  });

  it('agrees with a Monte-Carlo estimate of the projected point distribution', () => {
    // Sample a small 3D Gaussian, push every sample through the exact (non-linear) projection, and
    // compare the empirical pixel covariance with the EWA linearisation.
    const r = mulberry32(5);
    const gauss = () => Math.sqrt(-2 * Math.log(Math.max(r(), 1e-12))) * Math.cos(2 * Math.PI * r());
    const V = m4.lookAt([0.5, 0.8, 4], [0, 0, 0], [0, 1, 0]);
    const rot = q.normalize([0.9, 0.2, -0.3, 0.25]);
    const scale: Vec3 = [0.06, 0.02, 0.035];
    const center: Vec3 = [0.3, -0.2, 0.1];
    const R = q.toMat3(rot);
    const N = 200000;
    let mx = 0, my = 0;
    const pts = new Float64Array(N * 2);
    for (let k = 0; k < N; k++) {
      const z = [gauss() * scale[0], gauss() * scale[1], gauss() * scale[2]];
      const p: Vec3 = [
        center[0] + R[0] * z[0] + R[1] * z[1] + R[2] * z[2],
        center[1] + R[3] * z[0] + R[4] * z[1] + R[5] * z[2],
        center[2] + R[6] * z[0] + R[7] * z[1] + R[8] * z[2],
      ];
      const px = projectPix(V, p);
      pts[2 * k] = px[0];
      pts[2 * k + 1] = px[1];
      mx += px[0];
      my += px[1];
    }
    mx /= N;
    my /= N;
    let sxx = 0, sxy = 0, syy = 0;
    for (let k = 0; k < N; k++) {
      const dx = pts[2 * k] - mx, dy = pts[2 * k + 1] - my;
      sxx += dx * dx;
      sxy += dx * dy;
      syy += dy * dy;
    }
    const emp: Sym2 = [sxx / N, sxy / N, syy / N];
    const t = m4.transformPoint(V, center);
    const ewa = projectCov3D(computeCov3D(scale, rot), t, { view: V, fx, fy, tanFovX: W / (2 * fx), tanFovY: H / (2 * fy) }, 0);
    const norm = Math.hypot(ewa[0], ewa[2]);
    for (let i = 0; i < 3; i++) expect(Math.abs(emp[i] - ewa[i]) / norm).toBeLessThan(0.02);
  });

  it('conic is the inverse of the 2D covariance', () => {
    const c: Sym2 = [12, 3, 5];
    const k = conicFromCov2D(c)!;
    // [[a b][b c]] * [[A B][B C]] = I
    expect(c[0] * k[0] + c[1] * k[1]).toBeCloseTo(1, 12);
    expect(c[0] * k[1] + c[1] * k[2]).toBeCloseTo(0, 12);
    expect(c[1] * k[1] + c[2] * k[2]).toBeCloseTo(1, 12);
    expect(conicFromCov2D([1, 2, 1])).toBeNull();
  });

  it('oriented-quad (uv) falloff equals the conic falloff at every pixel', () => {
    const r = mulberry32(9);
    for (let k = 0; k < 30; k++) {
      const a = 1 + 40 * r(), c = 1 + 40 * r();
      const b = (r() * 2 - 1) * Math.sqrt(a * c) * 0.95;
      const cov: Sym2 = [a, b, c];
      const conic = conicFromCov2D(cov)!;
      const { axis1, axis2 } = quadAxes(cov);
      // eigenvalues reproduce trace and determinant
      const [l1, l2] = eigenvalues2(cov);
      expect(l1 + l2).toBeCloseTo(a + c, 8);
      expect(l1 * l2).toBeCloseTo(a * c - b * b, 6);
      // axes are orthogonal
      expect(axis1[0] * axis2[0] + axis1[1] * axis2[1]).toBeCloseTo(0, 6);
      for (let s = 0; s < 10; s++) {
        const u = r() * 2 - 1, v = r() * 2 - 1;
        const dx = u * axis1[0] + v * axis2[0];
        const dy = u * axis1[1] + v * axis2[1];
        const quadW = Math.exp(-4.5 * (u * u + v * v));
        expect(quadW).toBeCloseTo(conicWeight(conic, dx, dy), 8);
      }
    }
  });
});

describe('projectSplat (CPU reference of preprocess.wgsl)', () => {
  const cam = {
    view: m4.lookAt([0, 0, 5], [0, 0, 0], [0, 1, 0]),
    proj,
    viewport: [W, H] as [number, number],
    camPos: [0, 0, 5] as Vec3,
    near: 0.05,
  };
  const base = { scale: [0.1, 0.1, 0.1] as Vec3, rotation: [1, 0, 0, 0] as q.Quat, opacity: 0.8, sh: [colorToDC(0.25), colorToDC(0.5), colorToDC(0.75)], shDegree: 0 };

  it('projects the origin to the viewport centre with the expected radius and colour', () => {
    const p = projectSplat({ ...base, position: [0, 0, 0] }, cam)!;
    expect(p.center[0]).toBeCloseTo(W / 2, 3);
    expect(p.center[1]).toBeCloseTo(H / 2, 3);
    expect(p.depth).toBeCloseTo(5, 5);
    const sigmaPx = Math.sqrt((fx * 0.1 / 5) ** 2 + 0.3);
    expect(p.radius).toBe(Math.ceil(3 * sigmaPx));
    p.color.forEach((v, i) => expect(v).toBeCloseTo([0.25, 0.5, 0.75][i], 6));
    expect(p.ndcZ).toBeGreaterThan(0);
    expect(p.ndcZ).toBeLessThan(1);
  });

  it('culls splats behind the camera, off-screen, or transparent', () => {
    expect(projectSplat({ ...base, position: [0, 0, 6] }, cam)).toBeNull();
    expect(projectSplat({ ...base, position: [100, 0, 0] }, cam)).toBeNull();
    expect(projectSplat({ ...base, position: [0, 0, 0], opacity: 0.001 }, cam)).toBeNull();
  });

  it('is invariant to the 1.3x frustum clamp for on-screen splats', () => {
    const p = projectSplat({ ...base, position: [1, 0.5, 0] }, cam)!;
    const t = m4.transformPoint(cam.view, [1, 0.5, 0]);
    const ref: Sym3 = computeCov3D([0.1, 0.1, 0.1], [1, 0, 0, 0]);
    const J = projectionJacobian(t, fx, fy);
    // For an isotropic Gaussian with identity view rotation: Σ' = s² J Jᵀ + 0.3
    const s2 = ref[0];
    expect(p.cov2d[0]).toBeCloseTo(s2 * (J[0] ** 2 + J[2] ** 2) + 0.3, 6);
    expect(p.cov2d[1]).toBeCloseTo(s2 * (J[0] * J[3] + J[2] * J[5]), 6);
  });
});
