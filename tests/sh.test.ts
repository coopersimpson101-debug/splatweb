import { describe, expect, it } from 'vitest';
import { SH_C0, SH_C1, colorToDC, dcToColor, degreeForRestCount, evalSH, shBasis } from '../src/core/sh';
import type { Vec3 } from '../src/core/math/vec3';

describe('spherical harmonics', () => {
  it('degree 0 reproduces the base colour from the DC term', () => {
    const coeffs = [colorToDC(0.2), colorToDC(0.6), colorToDC(0.9)];
    const c = evalSH(0, coeffs, [0.3, 0.1, -1]);
    expect(c[0]).toBeCloseTo(0.2, 6);
    expect(c[1]).toBeCloseTo(0.6, 6);
    expect(c[2]).toBeCloseTo(0.9, 6);
    expect(dcToColor(colorToDC(0.42))).toBeCloseTo(0.42, 12);
  });

  it('matches hand-computed degree-1 values (3DGS sign convention)', () => {
    // coefficient for basis 1 (-C1*y) on red, basis 2 (+C1*z) on green, basis 3 (-C1*x) on blue
    const coeffs = new Array(12).fill(0);
    coeffs[1 * 3 + 0] = 1;
    coeffs[2 * 3 + 1] = 1;
    coeffs[3 * 3 + 2] = 1;
    const dir: Vec3 = [0, 1, 0];
    const c = evalSH(1, coeffs, dir);
    expect(c[0]).toBeCloseTo(Math.max(0, 0.5 - SH_C1), 9);
    expect(c[1]).toBeCloseTo(0.5, 9);
    expect(c[2]).toBeCloseTo(0.5, 9);
    const c2 = evalSH(1, coeffs, [-2, 0, 0]); // normalised internally
    expect(c2[2]).toBeCloseTo(0.5 + SH_C1, 9);
  });

  it('clamps negative colours to zero', () => {
    expect(evalSH(0, [-10, -10, -10], [0, 0, 1])).toEqual([0, 0, 0]);
  });

  it('basis functions are orthonormal on the sphere (numerical quadrature)', () => {
    // Gauss-Legendre in cos(theta) x uniform in phi, exact for these polynomial degrees.
    const nodes = [-0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498,
      0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975363];
    const weights = [0.1012285362903763, 0.2223810344533745, 0.3137066458778873, 0.3626837833783620,
      0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763];
    const nPhi = 16;
    const G = Array.from({ length: 16 }, () => new Array(16).fill(0));
    for (let a = 0; a < nodes.length; a++) {
      const z = nodes[a];
      const r = Math.sqrt(1 - z * z);
      for (let p = 0; p < nPhi; p++) {
        const phi = (2 * Math.PI * (p + 0.5)) / nPhi;
        const b = shBasis(3, [r * Math.cos(phi), r * Math.sin(phi), z]);
        const w = weights[a] * ((2 * Math.PI) / nPhi);
        for (let i = 0; i < 16; i++) for (let j = 0; j < 16; j++) G[i][j] += w * b[i] * b[j];
      }
    }
    for (let i = 0; i < 16; i++)
      for (let j = 0; j < 16; j++) expect(G[i][j], `<Y${i},Y${j}>`).toBeCloseTo(i === j ? 1 : 0, 9);
    expect(SH_C0).toBeCloseTo(1 / (2 * Math.sqrt(Math.PI)), 12);
  });

  it('infers degree from the number of f_rest properties', () => {
    expect(degreeForRestCount(0)).toBe(0);
    expect(degreeForRestCount(9)).toBe(1);
    expect(degreeForRestCount(24)).toBe(2);
    expect(degreeForRestCount(45)).toBe(3);
    expect(() => degreeForRestCount(10)).toThrow();
  });
});
