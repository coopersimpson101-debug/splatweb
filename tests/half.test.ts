import { describe, expect, it } from 'vitest';
import { f16ToF32, f32ToF16, packHalf2, unpackHalf2 } from '../src/core/half';

describe('float16', () => {
  it('encodes reference bit patterns', () => {
    const cases: [number, number][] = [
      [0, 0x0000], [-0, 0x8000], [1, 0x3c00], [-2, 0xc000], [0.5, 0x3800],
      [65504, 0x7bff], [65520, 0x7c00] /* rounds to inf */, [Infinity, 0x7c00], [-Infinity, 0xfc00],
      [2 ** -14, 0x0400] /* min normal */, [2 ** -24, 0x0001] /* min subnormal */, [2 ** -25, 0x0000] /* tie -> even (0) */,
      [0.333251953125, 0x3555], [1 / 3, 0x3555], [1.0009765625, 0x3c01], [1.00048828125, 0x3c00] /* tie -> even */,
      [1.00146484375, 0x3c02] /* tie -> even (up) */, [6.103515625e-5 * 0.5, 0x0200],
    ];
    for (const [v, bits] of cases) expect(f32ToF16(v), `value ${v}`).toBe(bits);
    expect(f32ToF16(NaN) & 0x7c00).toBe(0x7c00);
    expect(f32ToF16(NaN) & 0x3ff).not.toBe(0);
  });

  it('round-trips every half bit pattern exactly', () => {
    for (let h = 0; h < 0x10000; h++) {
      const v = f16ToF32(h);
      if (Number.isNaN(v)) continue;
      expect(f32ToF16(v)).toBe(h);
    }
  });

  it('has bounded relative error in the normal range', () => {
    for (let k = 0; k < 2000; k++) {
      const v = Math.exp((Math.random() - 0.5) * 18);
      const r = f16ToF32(f32ToF16(v));
      expect(Math.abs(r - v) / v).toBeLessThanOrEqual(2 ** -11 + 1e-9);
    }
  });

  it('packs two halves like WGSL pack2x16float (first in low bits)', () => {
    const u = packHalf2(1, -2);
    expect(u).toBe((0xc000 << 16 | 0x3c00) >>> 0);
    expect(unpackHalf2(u)).toEqual([1, -2]);
  });
});
