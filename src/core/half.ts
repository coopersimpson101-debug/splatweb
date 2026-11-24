/**
 * IEEE 754 binary16 ("half") conversion, implemented by hand so packed buffers are
 * bit-identical to what WGSL `unpack2x16float` expects.
 */
const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

/** float32 -> float16 bits, round-to-nearest-even, with subnormals, inf and NaN. */
export function f32ToF16(value: number): number {
  f32[0] = value;
  const x = u32[0];
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  let mant = x & 0x7fffff;

  if (exp === 0xff) {
    // Inf or NaN (keep a quiet NaN bit so NaN stays NaN).
    return sign | 0x7c00 | (mant ? 0x200 : 0);
  }
  // Re-bias exponent from 127 to 15.
  const e = exp - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00; // overflow -> inf
  if (e <= 0) {
    // Subnormal half (or zero).
    if (e < -10) return sign; // too small: rounds to signed zero
    mant |= 0x800000; // implicit leading 1
    const shift = 14 - e; // 14..24
    let half = mant >>> shift;
    const rem = mant & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    if (rem > halfway || (rem === halfway && (half & 1))) half++;
    return sign | half;
  }
  let half = (e << 10) | (mant >>> 13);
  const rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && (half & 1))) half++; // may carry into exponent: correct
  return sign | half;
}

/** float16 bits -> float32 value. */
export function f16ToF32(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >>> 10) & 0x1f;
  const mant = h & 0x3ff;
  if (exp === 0) return sign * mant * 2 ** -24;
  if (exp === 0x1f) return mant ? NaN : sign * Infinity;
  return sign * (1 + mant / 1024) * 2 ** (exp - 15);
}

/** Packs two floats into one u32 exactly like WGSL `pack2x16float(vec2(a, b))` (a in the low bits). */
export function packHalf2(a: number, b: number): number {
  return (f32ToF16(a) | (f32ToF16(b) << 16)) >>> 0;
}

export function unpackHalf2(u: number): [number, number] {
  return [f16ToF32(u & 0xffff), f16ToF32(u >>> 16)];
}
