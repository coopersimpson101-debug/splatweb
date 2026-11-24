/**
 * The compact ".splat" format popularised by antimatter15/splat: 32 bytes per splat, no header.
 *
 *   offset  0: position   3 x float32
 *   offset 12: scale      3 x float32 (linear, already exp'd)
 *   offset 24: colour     4 x uint8   (r, g, b = SH DC as 0..255; a = opacity 0..255)
 *   offset 28: rotation   4 x uint8   (w, x, y, z mapped from [-1, 1] via q*128 + 128)
 *
 * Degree-0 only: view-dependent SH is lost in this format.
 */
import { colorToDC, dcToColor } from '../core/sh';
import type { SplatData } from './splatData';
import { createSplatData } from './splatData';
import type { StreamParser } from './stream';

export const SPLAT_RECORD_BYTES = 32;

function decodeInto(out: SplatData, bytes: Uint8Array, start: number, count: number, first: number) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset + start, count * SPLAT_RECORD_BYTES);
  for (let k = 0; k < count; k++) {
    const b = k * SPLAT_RECORD_BYTES;
    const i = first + k;
    for (let a = 0; a < 3; a++) {
      out.positions[i * 3 + a] = dv.getFloat32(b + a * 4, true);
      out.scales[i * 3 + a] = dv.getFloat32(b + 12 + a * 4, true);
    }
    for (let c = 0; c < 3; c++) out.sh[i * 3 + c] = colorToDC(dv.getUint8(b + 24 + c) / 255);
    out.opacities[i] = dv.getUint8(b + 27) / 255;
    let w = (dv.getUint8(b + 28) - 128) / 128;
    let x = (dv.getUint8(b + 29) - 128) / 128;
    let y = (dv.getUint8(b + 30) - 128) / 128;
    let z = (dv.getUint8(b + 31) - 128) / 128;
    const l = Math.hypot(w, x, y, z) || 1;
    w /= l; x /= l; y /= l; z /= l;
    out.rotations[i * 4] = w;
    out.rotations[i * 4 + 1] = x;
    out.rotations[i * 4 + 2] = y;
    out.rotations[i * 4 + 3] = z;
  }
}

/**
 * Streaming .splat parser. When the total byte length is known (Content-Length / File.size)
 * the output is pre-allocated and fills progressively; otherwise chunks are buffered and
 * decoded at finish().
 */
export class SplatStreamParser implements StreamParser {
  private out: SplatData | null = null;
  private pending = new Uint8Array(0);
  private chunks: Uint8Array[] = [];
  private index = 0;

  constructor(totalBytes?: number) {
    if (totalBytes !== undefined) {
      if (totalBytes % SPLAT_RECORD_BYTES !== 0) throw new Error('.splat size is not a multiple of 32 bytes');
      this.out = createSplatData(totalBytes / SPLAT_RECORD_BYTES, 0);
    }
  }

  get data(): SplatData | null {
    return this.out;
  }

  get total(): number {
    return this.out?.count ?? 0;
  }

  push(chunk: Uint8Array): number {
    if (!this.out) {
      this.chunks.push(chunk.slice());
      return 0;
    }
    let buf = chunk;
    if (this.pending.length) {
      buf = new Uint8Array(this.pending.length + chunk.length);
      buf.set(this.pending);
      buf.set(chunk, this.pending.length);
    }
    const n = Math.min(Math.floor(buf.length / SPLAT_RECORD_BYTES), this.out.count - this.index);
    decodeInto(this.out, buf, 0, n, this.index);
    this.index += n;
    this.out.loaded = this.index;
    this.pending = buf.slice(n * SPLAT_RECORD_BYTES);
    return n;
  }

  finish(): SplatData {
    if (!this.out) {
      const len = this.chunks.reduce((s, c) => s + c.length, 0);
      if (len % SPLAT_RECORD_BYTES !== 0) throw new Error('.splat size is not a multiple of 32 bytes');
      const all = new Uint8Array(len);
      let o = 0;
      for (const c of this.chunks) {
        all.set(c, o);
        o += c.length;
      }
      this.chunks = [];
      this.out = createSplatData(len / SPLAT_RECORD_BYTES, 0);
      decodeInto(this.out, all, 0, this.out.count, 0);
      this.index = this.out.count;
      this.out.loaded = this.index;
    }
    if (this.index < this.out.count) throw new Error(`.splat stream truncated: ${this.index}/${this.out.count}`);
    return this.out;
  }
}

export function parseSplat(buffer: ArrayBuffer | Uint8Array): SplatData {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  const p = new SplatStreamParser(bytes.length);
  p.push(bytes);
  return p.finish();
}

const clampByte = (v: number) => Math.max(0, Math.min(255, Math.round(v)));

/** Encodes SplatData as .splat (drops SH above degree 0). */
export function writeSplat(d: SplatData): Uint8Array {
  const n = d.loaded;
  const out = new Uint8Array(n * SPLAT_RECORD_BYTES);
  const dv = new DataView(out.buffer);
  const shStride = (d.shDegree + 1) * (d.shDegree + 1) * 3;
  for (let i = 0; i < n; i++) {
    const b = i * SPLAT_RECORD_BYTES;
    for (let a = 0; a < 3; a++) {
      dv.setFloat32(b + a * 4, d.positions[i * 3 + a], true);
      dv.setFloat32(b + 12 + a * 4, d.scales[i * 3 + a], true);
    }
    for (let c = 0; c < 3; c++) dv.setUint8(b + 24 + c, clampByte(dcToColor(d.sh[i * shStride + c]) * 255));
    dv.setUint8(b + 27, clampByte(d.opacities[i] * 255));
    for (let k = 0; k < 4; k++) dv.setUint8(b + 28 + k, clampByte(d.rotations[i * 4 + k] * 128 + 128));
  }
  return out;
}
