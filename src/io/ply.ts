/**
 * Streaming parser for the standard 3D Gaussian Splatting .ply layout (Inria / gsplat /
 * nerfstudio exports):
 *
 *   x y z [nx ny nz] f_dc_0..2 f_rest_0..(3·((d+1)²-1)-1) opacity scale_0..2 rot_0..3
 *
 * Properties may appear in any order and with any PLY scalar type. Plain point clouds
 * (x y z + red green blue, e.g. COLMAP's points3D.ply) are accepted too and turned into
 * small isotropic splats.
 *
 * Activations applied while parsing: scale = exp(scale_i), opacity = sigmoid(opacity),
 * rotation normalised. f_rest is stored channel-major in the file (all R, then G, then B)
 * and is re-interleaved to [k][rgb].
 */
import { colorToDC, coeffCountForDegree, degreeForRestCount } from '../core/sh';
import type { SplatData } from './splatData';
import { createSplatData } from './splatData';
import type { StreamParser } from './stream';

type PlyType = 'int8' | 'uint8' | 'int16' | 'uint16' | 'int32' | 'uint32' | 'float32' | 'float64';

const TYPE_ALIASES: Record<string, PlyType> = {
  char: 'int8', int8: 'int8', uchar: 'uint8', uint8: 'uint8',
  short: 'int16', int16: 'int16', ushort: 'uint16', uint16: 'uint16',
  int: 'int32', int32: 'int32', uint: 'uint32', uint32: 'uint32',
  float: 'float32', float32: 'float32', double: 'float64', float64: 'float64',
};
const TYPE_SIZE: Record<PlyType, number> = {
  int8: 1, uint8: 1, int16: 2, uint16: 2, int32: 4, uint32: 4, float32: 4, float64: 8,
};

export interface PlyProperty {
  name: string;
  type: PlyType;
  offset: number;
}

export interface PlyHeader {
  format: 'binary_little_endian' | 'binary_big_endian';
  vertexCount: number;
  properties: PlyProperty[];
  stride: number;
  /** Byte length of the header including the trailing newline after end_header. */
  headerLength: number;
  comments: string[];
}

const MAX_HEADER = 64 * 1024;

/** Returns the parsed header, or null if `end_header` has not arrived yet. */
export function parsePlyHeader(bytes: Uint8Array): PlyHeader | null {
  const limit = Math.min(bytes.length, MAX_HEADER);
  const text = new TextDecoder('latin1').decode(bytes.subarray(0, limit));
  const m = /end_header\r?\n/.exec(text);
  if (!m) {
    if (bytes.length >= MAX_HEADER) throw new Error('PLY header too large or missing end_header');
    if (bytes.length >= 4 && !text.startsWith('ply')) throw new Error('Not a PLY file (missing "ply" magic)');
    return null;
  }
  const headerLength = m.index + m[0].length;
  const lines = text.slice(0, m.index).split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  if (lines[0] !== 'ply') throw new Error('Not a PLY file (missing "ply" magic)');
  let format: PlyHeader['format'] | null = null;
  let vertexCount = -1;
  let inVertex = false;
  let seenOtherElementBeforeVertex = false;
  const properties: PlyProperty[] = [];
  const comments: string[] = [];
  let stride = 0;
  for (const line of lines.slice(1)) {
    const parts = line.split(/\s+/);
    switch (parts[0]) {
      case 'format':
        if (parts[1] === 'ascii') throw new Error('ASCII PLY is not supported; re-export as binary_little_endian');
        if (parts[1] !== 'binary_little_endian' && parts[1] !== 'binary_big_endian')
          throw new Error(`Unknown PLY format ${parts[1]}`);
        format = parts[1];
        break;
      case 'comment':
      case 'obj_info':
        comments.push(parts.slice(1).join(' '));
        break;
      case 'element':
        if (parts[1] === 'vertex') {
          if (seenOtherElementBeforeVertex) throw new Error('PLY elements before "vertex" are not supported');
          vertexCount = parseInt(parts[2], 10);
          inVertex = true;
        } else {
          if (vertexCount < 0) seenOtherElementBeforeVertex = true;
          inVertex = false; // later elements (faces...) are ignored: we stop after the vertices
        }
        break;
      case 'property':
        if (!inVertex) break;
        if (parts[1] === 'list') throw new Error('List properties on vertices are not supported');
        {
          const type = TYPE_ALIASES[parts[1]];
          if (!type) throw new Error(`Unknown PLY property type ${parts[1]}`);
          properties.push({ name: parts[2], type, offset: stride });
          stride += TYPE_SIZE[type];
        }
        break;
      default:
        break;
    }
  }
  if (!format) throw new Error('PLY header has no format line');
  if (vertexCount < 0) throw new Error('PLY has no vertex element');
  return { format, vertexCount, properties, stride, headerLength, comments };
}

type Reader = (dv: DataView, offset: number) => number;

function makeReader(p: PlyProperty, little: boolean): Reader {
  const o = p.offset;
  switch (p.type) {
    case 'int8': return (dv, b) => dv.getInt8(b + o);
    case 'uint8': return (dv, b) => dv.getUint8(b + o);
    case 'int16': return (dv, b) => dv.getInt16(b + o, little);
    case 'uint16': return (dv, b) => dv.getUint16(b + o, little);
    case 'int32': return (dv, b) => dv.getInt32(b + o, little);
    case 'uint32': return (dv, b) => dv.getUint32(b + o, little);
    case 'float32': return (dv, b) => dv.getFloat32(b + o, little);
    case 'float64': return (dv, b) => dv.getFloat64(b + o, little);
  }
}

export interface PlyParseOptions {
  /** Scale used for point-cloud PLYs without scale_* properties (world units). */
  defaultPointScale?: number;
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

export class PlyStreamParser implements StreamParser {
  private pending: Uint8Array = new Uint8Array(0);
  private header: PlyHeader | null = null;
  private out: SplatData | null = null;
  private vertexIndex = 0;
  private decode!: (dv: DataView, base: number, i: number) => void;

  constructor(private readonly options: PlyParseOptions = {}) {}

  get data(): SplatData | null {
    return this.out;
  }

  get total(): number {
    return this.header?.vertexCount ?? 0;
  }

  get plyHeader(): PlyHeader | null {
    return this.header;
  }

  /** Feeds bytes; returns the number of new splats that became available. */
  push(chunk: Uint8Array): number {
    if (this.header && this.vertexIndex >= this.header.vertexCount) return 0; // ignore trailing elements
    let buf = chunk;
    if (this.pending.length) {
      buf = new Uint8Array(this.pending.length + chunk.length);
      buf.set(this.pending, 0);
      buf.set(chunk, this.pending.length);
    }
    let pos = 0;
    if (!this.header) {
      const h = parsePlyHeader(buf);
      if (!h) {
        this.pending = buf;
        return 0;
      }
      this.header = h;
      this.setup(h);
      pos = h.headerLength;
    }
    const h = this.header;
    const remaining = h.vertexCount - this.vertexIndex;
    const available = Math.min(remaining, Math.floor((buf.length - pos) / h.stride));
    if (available > 0) {
      const dv = new DataView(buf.buffer, buf.byteOffset + pos, available * h.stride);
      for (let k = 0; k < available; k++) this.decode(dv, k * h.stride, this.vertexIndex + k);
      this.vertexIndex += available;
      pos += available * h.stride;
      this.out!.loaded = this.vertexIndex;
    }
    // Copy (not view) the leftover so the caller can reuse its chunk buffer.
    this.pending = buf.slice(pos);
    return available;
  }

  finish(): SplatData {
    if (!this.header || !this.out) throw new Error('PLY stream ended before the header was complete');
    if (this.vertexIndex < this.header.vertexCount)
      throw new Error(`PLY stream truncated: got ${this.vertexIndex} of ${this.header.vertexCount} vertices`);
    return this.out;
  }

  private setup(h: PlyHeader) {
    const little = h.format === 'binary_little_endian';
    const byName = new Map(h.properties.map((p) => [p.name, p]));
    const rd = (name: string): Reader | null => {
      const p = byName.get(name);
      return p ? makeReader(p, little) : null;
    };
    const need = (name: string): Reader => {
      const r = rd(name);
      if (!r) throw new Error(`PLY is missing required property "${name}"`);
      return r;
    };
    const px = need('x'), py = need('y'), pz = need('z');
    const restNames = h.properties.map((p) => p.name).filter((n) => /^f_rest_\d+$/.test(n));
    const degree = degreeForRestCount(restNames.length);
    const coeffs = coeffCountForDegree(degree);
    const perChannel = coeffs - 1;
    const rest: Reader[] = [];
    for (let i = 0; i < restNames.length; i++) rest.push(need(`f_rest_${i}`));
    const dc = [rd('f_dc_0'), rd('f_dc_1'), rd('f_dc_2')];
    const rgb = [rd('red'), rd('green'), rd('blue')];
    const rgbIsByte = ['red', 'green', 'blue'].map((n) => {
      const t = byName.get(n)?.type;
      return t === 'uint8' || t === 'int8';
    });
    const op = rd('opacity');
    const sc = [rd('scale_0'), rd('scale_1'), rd('scale_2')];
    const rot = [rd('rot_0'), rd('rot_1'), rd('rot_2'), rd('rot_3')];
    const hasScale = sc.every(Boolean);
    const hasRot = rot.every(Boolean);
    const hasDC = dc.every(Boolean);
    const hasRGB = rgb.every(Boolean);
    const defaultScale = this.options.defaultPointScale ?? 0.01;

    const out = createSplatData(h.vertexCount, degree);
    this.out = out;
    const { positions, scales, rotations, opacities, sh } = out;
    const shStride = coeffs * 3;

    this.decode = (dv, b, i) => {
      positions[i * 3] = px(dv, b);
      positions[i * 3 + 1] = py(dv, b);
      positions[i * 3 + 2] = pz(dv, b);
      if (hasScale) {
        scales[i * 3] = Math.exp(sc[0]!(dv, b));
        scales[i * 3 + 1] = Math.exp(sc[1]!(dv, b));
        scales[i * 3 + 2] = Math.exp(sc[2]!(dv, b));
      } else {
        scales[i * 3] = scales[i * 3 + 1] = scales[i * 3 + 2] = defaultScale;
      }
      if (hasRot) {
        const w = rot[0]!(dv, b), x = rot[1]!(dv, b), y = rot[2]!(dv, b), z = rot[3]!(dv, b);
        const l = Math.hypot(w, x, y, z) || 1;
        rotations[i * 4] = w / l;
        rotations[i * 4 + 1] = x / l;
        rotations[i * 4 + 2] = y / l;
        rotations[i * 4 + 3] = z / l;
      } else {
        rotations[i * 4] = 1;
        rotations[i * 4 + 1] = rotations[i * 4 + 2] = rotations[i * 4 + 3] = 0;
      }
      opacities[i] = op ? sigmoid(op(dv, b)) : 1;
      const s = i * shStride;
      if (hasDC) {
        sh[s] = dc[0]!(dv, b);
        sh[s + 1] = dc[1]!(dv, b);
        sh[s + 2] = dc[2]!(dv, b);
      } else if (hasRGB) {
        for (let c = 0; c < 3; c++) {
          const v = rgb[c]!(dv, b);
          sh[s + c] = colorToDC(rgbIsByte[c] ? v / 255 : v);
        }
      } else {
        sh[s] = sh[s + 1] = sh[s + 2] = colorToDC(0.8);
      }
      for (let k = 1; k < coeffs; k++)
        for (let c = 0; c < 3; c++) sh[s + k * 3 + c] = rest[c * perChannel + (k - 1)](dv, b);
    };
  }
}

/** Convenience: parse a whole PLY buffer at once. */
export function parsePly(buffer: ArrayBuffer | Uint8Array, options?: PlyParseOptions): SplatData {
  const p = new PlyStreamParser(options);
  p.push(buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer));
  return p.finish();
}

/**
 * Writes a SplatData as a standard 3DGS binary little-endian .ply (inverse activations
 * applied), readable by other 3DGS tools. Used by tests and the "export" feature.
 */
export function writePly(d: SplatData): Uint8Array {
  const n = d.loaded;
  const coeffs = coeffCountForDegree(d.shDegree);
  const restCount = (coeffs - 1) * 3;
  const names = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2'];
  for (let i = 0; i < restCount; i++) names.push(`f_rest_${i}`);
  names.push('opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3');
  const header =
    `ply\nformat binary_little_endian 1.0\ncomment written by splatweb\nelement vertex ${n}\n` +
    names.map((nm) => `property float ${nm}`).join('\n') +
    '\nend_header\n';
  const headerBytes = new TextEncoder().encode(header);
  const stride = names.length * 4;
  const out = new Uint8Array(headerBytes.length + n * stride);
  out.set(headerBytes, 0);
  const dv = new DataView(out.buffer, headerBytes.length);
  const shStride = coeffs * 3;
  for (let i = 0; i < n; i++) {
    let o = i * stride;
    const put = (v: number) => {
      dv.setFloat32(o, v, true);
      o += 4;
    };
    put(d.positions[i * 3]); put(d.positions[i * 3 + 1]); put(d.positions[i * 3 + 2]);
    put(0); put(0); put(0);
    const s = i * shStride;
    put(d.sh[s]); put(d.sh[s + 1]); put(d.sh[s + 2]);
    for (let c = 0; c < 3; c++) for (let k = 1; k < coeffs; k++) put(d.sh[s + k * 3 + c]);
    const a = Math.min(Math.max(d.opacities[i], 1e-6), 1 - 1e-6);
    put(Math.log(a / (1 - a)));
    for (let k = 0; k < 3; k++) put(Math.log(Math.max(d.scales[i * 3 + k], 1e-12)));
    for (let k = 0; k < 4; k++) put(d.rotations[i * 4 + k]);
  }
  return out;
}
