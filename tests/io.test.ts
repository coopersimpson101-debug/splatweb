import { describe, expect, it } from 'vitest';
import { PlyStreamParser, parsePly, parsePlyHeader, writePly } from '../src/io/ply';
import { SplatStreamParser, parseSplat, writeSplat } from '../src/io/splat';
import { guessFormat, loadFromStream } from '../src/io/stream';
import { generateDemoScene, mulberry32 } from '../src/io/procedural';
import { computeBounds, createSplatData } from '../src/io/splatData';
import { colorToDC, dcToColor } from '../src/core/sh';
import { packSH, packSplats, shWordsPerSplat, writePreprocessUniforms } from '../src/gpu/packing';
import { unpackHalf2 } from '../src/core/half';
import * as m4 from '../src/core/math/mat4';

const enc = new TextEncoder();

function streamOf(bytes: Uint8Array, chunk: number): ReadableStream<Uint8Array> {
  let o = 0;
  return new ReadableStream({
    pull(ctrl) {
      if (o >= bytes.length) return ctrl.close();
      ctrl.enqueue(bytes.slice(o, o + chunk));
      o += chunk;
    },
  });
}

describe('PLY', () => {
  const scene = generateDemoScene({ count: 3000, seed: 42 });

  it('round-trips a degree-3 scene through writePly/parsePly', () => {
    const back = parsePly(writePly(scene));
    expect(back.count).toBe(scene.loaded);
    expect(back.shDegree).toBe(3);
    for (let i = 0; i < 3000 * 3; i += 97) {
      expect(back.positions[i]).toBeCloseTo(scene.positions[i], 6);
      expect(back.scales[i] / scene.scales[i]).toBeCloseTo(1, 4);
    }
    for (let i = 0; i < back.sh.length; i += 131) expect(back.sh[i]).toBeCloseTo(scene.sh[i], 5);
    for (let i = 0; i < 3000; i += 37) expect(back.opacities[i]).toBeCloseTo(scene.opacities[i], 4);
    for (let i = 0; i < 3000 * 4; i += 41) expect(Math.abs(back.rotations[i])).toBeCloseTo(Math.abs(scene.rotations[i]), 5);
  });

  it('streams correctly for every chunk size, including 1 byte', async () => {
    const small = generateDemoScene({ count: 1000, seed: 1 });
    small.loaded = 40;
    const bytes = writePly(small);
    const ref = parsePly(bytes);
    for (const chunk of [1, 7, 100, 4096]) {
      const progress: number[] = [];
      const data = await loadFromStream(streamOf(bytes, chunk), 'x.ply', bytes.length, {
        onProgress: (p) => progress.push(p.loadedSplats),
      });
      expect(data.positions).toEqual(ref.positions);
      expect(data.sh).toEqual(ref.sh);
      expect(progress[progress.length - 1]).toBe(40);
    }
  });

  it('exposes partial data while streaming (progressive load)', () => {
    const bytes = writePly(scene);
    const p = new PlyStreamParser();
    const half = Math.floor(bytes.length / 2);
    p.push(bytes.subarray(0, half));
    expect(p.data!.count).toBe(3000);
    expect(p.data!.loaded).toBeGreaterThan(1000);
    expect(p.data!.loaded).toBeLessThan(3000);
    p.push(bytes.subarray(half));
    expect(p.finish().loaded).toBe(3000);
  });

  it('applies activations and re-interleaves channel-major f_rest', () => {
    // Hand-built degree-1 file: 1 vertex, mixed types, extra props, shuffled order.
    const props = ['float x', 'float y', 'float z', 'uchar flags', 'float f_dc_0', 'float f_dc_1', 'float f_dc_2',
      ...Array.from({ length: 9 }, (_, i) => `float f_rest_${i}`), 'double opacity', 'float scale_0', 'float scale_1',
      'float scale_2', 'float rot_0', 'float rot_1', 'float rot_2', 'float rot_3'];
    const header = `ply\r\nformat binary_little_endian 1.0\r\ncomment test\r\nelement vertex 1\r\n${props.map((p) => `property ${p}`).join('\r\n')}\r\nelement face 0\r\nproperty list uchar int vertex_indices\r\nend_header\r\n`;
    const body = new DataView(new ArrayBuffer(4 * 3 + 1 + 4 * 3 + 4 * 9 + 8 + 4 * 3 + 4 * 4));
    let o = 0;
    const f = (v: number) => { body.setFloat32(o, v, true); o += 4; };
    f(1); f(2); f(3);
    body.setUint8(o, 7); o += 1;
    f(0.1); f(0.2); f(0.3);
    for (let i = 0; i < 9; i++) f(10 + i); // R: 10,11,12  G: 13,14,15  B: 16,17,18
    body.setFloat64(o, 0, true); o += 8; // logit 0 -> 0.5
    f(Math.log(0.5)); f(0); f(Math.log(2));
    f(2); f(0); f(0); f(0); // unnormalised quaternion
    const bytes = new Uint8Array([...enc.encode(header), ...new Uint8Array(body.buffer)]);
    const d = parsePly(bytes);
    expect(d.shDegree).toBe(1);
    expect(Array.from(d.positions)).toEqual([1, 2, 3]);
    expect(d.opacities[0]).toBeCloseTo(0.5, 9);
    expect(d.scales[0]).toBeCloseTo(0.5, 6);
    expect(d.scales[1]).toBeCloseTo(1, 6);
    expect(d.scales[2]).toBeCloseTo(2, 6);
    expect(Array.from(d.rotations)).toEqual([1, 0, 0, 0]);
    // [k][rgb]: k=1 -> (10, 13, 16), k=2 -> (11, 14, 17), k=3 -> (12, 15, 18)
    expect(Array.from(d.sh).map((v) => Math.round(v * 10) / 10)).toEqual([0.1, 0.2, 0.3, 10, 13, 16, 11, 14, 17, 12, 15, 18]);
  });

  it('accepts plain point-cloud PLYs (COLMAP points3D) as isotropic splats', () => {
    const header = 'ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n';
    const body = new DataView(new ArrayBuffer(2 * 15));
    for (let i = 0; i < 2; i++) {
      body.setFloat32(i * 15, i, true);
      body.setFloat32(i * 15 + 4, 0, true);
      body.setFloat32(i * 15 + 8, 0, true);
      body.setUint8(i * 15 + 12, 255);
      body.setUint8(i * 15 + 13, 0);
      body.setUint8(i * 15 + 14, 51);
    }
    const d = parsePly(new Uint8Array([...enc.encode(header), ...new Uint8Array(body.buffer)]), { defaultPointScale: 0.02 });
    expect(d.count).toBe(2);
    expect(d.scales[3]).toBeCloseTo(0.02, 6);
    expect(dcToColor(d.sh[0])).toBeCloseTo(1, 6);
    expect(dcToColor(d.sh[2])).toBeCloseTo(0.2, 6);
    expect(d.opacities[0]).toBe(1);
  });

  it('rejects ascii, missing magic and truncated files with clear errors', () => {
    expect(() => parsePly(enc.encode('ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n'))).toThrow(/ASCII/);
    expect(() => parsePly(enc.encode('obj\nfoo'))).toThrow(/PLY/);
    const bytes = writePly(scene);
    expect(() => parsePly(bytes.subarray(0, bytes.length - 10))).toThrow(/truncated/);
    expect(parsePlyHeader(enc.encode('ply\nformat binary_little_endian 1.0\n'))).toBeNull();
  });

  it('handles big-endian files', () => {
    const header = 'ply\nformat binary_big_endian 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n';
    const body = new DataView(new ArrayBuffer(12));
    body.setFloat32(0, 1.5, false);
    body.setFloat32(4, -2, false);
    body.setFloat32(8, 3.25, false);
    const d = parsePly(new Uint8Array([...enc.encode(header), ...new Uint8Array(body.buffer)]));
    expect(Array.from(d.positions)).toEqual([1.5, -2, 3.25]);
  });
});

describe('.splat', () => {
  it('round-trips within 8-bit quantisation', () => {
    const scene = generateDemoScene({ count: 2000, seed: 3 });
    const back = parseSplat(writeSplat(scene));
    expect(back.count).toBe(2000);
    expect(back.shDegree).toBe(0);
    expect(back.positions).toEqual(scene.positions);
    expect(back.scales).toEqual(scene.scales);
    const shStride = 16 * 3;
    for (let i = 0; i < 2000; i += 13) {
      expect(Math.abs(back.opacities[i] - scene.opacities[i])).toBeLessThan(1 / 255 + 1e-6);
      for (let c = 0; c < 3; c++) {
        const want = Math.min(1, Math.max(0, dcToColor(scene.sh[i * shStride + c])));
        expect(Math.abs(dcToColor(back.sh[i * 3 + c]) - want)).toBeLessThan(1 / 255 + 1e-6);
      }
      const dot = [0, 1, 2, 3].reduce((s, k) => s + back.rotations[i * 4 + k] * scene.rotations[i * 4 + k], 0);
      expect(Math.abs(dot)).toBeGreaterThan(0.995);
    }
  });

  it('streams with and without a known length', async () => {
    const scene = generateDemoScene({ count: 1000, seed: 4 });
    const bytes = writeSplat(scene);
    const known = await loadFromStream(streamOf(bytes, 333), 'a.splat', bytes.length);
    const unknown = await loadFromStream(streamOf(bytes, 333), 'a.splat', undefined);
    expect(known.positions).toEqual(unknown.positions);
    const p = new SplatStreamParser(bytes.length);
    p.push(bytes.subarray(0, 32 * 10 + 5));
    expect(p.data!.loaded).toBe(10);
    expect(() => new SplatStreamParser(33)).toThrow();
  });

  it('guesses formats from magic bytes and extension', () => {
    expect(guessFormat('scene.bin', enc.encode('ply\nformat'))).toBe('ply');
    expect(guessFormat('scene.splat', new Uint8Array([1, 2, 3, 4]))).toBe('splat');
    expect(guessFormat('https://x/y.PLY?raw=1')).toBe('ply');
  });
});

describe('procedural scene', () => {
  it('is deterministic, finite and well-formed', () => {
    const a = generateDemoScene({ count: 5000, seed: 9 });
    const b = generateDemoScene({ count: 5000, seed: 9 });
    expect(a.positions).toEqual(b.positions);
    expect(a.loaded).toBe(5000);
    for (const arr of [a.positions, a.scales, a.rotations, a.opacities, a.sh]) for (const v of arr) expect(Number.isFinite(v)).toBe(true);
    for (let i = 0; i < 5000; i++) {
      const l = Math.hypot(a.rotations[i * 4], a.rotations[i * 4 + 1], a.rotations[i * 4 + 2], a.rotations[i * 4 + 3]);
      expect(l).toBeCloseTo(1, 5);
      expect(a.opacities[i]).toBeGreaterThanOrEqual(0);
      expect(a.opacities[i]).toBeLessThanOrEqual(1);
    }
    const bounds = computeBounds(a);
    expect(bounds.radius).toBeGreaterThan(2);
    expect(bounds.radius).toBeLessThan(10);
  });

  it('mulberry32 is uniform-ish', () => {
    const r = mulberry32(1);
    let s = 0;
    for (let i = 0; i < 10000; i++) s += r();
    expect(s / 10000).toBeCloseTo(0.5, 1);
  });
});

describe('GPU packing layout', () => {
  it('packs splats and SH as documented', () => {
    const d = createSplatData(2, 1);
    d.loaded = 2;
    d.positions.set([1.25, -2.5, 3.125, 4, 5, 6]);
    d.scales.set([0.5, 0.25, 2, 1, 1, 1]);
    d.rotations.set([1, 0, 0, 0, 0.5, 0.5, 0.5, 0.5]);
    d.opacities.set([0.75, 1]);
    for (let i = 0; i < d.sh.length; i++) d.sh[i] = i * 0.5;
    const s = packSplats(d);
    const f = new Float32Array(s.buffer);
    expect(Array.from(f.subarray(0, 3))).toEqual([1.25, -2.5, 3.125]);
    expect(unpackHalf2(s[3])).toEqual([0.5, 0.25]);
    expect(unpackHalf2(s[4])).toEqual([2, 0.75]);
    expect(unpackHalf2(s[5])).toEqual([1, 0]);
    expect(unpackHalf2(s[8 + 6])).toEqual([0.5, 0.5]);
    expect(shWordsPerSplat(0)).toBe(2);
    expect(shWordsPerSplat(3)).toBe(24);
    const sh = packSH(d);
    expect(sh.length).toBe(2 * 6);
    expect(unpackHalf2(sh[0])).toEqual([0, 0.5]);
    expect(unpackHalf2(sh[6])).toEqual([6, 6.5]); // second splat starts at float 12
    // partial range packing
    expect(packSplats(d, 1, 2)[0]).toBe(s[8]);
  });

  it('writes the 192-byte uniform block', () => {
    const P = m4.perspectiveZO(1, 2, 0.1, 10);
    const buf = writePreprocessUniforms({
      view: m4.create(), proj: P, camPos: [1, 2, 3], viewport: [800, 400], splatCount: 77, shDegree: 2, shStride: 14,
      frontToBack: true, scaleModifier: 1.5, near: 0.2,
    });
    const f = new Float32Array(buf), u = new Uint32Array(buf);
    expect(buf.byteLength).toBe(192);
    expect(f[16]).toBeCloseTo(P[0], 6);
    expect(Array.from(f.subarray(32, 35))).toEqual([1, 2, 3]);
    expect(f[38]).toBeCloseTo(P[0] * 400, 4);
    expect(Array.from(u.subarray(40, 44))).toEqual([77, 2, 14, 1]);
    expect(f[44]).toBeCloseTo(1.5);
    expect(f[45]).toBeCloseTo(0.2);
    expect(colorToDC(0.5)).toBe(0);
  });
});
