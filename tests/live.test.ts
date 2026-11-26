/**
 * Live-stream loader (splatscan protocol v1): decoding, the slot table (add / update / swap-remove /
 * growth / clip / reset), dirty-range coalescing, the renderer flush contract, pose conversion,
 * a fixture written by splatscan's Python encoder, and a real WebSocket round trip in Node.
 */
import { createServer } from 'node:http';
import type { AddressInfo } from 'node:net';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  LIVE_MAGIC, LiveScene, MSG, connectLive, decodeLiveMessage, flushLiveScene, recordFloats, upAxisModelMatrix, viewFromFrame,
} from '../src/io/live';
import type { LiveRendererTarget } from '../src/io/live';
import type { SplatData } from '../src/io/splatData';
import * as m4 from '../src/core/math/mat4';

// ---- a tiny encoder mirroring splatscan/protocol.py (tests only) ----
function header(type: number, seq: number, bodyBytes: number) {
  const buf = new ArrayBuffer(12 + bodyBytes);
  const dv = new DataView(buf);
  dv.setUint32(0, LIVE_MAGIC, true);
  dv.setUint8(4, type);
  dv.setUint32(8, seq, true);
  return { buf, dv };
}
const hello = (shDegree: number, cap = 0, up = 2) => {
  const { buf, dv } = header(MSG.HELLO, 1, 12);
  dv.setUint16(12, 1, true); dv.setUint8(14, shDegree); dv.setUint8(15, up); dv.setUint32(16, cap, true); dv.setFloat32(20, 5, true);
  return buf;
};
interface Rec { id: number; pos: number[]; scale?: number[]; rot?: number[]; opacity?: number; sh?: number[] }
function splats(type: number, deg: number, recs: Rec[]) {
  const nf = recordFloats(deg);
  const { buf, dv } = header(type, 2, 4 + recs.length * (1 + nf) * 4);
  dv.setUint32(12, recs.length, true);
  recs.forEach((r, i) => {
    let o = 16 + i * (1 + nf) * 4;
    dv.setUint32(o, r.id, true); o += 4;
    const vals = [...r.pos, ...(r.scale ?? [0.1, 0.1, 0.1]), ...(r.rot ?? [1, 0, 0, 0]), r.opacity ?? 0.5, ...(r.sh ?? new Array(nf - 11).fill(0))];
    vals.forEach((v) => { dv.setFloat32(o, v, true); o += 4; });
  });
  return buf;
}
function remove(ids: number[]) {
  const { buf, dv } = header(MSG.REMOVE, 3, 4 + 4 * ids.length);
  dv.setUint32(12, ids.length, true);
  ids.forEach((id, i) => dv.setUint32(16 + 4 * i, id, true));
  return buf;
}
const reset = () => header(MSG.RESET, 4, 0).buf;

function positionsById(s: LiveScene): Map<number, number[]> {
  const out = new Map<number, number[]>();
  for (let i = 0; i < s.size; i++) out.set(s.idAt(i), Array.from(s.data.positions.subarray(i * 3, i * 3 + 3)));
  return out;
}

describe('live protocol decode', () => {
  it('round-trips HELLO / ADD / REMOVE / RESET', () => {
    const h = decodeLiveMessage(hello(1, 100), null);
    expect(h).toMatchObject({ type: 'hello', hello: { version: 1, shDegree: 1, upAxis: 2, capacityHint: 100 } });
    const a = decodeLiveMessage(splats(MSG.ADD, 1, [{ id: 7, pos: [1, 2, 3], opacity: 0.25 }]), 1);
    if (a.type !== 'add') throw new Error();
    expect(a.splats.ids[0]).toBe(7);
    expect(Array.from(a.splats.floats.subarray(0, 3))).toEqual([1, 2, 3]);
    expect(a.splats.floats[10]).toBe(0.25);
    expect(decodeLiveMessage(remove([4, 5]), 1)).toMatchObject({ type: 'remove' });
    expect(decodeLiveMessage(reset(), 1).type).toBe('reset');
  });

  it('rejects bad magic, wrong sizes, ADD before HELLO and unknown versions', () => {
    const bad = hello(0);
    new DataView(bad).setUint32(0, 0xdeadbeef, true);
    expect(() => decodeLiveMessage(bad, null)).toThrow(/magic/);
    expect(() => decodeLiveMessage(splats(MSG.ADD, 1, [{ id: 1, pos: [0, 0, 0] }]), null)).toThrow(/HELLO/);
    expect(() => decodeLiveMessage(splats(MSG.ADD, 1, [{ id: 1, pos: [0, 0, 0] }]), 0)).toThrow(/size mismatch/);
    const v2 = hello(0);
    new DataView(v2).setUint16(12, 2, true);
    expect(() => decodeLiveMessage(v2, null)).toThrow(/version/);
  });
});

describe('LiveScene slot table', () => {
  it('adds, updates in place and swap-removes keeping slots dense', () => {
    const s = new LiveScene({ initialCapacity: 4 });
    s.applyBuffer(hello(0));
    s.applyBuffer(splats(MSG.ADD, 0, [0, 1, 2, 3].map((id) => ({ id, pos: [id, 0, 0] }))));
    expect(s.size).toBe(4);
    s.applyBuffer(splats(MSG.UPDATE, 0, [{ id: 2, pos: [20, 0, 0] }]));
    expect(s.slotOfId(2)).toBe(2);
    expect(s.data.positions[6]).toBe(20);
    s.applyBuffer(remove([1, 99])); // unknown id ignored
    expect(s.size).toBe(3);
    expect(s.slotOfId(3)).toBe(1); // last slot moved into the hole
    expect(positionsById(s)).toEqual(new Map([[0, [0, 0, 0]], [3, [3, 0, 0]], [2, [20, 0, 0]]]));
    expect(s.counts).toMatchObject({ added: 4, updated: 1, removed: 1 });
  });

  it('grows capacity by doubling without losing data, and ADD of a known id acts as UPDATE', () => {
    const s = new LiveScene({ initialCapacity: 2 });
    s.applyBuffer(hello(1));
    const recs = Array.from({ length: 37 }, (_, id) => ({ id: id * 3, pos: [id, -id, 0.5 * id], sh: new Array(12).fill(id) }));
    s.applyBuffer(splats(MSG.ADD, 1, recs));
    expect(s.size).toBe(37);
    expect(s.data.count).toBe(64);
    for (const r of recs) {
      const slot = s.slotOfId(r.id)!;
      expect(Array.from(s.data.positions.subarray(slot * 3, slot * 3 + 3))).toEqual(r.pos.map(Math.fround));
      expect(s.data.sh[slot * 12 + 11]).toBe(r.id / 3);
    }
    s.applyBuffer(splats(MSG.ADD, 1, [{ id: 0, pos: [9, 9, 9] }]));
    expect(s.size).toBe(37);
    expect(s.counts.updated).toBe(1);
  });

  it('applies the crop box and RESET', () => {
    const s = new LiveScene({ clip: { min: [-1, -1, -1], max: [1, 1, 1] } });
    s.applyBuffer(hello(0));
    s.applyBuffer(splats(MSG.ADD, 0, [{ id: 1, pos: [0, 0, 0] }, { id: 2, pos: [0, 0, 5] }]));
    expect(s.size).toBe(1);
    s.applyBuffer(splats(MSG.UPDATE, 0, [{ id: 1, pos: [0, 0, 3] }])); // moved out of the box -> dropped
    expect(s.size).toBe(0);
    expect(s.counts.clipped).toBe(2);
    s.applyBuffer(splats(MSG.ADD, 0, [{ id: 3, pos: [0, 0, 0] }]));
    s.applyBuffer(reset());
    expect(s.size).toBe(0);
  });

  it('coalesces dirty slots into ranges', () => {
    const s = new LiveScene({ initialCapacity: 256 });
    s.applyBuffer(hello(0));
    s.applyBuffer(splats(MSG.ADD, 0, Array.from({ length: 200 }, (_, id) => ({ id, pos: [id, 0, 0] }))));
    s.takeDirty();
    s.applyBuffer(splats(MSG.UPDATE, 0, [5, 6, 7, 30, 150].map((id) => ({ id, pos: [0, 1, 0] }))));
    expect(s.takeDirty(8).ranges).toEqual([[5, 8], [30, 31], [150, 151]]);
    s.applyBuffer(splats(MSG.UPDATE, 0, [5, 30].map((id) => ({ id, pos: [0, 1, 0] }))));
    expect(s.takeDirty(32).ranges).toEqual([[5, 31]]);
  });
});

describe('flushLiveScene', () => {
  class FakeRenderer implements LiveRendererTarget {
    calls: string[] = [];
    splatCount = 0;
    capacity = 0;
    setScene(d: SplatData) { this.calls.push(`setScene(${d.loaded}/${d.count})`); this.capacity = d.count; this.splatCount = d.loaded; }
    ensureCapacity(d: SplatData) { if (d.count > this.capacity) { this.calls.push(`grow(${d.count})`); this.capacity = d.count; } }
    uploadRange(_d: SplatData, a: number, b: number) { this.calls.push(`upload(${a},${b})`); }
    setCount(n: number) { this.splatCount = n; }
  }

  it('allocates once, then grows in place and uploads only dirty ranges', () => {
    const s = new LiveScene({ initialCapacity: 4 });
    const r = new FakeRenderer();
    const state = {};
    s.applyBuffer(hello(0));
    s.applyBuffer(splats(MSG.ADD, 0, [0, 1, 2].map((id) => ({ id, pos: [id, 0, 0] }))));
    flushLiveScene(s, r, state);
    s.applyBuffer(splats(MSG.ADD, 0, [3, 4, 5].map((id) => ({ id, pos: [id, 0, 0] }))));
    flushLiveScene(s, r, state);
    s.applyBuffer(remove([0]));
    flushLiveScene(s, r, state);
    expect(r.calls).toEqual(['setScene(3/4)', 'grow(8)', 'upload(3,6)', 'upload(0,1)']);
    expect(r.splatCount).toBe(5);
  });
});

describe('poses', () => {
  it('viewFromFrame equals lookAt for an equivalent camera (z-up stream into y-up splatweb)', () => {
    // CV camera at p looking along world +x (z-up world): forward=+x, right=-y, down=-z.
    const R_wc = [0, 0, 1, -1, 0, 0, 0, -1, 0]; // columns are camera x,y,z axes in world, row-major
    const p: [number, number, number] = [1, 2, 1.5];
    const model = upAxisModelMatrix(2);
    const view = viewFromFrame({ t: 0, frameIndex: 0, splatCount: 0, R_wc, p_wc: p, fx: 1, fy: 1, cx: 0, cy: 0, width: 1, height: 1, psnr: 0 }, model);
    // same camera in splatweb (y-up) coordinates: (x, y, z) -> (x, z, -y)
    const eye: [number, number, number] = [p[0], p[2], -p[1]];
    const ref = m4.lookAt(eye, [eye[0] + 1, eye[1], eye[2]], [0, 1, 0]);
    for (let i = 0; i < 16; i++) expect(view[i]).toBeCloseTo(ref[i], 6);
  });
});

describe('cross-implementation fixture', () => {
  it('decodes the stream written by splatscan (Python) into the expected final state', () => {
    const raw = readFileSync(new URL('./fixtures/splatscan_v1.sslog', import.meta.url));
    const expected = JSON.parse(readFileSync(new URL('./fixtures/splatscan_v1.expected.json', import.meta.url), 'utf8'));
    expect(raw.subarray(0, 8).toString('latin1')).toBe('SSCLOG01');
    const s = new LiveScene({ initialCapacity: 2 });
    let o = 8;
    let frames = 0;
    while (o < raw.length) {
      const n = raw.readUInt32LE(o);
      const msg = raw.buffer.slice(raw.byteOffset + o + 4, raw.byteOffset + o + 4 + n);
      if (s.applyBuffer(msg).type === 'frame') frames++;
      o += 4 + n;
    }
    expect(frames).toBe(expected.frames);
    expect(s.size).toBe(expected.splats.length);
    for (const e of expected.splats) {
      const slot = s.slotOfId(e.id);
      expect(slot, `id ${e.id}`).toBeDefined();
      const get = (a: Float32Array, k: number) => Array.from(a.subarray(slot! * k, slot! * k + k));
      get(s.data.positions, 3).forEach((v, i) => expect(v).toBeCloseTo(e.pos[i], 5));
      get(s.data.scales, 3).forEach((v, i) => expect(v).toBeCloseTo(e.scale[i], 5));
      get(s.data.rotations, 4).forEach((v, i) => expect(v).toBeCloseTo(e.rot[i], 5));
      expect(s.data.opacities[slot!]).toBeCloseTo(e.opacity, 5);
      get(s.data.sh, e.sh.length).forEach((v, i) => expect(v).toBeCloseTo(e.sh[i], 5));
    }
    expect(s.lastFrame!.frameIndex).toBe(expected.last_frame.frame_index);
    expect(s.lastFrame!.p_wc[0]).toBeCloseTo(expected.last_frame.p_wc[0], 5);
  });
});

describe('WebSocket transport (Node built-in client)', () => {
  it.skipIf(typeof WebSocket === 'undefined')('streams messages over a real socket into a LiveScene', async () => {
    // Minimal RFC 6455 server: handshake + unmasked binary frames.
    const msgs = [hello(0), splats(MSG.ADD, 0, [{ id: 10, pos: [1, 1, 1] }, { id: 11, pos: [2, 2, 2] }]), remove([10])];
    const server = createServer();
    server.on('upgrade', (req, socket) => {
      const accept = createHash('sha1').update(`${req.headers['sec-websocket-key']}258EAFA5-E914-47DA-95CA-C5AB0DC85B11`).digest('base64');
      socket.write(`HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: ${accept}\r\n\r\n`);
      for (const m of msgs) {
        const p = Buffer.from(m);
        socket.write(Buffer.concat([Buffer.from([0x82, p.length < 126 ? p.length : 126, ...(p.length < 126 ? [] : [p.length >> 8, p.length & 255])]), p]));
      }
    });
    await new Promise<void>((r) => server.listen(0, '127.0.0.1', r));
    const port = (server.address() as AddressInfo).port;
    let seen = 0;
    const done = new Promise<void>((resolve) => {
      const c = connectLive(`ws://127.0.0.1:${port}`, {
        onMessage: () => {
          if (++seen === msgs.length) {
            c.close();
            resolve();
          }
        },
      });
      void c.opened;
      setTimeout(() => resolve(), 5000);
      (globalThis as { __live?: typeof c }).__live = c;
    });
    await done;
    const c = (globalThis as { __live?: ReturnType<typeof connectLive> }).__live!;
    server.close();
    expect(seen).toBe(3);
    expect(c.scene.size).toBe(1);
    expect(c.scene.slotOfId(11)).toBe(0);
  });
});
