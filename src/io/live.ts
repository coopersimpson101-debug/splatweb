/**
 * Live splat streams (splatscan protocol v1): decode incremental ADD / UPDATE / REMOVE
 * messages into a growable SplatData and push only the changed ranges to the GPU.
 *
 * Wire format (little-endian, one protocol message per WebSocket binary message) — the
 * normative spec is docs/live-stream.md (mirrors splatscan's docs/protocol.md):
 *
 *   header  u32 magic 0x31435353 "SSC1" | u8 type | u8 flags | u16 reserved | u32 seq
 *   HELLO   u16 version, u8 shDegree, u8 upAxis (1=+y, 2=+z), u32 capacityHint, f32 sceneRadius
 *   ADD     u32 n, n records       UPDATE  u32 n, n records       REMOVE  u32 n, n x u32 id
 *   FRAME   f64 t, u32 frameIndex, u32 splatCount, f32 R_wc[9] row-major, f32 p_wc[3],
 *           f32 fx, fy, cx, cy, u16 width, u16 height, f32 psnr
 *   RESET   (empty)
 *   record  u32 id, f32 pos[3], f32 scale[3], f32 rot[4] (w,x,y,z), f32 opacity, f32 sh[K*3]
 *
 * Splats live in dense slots [0, loaded). REMOVE swaps the last slot into the hole, so the
 * renderer never draws tombstones and the GPU buffers stay compact.
 */
import { coeffCountForDegree } from '../core/sh';
import type { SplatData } from './splatData';
import { createSplatData } from './splatData';

export const LIVE_MAGIC = 0x31435353;
export const LIVE_VERSION = 1;
export const MSG = { HELLO: 1, ADD: 2, UPDATE: 3, REMOVE: 4, FRAME: 5, RESET: 6 } as const;
const HEADER_BYTES = 12;

export interface LiveHello { version: number; shDegree: number; upAxis: number; capacityHint: number; sceneRadius: number }
export interface LiveFrame {
  t: number; frameIndex: number; splatCount: number;
  /** Camera -> world rotation, row-major (computer-vision camera: x right, y down, z forward). */
  R_wc: number[]; p_wc: [number, number, number];
  fx: number; fy: number; cx: number; cy: number; width: number; height: number; psnr: number;
}
export interface LiveSplats { ids: Uint32Array; floats: Float32Array; floatsPerRecord: number }
export type LiveMessage =
  | { type: 'hello'; seq: number; hello: LiveHello }
  | { type: 'add' | 'update'; seq: number; splats: LiveSplats }
  | { type: 'remove'; seq: number; ids: Uint32Array }
  | { type: 'frame'; seq: number; frame: LiveFrame }
  | { type: 'reset'; seq: number };

export const recordFloats = (shDegree: number) => 11 + 3 * coeffCountForDegree(shDegree);

/** Decodes one message. ADD/UPDATE need the SH degree announced by HELLO. */
export function decodeLiveMessage(buf: ArrayBuffer, shDegree: number | null): LiveMessage {
  if (buf.byteLength < HEADER_BYTES) throw new Error('live: short message');
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== LIVE_MAGIC) throw new Error(`live: bad magic 0x${dv.getUint32(0, true).toString(16)}`);
  const type = dv.getUint8(4);
  const seq = dv.getUint32(8, true);
  const o = HEADER_BYTES;
  switch (type) {
    case MSG.HELLO: {
      const version = dv.getUint16(o, true);
      if (version !== LIVE_VERSION) throw new Error(`live: unsupported protocol version ${version}`);
      return {
        type: 'hello', seq,
        hello: { version, shDegree: dv.getUint8(o + 2), upAxis: dv.getUint8(o + 3), capacityHint: dv.getUint32(o + 4, true), sceneRadius: dv.getFloat32(o + 8, true) },
      };
    }
    case MSG.ADD:
    case MSG.UPDATE: {
      if (shDegree === null) throw new Error('live: ADD/UPDATE before HELLO');
      const n = dv.getUint32(o, true);
      const nf = recordFloats(shDegree);
      const stride = 1 + nf;
      if (buf.byteLength !== o + 4 + n * stride * 4) throw new Error(`live: record size mismatch (${buf.byteLength} bytes for ${n} records)`);
      const ids = new Uint32Array(n);
      const floats = new Float32Array(n * nf);
      // records start at byte 16 (4-aligned), so typed views are safe on little-endian hosts
      const words = new Uint32Array(buf, o + 4, n * stride);
      const f = new Float32Array(buf, o + 4, n * stride);
      for (let i = 0; i < n; i++) {
        ids[i] = words[i * stride];
        floats.set(f.subarray(i * stride + 1, (i + 1) * stride), i * nf);
      }
      return { type: type === MSG.ADD ? 'add' : 'update', seq, splats: { ids, floats, floatsPerRecord: nf } };
    }
    case MSG.REMOVE: {
      const n = dv.getUint32(o, true);
      if (buf.byteLength !== o + 4 + 4 * n) throw new Error('live: REMOVE size mismatch');
      return { type: 'remove', seq, ids: new Uint32Array(buf.slice(o + 4, o + 4 + 4 * n)) };
    }
    case MSG.FRAME: {
      const R: number[] = [];
      for (let k = 0; k < 9; k++) R.push(dv.getFloat32(o + 16 + 4 * k, true));
      const q = o + 52;
      return {
        type: 'frame', seq,
        frame: {
          t: dv.getFloat64(o, true), frameIndex: dv.getUint32(o + 8, true), splatCount: dv.getUint32(o + 12, true), R_wc: R,
          p_wc: [dv.getFloat32(q, true), dv.getFloat32(q + 4, true), dv.getFloat32(q + 8, true)],
          fx: dv.getFloat32(q + 12, true), fy: dv.getFloat32(q + 16, true), cx: dv.getFloat32(q + 20, true), cy: dv.getFloat32(q + 24, true),
          width: dv.getUint16(q + 28, true), height: dv.getUint16(q + 30, true), psnr: dv.getFloat32(q + 32, true),
        },
      };
    }
    case MSG.RESET:
      return { type: 'reset', seq };
    default:
      throw new Error(`live: unknown message type ${type}`);
  }
}

export interface LiveSceneOptions {
  /** Initial slot capacity (grows by doubling). Default 4096 or HELLO's capacityHint. */
  initialCapacity?: number;
  /** Optional crop box in *stream* coordinates: splats outside it are not drawn (e.g. a ceiling cut). */
  clip?: { min: [number, number, number]; max: [number, number, number] };
}

/** Dirty slot ranges [start, end) to upload since the last flush. */
export interface DirtyRanges { ranges: [number, number][]; grew: boolean }

/**
 * CPU mirror of a live scene. Feed it decoded messages with `apply()`; call `takeDirty()` (or
 * `flushTo(renderer)`) once per frame to upload only what changed.
 */
export class LiveScene {
  data: SplatData;
  hello: LiveHello | null = null;
  lastFrame: LiveFrame | null = null;
  /** Monotonic counter bumped on every applied message (handy for tests / UI). */
  version = 0;
  readonly counts = { added: 0, updated: 0, removed: 0, clipped: 0, resets: 0, messages: 0, bytes: 0 };
  private slotOf = new Map<number, number>();
  private idOf: Uint32Array;
  private dirty = new Set<number>();
  private grew = false;
  private readonly opts: LiveSceneOptions;

  constructor(opts: LiveSceneOptions = {}) {
    this.opts = opts;
    this.data = createSplatData(Math.max(1, opts.initialCapacity ?? 4096), 0);
    this.idOf = new Uint32Array(this.data.count);
  }

  get size(): number {
    return this.data.loaded;
  }

  /** Decodes and applies one raw message. */
  applyBuffer(buf: ArrayBuffer): LiveMessage {
    const m = decodeLiveMessage(buf, this.hello ? this.hello.shDegree : null);
    this.counts.bytes += buf.byteLength;
    this.apply(m);
    return m;
  }

  apply(m: LiveMessage): void {
    this.counts.messages++;
    this.version++;
    switch (m.type) {
      case 'hello': {
        const sameDeg = this.hello?.shDegree === m.hello.shDegree;
        this.hello = m.hello;
        if (!sameDeg || this.data.loaded === 0) {
          const cap = Math.max(this.data.count, m.hello.capacityHint || 0, 1);
          this.reallocate(cap, m.hello.shDegree, 0);
        }
        break;
      }
      case 'reset':
        this.counts.resets++;
        this.slotOf.clear();
        this.data.loaded = 0;
        this.dirty.clear();
        this.grew = true; // forces a count update even though nothing is uploaded
        break;
      case 'add':
      case 'update':
        this.upsert(m.splats, m.type);
        break;
      case 'remove':
        for (const id of m.ids) this.removeId(id);
        break;
      case 'frame':
        this.lastFrame = m.frame;
        break;
    }
  }

  private reallocate(capacity: number, shDegree: number, keep: number) {
    const old = this.data;
    const d = createSplatData(capacity, shDegree);
    if (keep > 0 && old.shDegree === shDegree) {
      d.positions.set(old.positions.subarray(0, keep * 3));
      d.scales.set(old.scales.subarray(0, keep * 3));
      d.rotations.set(old.rotations.subarray(0, keep * 4));
      d.opacities.set(old.opacities.subarray(0, keep));
      const f = coeffCountForDegree(shDegree) * 3;
      d.sh.set(old.sh.subarray(0, keep * f));
    }
    d.loaded = keep;
    const ids = new Uint32Array(capacity);
    ids.set(this.idOf.subarray(0, Math.min(keep, this.idOf.length)));
    this.idOf = ids;
    this.data = d;
    if (keep === 0) this.slotOf.clear();
    this.grew = true;
  }

  private inClip(f: Float32Array, o: number): boolean {
    const c = this.opts.clip;
    if (!c) return true;
    for (let a = 0; a < 3; a++) if (f[o + a] < c.min[a] || f[o + a] > c.max[a]) return false;
    return true;
  }

  private upsert(s: LiveSplats, kind: 'add' | 'update') {
    const d = this.data;
    const nf = s.floatsPerRecord;
    const shF = coeffCountForDegree(d.shDegree) * 3;
    if (nf !== 11 + shF) throw new Error('live: record layout does not match the scene SH degree');
    for (let i = 0; i < s.ids.length; i++) {
      const id = s.ids[i];
      const o = i * nf;
      if (!this.inClip(s.floats, o)) {
        this.counts.clipped++;
        this.removeId(id);
        continue;
      }
      let slot = this.slotOf.get(id);
      if (slot === undefined) {
        if (this.data.loaded >= this.data.count) this.reallocate(this.data.count * 2, this.data.shDegree, this.data.loaded);
        slot = this.data.loaded++;
        this.slotOf.set(id, slot);
        this.idOf[slot] = id;
        this.counts.added++;
      } else {
        this.counts.updated++;
      }
      this.writeSlot(slot, s.floats, o, shF);
    }
    void kind;
  }

  private writeSlot(slot: number, f: Float32Array, o: number, shF: number) {
    const d = this.data;
    d.positions[slot * 3] = f[o]; d.positions[slot * 3 + 1] = f[o + 1]; d.positions[slot * 3 + 2] = f[o + 2];
    d.scales[slot * 3] = f[o + 3]; d.scales[slot * 3 + 1] = f[o + 4]; d.scales[slot * 3 + 2] = f[o + 5];
    for (let k = 0; k < 4; k++) d.rotations[slot * 4 + k] = f[o + 6 + k];
    d.opacities[slot] = f[o + 10];
    d.sh.set(f.subarray(o + 11, o + 11 + shF), slot * shF);
    this.dirty.add(slot);
  }

  private removeId(id: number) {
    const slot = this.slotOf.get(id);
    if (slot === undefined) return;
    const d = this.data;
    const last = d.loaded - 1;
    this.slotOf.delete(id);
    if (slot !== last) {
      const lastId = this.idOf[last];
      d.positions.copyWithin(slot * 3, last * 3, last * 3 + 3);
      d.scales.copyWithin(slot * 3, last * 3, last * 3 + 3);
      d.rotations.copyWithin(slot * 4, last * 4, last * 4 + 4);
      d.opacities[slot] = d.opacities[last];
      const shF = coeffCountForDegree(d.shDegree) * 3;
      d.sh.copyWithin(slot * shF, last * shF, last * shF + shF);
      this.idOf[slot] = lastId;
      this.slotOf.set(lastId, slot);
      this.dirty.add(slot);
    }
    this.dirty.delete(last);
    d.loaded = last;
    this.counts.removed++;
  }

  /** Splat id stored in a slot (tests / picking). */
  idAt(slot: number): number {
    return this.idOf[slot];
  }

  slotOfId(id: number): number | undefined {
    return this.slotOf.get(id);
  }

  /**
   * Coalesces dirty slots into ranges (merging gaps up to `mergeGap` slots, since one bigger
   * writeBuffer beats many tiny ones) and clears the dirty set.
   */
  takeDirty(mergeGap = 32): DirtyRanges {
    const slots = [...this.dirty].filter((s) => s < this.data.loaded).sort((a, b) => a - b);
    this.dirty.clear();
    const ranges: [number, number][] = [];
    for (const s of slots) {
      const r = ranges[ranges.length - 1];
      if (r && s - r[1] <= mergeGap) r[1] = s + 1;
      else ranges.push([s, s + 1]);
    }
    const grew = this.grew;
    this.grew = false;
    return { ranges, grew };
  }
}

/** The subset of SplatRenderer that live streaming needs (lets tests use a fake). */
export interface LiveRendererTarget {
  setScene(d: SplatData): void;
  ensureCapacity(d: SplatData): void;
  uploadRange(d: SplatData, start: number, end: number): void;
  setCount(n: number): void;
  readonly splatCount: number;
}

/**
 * Pushes pending changes of `scene` to `renderer`: the first call allocates; later calls grow
 * GPU buffers in place (GPU-side copy) when needed and upload only dirty ranges.
 * Returns the number of splats uploaded.
 */
export function flushLiveScene(scene: LiveScene, renderer: LiveRendererTarget, state: { bound?: SplatData | null }): number {
  const { ranges, grew } = scene.takeDirty();
  const d = scene.data;
  if (!state.bound) {
    renderer.setScene(d); // uploads [0, loaded)
    state.bound = d;
    return d.loaded;
  }
  if (grew || state.bound !== d) {
    renderer.ensureCapacity(d);
    state.bound = d;
  }
  let n = 0;
  for (const [a, b] of ranges) {
    renderer.uploadRange(d, a, b);
    n += b - a;
  }
  renderer.setCount(d.loaded);
  return n;
}

export interface LiveConnection {
  scene: LiveScene;
  socket: WebSocket;
  close(): void;
  /** Resolves when the socket opens. */
  opened: Promise<void>;
}

/** Opens `url` (ws:// or wss://) and applies every message to a LiveScene. */
export function connectLive(url: string, opts: LiveSceneOptions & {
  onMessage?: (m: LiveMessage, scene: LiveScene) => void;
  onError?: (e: unknown) => void;
  onClose?: () => void;
  WebSocketImpl?: typeof WebSocket;
} = {}): LiveConnection {
  const WS = opts.WebSocketImpl ?? WebSocket;
  const scene = new LiveScene(opts);
  const socket = new WS(url);
  socket.binaryType = 'arraybuffer';
  const opened = new Promise<void>((resolve, reject) => {
    socket.addEventListener('open', () => resolve());
    socket.addEventListener('error', (e) => reject(e));
  });
  socket.addEventListener('message', (ev: MessageEvent) => {
    if (!(ev.data instanceof ArrayBuffer)) return;
    try {
      const m = scene.applyBuffer(ev.data);
      opts.onMessage?.(m, scene);
    } catch (e) {
      opts.onError?.(e);
    }
  });
  socket.addEventListener('close', () => opts.onClose?.());
  return { scene, socket, opened, close: () => socket.close() };
}

/**
 * Model matrix that maps a stream's world frame onto splatweb's +Y-up convention
 * (column-major). For upAxis 2 (+Z up): (x, y, z) -> (x, z, -y).
 */
export function upAxisModelMatrix(upAxis: number): Float32Array<ArrayBuffer> {
  const m = new Float32Array(16);
  if (upAxis === 2) {
    m[0] = 1; m[6] = -1; m[9] = 1; m[15] = 1; // col0=(1,0,0) col1=(0,0,-1) col2=(0,1,0)
  } else {
    m[0] = m[5] = m[10] = m[15] = 1;
  }
  return m;
}

/**
 * splatweb view matrix (world -> view, -Z forward, +Y up) for a stream FRAME pose
 * (CV camera -> stream world), composed with `model` (stream world -> splatweb world).
 */
export function viewFromFrame(f: LiveFrame, model: Float32Array): Float32Array<ArrayBuffer> {
  // T_cw (CV) = [R^T, -R^T p]; flip y,z to go from CV camera to GL camera.
  const R = f.R_wc;
  const p = f.p_wc;
  const v = new Float32Array(16);
  const flip = [1, -1, -1];
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) v[c * 4 + r] = flip[r] * R[c * 3 + r]; // (R^T)[r][c] = R[c][r]
    v[12 + r] = flip[r] * -(R[0 * 3 + r] * p[0] + R[1 * 3 + r] * p[1] + R[2 * 3 + r] * p[2]);
  }
  v[15] = 1;
  // view_splatweb = V_stream * model^-1 ; model is a rotation here, so its inverse is its transpose
  const inv = new Float32Array(16);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) inv[c * 4 + r] = model[r * 4 + c];
  inv[15] = 1;
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += v[k * 4 + r] * inv[c * 4 + k];
      out[c * 4 + r] = s;
    }
  return out;
}
