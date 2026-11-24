/**
 * Mouse / touch / keyboard input for the orbit and fly cameras.
 *
 * Orbit: left-drag rotate, right-drag or shift+drag pan, wheel zoom,
 *        one-finger rotate, two-finger pinch zoom + pan.
 * Fly:   drag to look, WASD move, Q/E down/up, shift = fast; on touch, one finger looks and a
 *        second finger held down moves forward.
 * F toggles the mode.
 */
import { FlyCamera, OrbitCamera, flyToOrbit, orbitToFly } from './camera';
import type { Mat4 } from '../core/math/mat4';

export type CameraMode = 'orbit' | 'fly';

export class CameraControls {
  readonly orbit = new OrbitCamera();
  readonly fly = new FlyCamera();
  mode: CameraMode = 'orbit';
  rotateSpeed = 0.005;
  onModeChange?: (m: CameraMode) => void;
  /** Bumped every time input changes the camera (lets the app skip redundant work). */
  version = 0;

  private pointers = new Map<number, { x: number; y: number; button: number; shift: boolean }>();
  private keys = new Set<string>();
  private lastPinch: { dist: number; cx: number; cy: number } | null = null;
  private readonly el: HTMLElement;
  private readonly abort = new AbortController();

  constructor(el: HTMLElement) {
    this.el = el;
    const o = { signal: this.abort.signal };
    el.style.touchAction = 'none';
    el.addEventListener('pointerdown', (e) => this.onDown(e), o);
    el.addEventListener('pointermove', (e) => this.onMove(e), o);
    el.addEventListener('pointerup', (e) => this.onUp(e), o);
    el.addEventListener('pointercancel', (e) => this.onUp(e), o);
    el.addEventListener('contextmenu', (e) => e.preventDefault(), o);
    el.addEventListener('wheel', (e) => this.onWheel(e), { ...o, passive: false });
    window.addEventListener('keydown', (e) => this.onKey(e, true), o);
    window.addEventListener('keyup', (e) => this.onKey(e, false), o);
    window.addEventListener('blur', () => this.keys.clear(), o);
  }

  dispose(): void {
    this.abort.abort();
  }

  setMode(m: CameraMode): void {
    if (m === this.mode) return;
    if (m === 'fly') orbitToFly(this.orbit, this.fly);
    else flyToOrbit(this.fly, this.orbit);
    this.mode = m;
    this.version++;
    this.onModeChange?.(m);
  }

  get lens() {
    return this.mode === 'orbit' ? this.orbit.lens : this.fly.lens;
  }

  viewMatrix(): Mat4 {
    return this.mode === 'orbit' ? this.orbit.viewMatrix() : this.fly.viewMatrix();
  }

  /** Applies held keys (fly mode). Call once per frame. */
  update(dt: number): void {
    const touchMove = this.mode === 'fly' && this.pointers.size >= 2 ? 1 : 0;
    if (this.mode !== 'fly' || (this.keys.size === 0 && !touchMove)) return;
    const k = (c: string) => (this.keys.has(c) ? 1 : 0);
    const x = k('KeyD') - k('KeyA');
    const y = k('KeyE') - k('KeyQ');
    const z = k('KeyW') - k('KeyS') + touchMove;
    if (x || y || z) {
      const fast = this.keys.has('ShiftLeft') || this.keys.has('ShiftRight') ? 4 : 1;
      this.fly.move([x, y, z], dt * fast);
      this.version++;
    }
  }

  private onKey(e: KeyboardEvent, down: boolean) {
    const t = e.target as HTMLElement | null;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA')) return;
    if (down && e.code === 'KeyF' && !e.repeat) {
      this.setMode(this.mode === 'orbit' ? 'fly' : 'orbit');
      return;
    }
    if (down) this.keys.add(e.code);
    else this.keys.delete(e.code);
  }

  private onDown(e: PointerEvent) {
    this.el.setPointerCapture(e.pointerId);
    this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, button: e.button, shift: e.shiftKey });
    this.lastPinch = null;
  }

  private onUp(e: PointerEvent) {
    this.pointers.delete(e.pointerId);
    this.lastPinch = null;
  }

  private onMove(e: PointerEvent) {
    const p = this.pointers.get(e.pointerId);
    if (!p) return;
    const dx = e.clientX - p.x;
    const dy = e.clientY - p.y;
    p.x = e.clientX;
    p.y = e.clientY;

    if (this.pointers.size >= 2 && this.mode === 'orbit') {
      const [a, b] = [...this.pointers.values()];
      const dist = Math.hypot(a.x - b.x, a.y - b.y);
      const cx = (a.x + b.x) / 2, cy = (a.y + b.y) / 2;
      if (this.lastPinch) {
        if (dist > 0 && this.lastPinch.dist > 0) this.orbit.zoom(this.lastPinch.dist / dist);
        const h = this.el.clientHeight || 1;
        this.orbit.pan((cx - this.lastPinch.cx) / h, (cy - this.lastPinch.cy) / h);
      }
      this.lastPinch = { dist, cx, cy };
      this.version++;
      return;
    }
    if (this.pointers.size >= 2) return; // fly: second finger = move forward (see update)

    if (this.mode === 'orbit') {
      if (p.button === 2 || p.button === 1 || p.shift || e.shiftKey) {
        const h = this.el.clientHeight || 1;
        this.orbit.pan(dx / h, dy / h);
      } else {
        this.orbit.rotate(-dx * this.rotateSpeed, dy * this.rotateSpeed);
      }
    } else {
      this.fly.look(-dx * this.rotateSpeed, -dy * this.rotateSpeed);
    }
    this.version++;
  }

  private onWheel(e: WheelEvent) {
    e.preventDefault();
    const delta = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
    if (this.mode === 'orbit') this.orbit.zoom(Math.exp(delta * 0.001));
    else this.fly.speed = Math.max(0.05, this.fly.speed * Math.exp(-delta * 0.001));
    this.version++;
  }
}
