/**
 * Orbit and fly cameras sharing one yaw/pitch convention, so switching modes keeps the view.
 *   offset(yaw, pitch) = (cos p · sin y, sin p, cos p · cos y)    (unit vector, Y up)
 *   orbit: eye = target + distance · offset
 *   fly:   forward = -offset
 */
import type { Mat4 } from '../core/math/mat4';
import { lookAt, perspectiveZO } from '../core/math/mat4';
import type { Vec3 } from '../core/math/vec3';
import { add, scale, sub, length, normalize, cross } from '../core/math/vec3';

export function offsetDir(yaw: number, pitch: number): Vec3 {
  return [Math.cos(pitch) * Math.sin(yaw), Math.sin(pitch), Math.cos(pitch) * Math.cos(yaw)];
}

const PITCH_LIMIT = Math.PI / 2 - 0.01;
const clampPitch = (p: number) => Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, p));

export interface Lens {
  fovY: number;
  near: number;
  far: number;
}

export class OrbitCamera {
  target: Vec3 = [0, 0, 0];
  distance = 5;
  yaw = 0;
  pitch = 0.3;
  lens: Lens = { fovY: (60 * Math.PI) / 180, near: 0.05, far: 1000 };

  get eye(): Vec3 {
    return add(this.target, scale(offsetDir(this.yaw, this.pitch), this.distance));
  }

  rotate(dYaw: number, dPitch: number): void {
    this.yaw += dYaw;
    this.pitch = clampPitch(this.pitch + dPitch);
  }

  zoom(factor: number): void {
    this.distance = Math.max(1e-3, this.distance * factor);
  }

  /** Pans in the view plane by (dx, dy) in units of the current distance. */
  pan(dx: number, dy: number): void {
    const f = scale(offsetDir(this.yaw, this.pitch), -1);
    const right = normalize(cross(f, [0, 1, 0]));
    const up = cross(right, f);
    this.target = add(this.target, add(scale(right, -dx * this.distance), scale(up, dy * this.distance)));
  }

  viewMatrix(): Mat4 {
    return lookAt(this.eye, this.target, [0, 1, 0]);
  }

  frame(center: Vec3, radius: number): void {
    this.target = [...center];
    this.distance = radius / Math.tan(this.lens.fovY / 2) * 0.9;
    this.lens.near = Math.max(1e-3, radius * 0.002);
    this.lens.far = radius * 100;
  }
}

export class FlyCamera {
  position: Vec3 = [0, 1, 5];
  yaw = 0;
  pitch = 0;
  speed = 2;
  lens: Lens = { fovY: (60 * Math.PI) / 180, near: 0.05, far: 1000 };

  get forward(): Vec3 {
    return scale(offsetDir(this.yaw, this.pitch), -1);
  }

  look(dYaw: number, dPitch: number): void {
    this.yaw += dYaw;
    this.pitch = clampPitch(this.pitch + dPitch);
  }

  /** Moves in camera space: x = right, y = world up, z = forward. */
  move(local: Vec3, dt: number): void {
    const f = this.forward;
    const right = normalize(cross(f, [0, 1, 0]));
    const d = add(add(scale(right, local[0]), scale([0, 1, 0], local[1])), scale(f, local[2]));
    this.position = add(this.position, scale(d, this.speed * dt));
  }

  viewMatrix(): Mat4 {
    return lookAt(this.position, add(this.position, this.forward), [0, 1, 0]);
  }
}

export function orbitToFly(o: OrbitCamera, f: FlyCamera): void {
  f.position = o.eye;
  f.yaw = o.yaw;
  f.pitch = o.pitch;
  f.lens = { ...o.lens };
  f.speed = Math.max(0.1, o.distance * 0.5);
}

export function flyToOrbit(f: FlyCamera, o: OrbitCamera): void {
  const dist = o.distance > 0 ? o.distance : 3;
  o.yaw = f.yaw;
  o.pitch = f.pitch;
  o.target = add(f.position, scale(f.forward, dist));
  o.distance = length(sub(o.target, f.position));
  o.lens = { ...f.lens };
}

export function projectionMatrix(lens: Lens, aspect: number): Mat4 {
  return perspectiveZO(lens.fovY, aspect, lens.near, lens.far);
}
