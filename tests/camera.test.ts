import { describe, expect, it } from 'vitest';
import { FlyCamera, OrbitCamera, flyToOrbit, offsetDir, orbitToFly } from '../src/camera/camera';
import * as m4 from '../src/core/math/mat4';

describe('cameras', () => {
  it('orbit eye sits at distance from the target and looks at it', () => {
    const o = new OrbitCamera();
    o.target = [1, 2, 3];
    o.distance = 4;
    o.yaw = 0.7;
    o.pitch = -0.4;
    const e = o.eye;
    expect(Math.hypot(e[0] - 1, e[1] - 2, e[2] - 3)).toBeCloseTo(4, 6);
    const t = m4.transformPoint(o.viewMatrix(), o.target);
    expect(t[0]).toBeCloseTo(0, 5);
    expect(t[1]).toBeCloseTo(0, 5);
    expect(t[2]).toBeCloseTo(-4, 5);
  });

  it('clamps pitch away from the poles', () => {
    const o = new OrbitCamera();
    o.rotate(0, 10);
    expect(o.pitch).toBeLessThan(Math.PI / 2);
    expect(Number.isFinite(o.viewMatrix()[0])).toBe(true);
  });

  it('switching orbit <-> fly preserves the view matrix', () => {
    const o = new OrbitCamera();
    o.target = [0.5, 1, -2];
    o.distance = 3;
    o.yaw = 1.2;
    o.pitch = 0.3;
    const f = new FlyCamera();
    orbitToFly(o, f);
    expect(m4.equalsApprox(f.viewMatrix(), o.viewMatrix(), 1e-5)).toBe(true);
    f.move([0, 0, 1], 0.1);
    const o2 = new OrbitCamera();
    o2.distance = 2;
    flyToOrbit(f, o2);
    expect(m4.equalsApprox(f.viewMatrix(), o2.viewMatrix(), 1e-4)).toBe(true);
  });

  it('fly forward moves along the view direction', () => {
    const f = new FlyCamera();
    f.position = [0, 0, 0];
    f.yaw = 0;
    f.pitch = 0;
    f.speed = 1;
    f.move([0, 0, 1], 2);
    expect(f.position[2]).toBeCloseTo(-2, 6); // yaw 0 looks down -Z
    expect(offsetDir(0, 0)).toEqual([0, 0, 1]);
  });

  it('pan moves the target in the view plane', () => {
    const o = new OrbitCamera();
    o.distance = 2;
    o.yaw = 0;
    o.pitch = 0;
    o.pan(0.5, 0);
    expect(o.target[0]).toBeCloseTo(-1, 6);
    expect(o.target[2]).toBeCloseTo(0, 6);
  });
});
