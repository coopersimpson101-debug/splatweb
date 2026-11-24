/**
 * Minimal "host engine" example: the host owns the device, the frame loop, its own geometry and
 * a depth buffer. splatweb only records its passes into the host's command encoder — exactly the
 * contract described in docs/vanta-integration.md.
 *
 * The host draws an opaque, depth-writing quad ("a wall") through the middle of the procedural
 * scene; splats are then depth-tested (read-only) against it, so they are correctly hidden
 * behind host geometry.
 */
import { SplatRenderer, generateDemoScene, mat4, OrbitCamera, projectionMatrix } from '../src/index';

const WALL_WGSL = /* wgsl */ `
struct U { viewProj: mat4x4<f32> };
@group(0) @binding(0) var<uniform> u: U;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var p = array<vec2<f32>, 4>(vec2(-3.0, 0.0), vec2(3.0, 0.0), vec2(-3.0, 2.2), vec2(3.0, 2.2));
  return u.viewProj * vec4<f32>(p[i].x, p[i].y, 0.15, 1.0);
}
@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(0.16, 0.18, 0.24, 1.0); }
`;

async function run() {
  const status = document.getElementById('status')!;
  if (!navigator.gpu) {
    status.textContent = 'WebGPU not available.';
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter!.requestDevice();
  const canvas = document.getElementById('c') as HTMLCanvasElement;
  const ctx = canvas.getContext('webgpu')!;
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: 'opaque' });
  const depthFormat: GPUTextureFormat = 'depth24plus';

  // --- host geometry ---
  const ubuf = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const module = device.createShaderModule({ code: WALL_WGSL });
  const wall = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module, entryPoint: 'vs' },
    fragment: { module, entryPoint: 'fs', targets: [{ format }] },
    primitive: { topology: 'triangle-strip' },
    depthStencil: { format: depthFormat, depthWriteEnabled: true, depthCompare: 'less' },
  });
  const wallBG = device.createBindGroup({ layout: wall.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: ubuf } }] });

  // --- splatweb core ---
  const splats = new SplatRenderer({ device, colorFormat: format, depthFormat, depthCompare: 'less' });
  splats.setScene(generateDemoScene({ count: 60_000 }));

  const cam = new OrbitCamera();
  cam.target = [0, 1, 0];
  cam.distance = 6;
  let depthTex: GPUTexture | null = null;

  const frame = (t: number) => {
    const w = Math.floor(canvas.clientWidth * devicePixelRatio);
    const h = Math.floor(canvas.clientHeight * devicePixelRatio);
    if (canvas.width !== w || canvas.height !== h || !depthTex) {
      canvas.width = w;
      canvas.height = h;
      depthTex?.destroy();
      depthTex = device.createTexture({ size: [w, h], format: depthFormat, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    }
    cam.yaw = t * 0.0002;
    const view = cam.viewMatrix();
    const proj = projectionMatrix(cam.lens, w / h);
    device.queue.writeBuffer(ubuf, 0, mat4.multiply(proj, view));

    const colorView = ctx.getCurrentTexture().createView();
    const depthView = depthTex.createView();
    const enc = device.createCommandEncoder();
    // 1) Host pass: clear + opaque geometry, writes depth.
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: colorView, loadOp: 'clear', clearValue: { r: 0.05, g: 0.06, b: 0.08, a: 1 }, storeOp: 'store' }],
      depthStencilAttachment: { view: depthView, depthLoadOp: 'clear', depthClearValue: 1, depthStoreOp: 'store' },
    });
    pass.setPipeline(wall);
    pass.setBindGroup(0, wallBG);
    pass.draw(4);
    pass.end();
    // 2) splatweb: preprocess + sort + draw on top, depth-tested against the host depth.
    splats.encode(enc, { colorView, depthView }, { view, proj, viewport: [w, h], near: cam.lens.near });
    device.queue.submit([enc.finish()]);
    splats.afterSubmit();
    requestAnimationFrame(frame);
  };
  status.textContent = 'Host-owned device + depth buffer; splats depth-tested against the host wall.';
  requestAnimationFrame(frame);
}

void run();
