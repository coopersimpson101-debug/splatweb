/**
 * EXPERIMENTAL WebXR immersive-vr support via the WebXR/WebGPU binding (XRGPUBinding).
 *
 * Status: written against the WebXR-WebGPU binding explainer; it is feature-detected and the
 * "Enter VR" button only appears when the browser exposes XRGPUBinding. It has NOT been
 * validated on a headset by the authors of this repo yet (see README "TODO for the author").
 *
 * Per frame and per eye we call SplatRenderer.encode() with the eye's view/projection and the
 * eye's sub-image viewport, using a distinct uniform slot per eye. Only clip-space x/y are used
 * by the splat pipeline, so the GL-vs-WebGPU depth-range difference of XR projection matrices
 * does not matter (we do not depth test in XR).
 */
import type { SplatRenderer } from '../gpu/renderer';

/* Minimal typings for the parts of WebXR + XRGPUBinding we use. */
interface XRViewLike {
  transform: { inverse: { matrix: Float32Array } };
  projectionMatrix: Float32Array;
}
interface XRSubImageLike {
  colorTexture: GPUTexture;
  viewport: { x: number; y: number; width: number; height: number };
  getViewDescriptor?: () => GPUTextureViewDescriptor;
}
interface XRBindingLike {
  getPreferredColorFormat(): GPUTextureFormat;
  createProjectionLayer(init: { colorFormat: GPUTextureFormat }): unknown;
  getViewSubImage(layer: unknown, view: XRViewLike): XRSubImageLike;
}
interface XRSessionLike extends EventTarget {
  renderState: { depthNear: number };
  updateRenderState(s: { layers: unknown[] }): void;
  requestReferenceSpace(t: string): Promise<unknown>;
  requestAnimationFrame(cb: (t: number, frame: XRFrameLike) => void): number;
  end(): Promise<void>;
}
interface XRFrameLike {
  getViewerPose(space: unknown): { views: XRViewLike[] } | null;
}
interface XRSystemLike {
  isSessionSupported(mode: string): Promise<boolean>;
  requestSession(mode: string, init: unknown): Promise<XRSessionLike>;
}

const xrSystem = (): XRSystemLike | undefined => (navigator as unknown as { xr?: XRSystemLike }).xr;
const hasBinding = (): boolean => typeof (globalThis as Record<string, unknown>).XRGPUBinding === 'function';

export async function isImmersiveVrAvailable(): Promise<boolean> {
  const xr = xrSystem();
  if (!xr || !hasBinding()) return false;
  try {
    return await xr.isSessionSupported('immersive-vr');
  } catch {
    return false;
  }
}

/**
 * Starts an immersive-vr session that renders the scene with `renderer`.
 * The device must come from an adapter requested with `{ xrCompatible: true }`.
 * The renderer must have been created with colorFormat === binding.getPreferredColorFormat();
 * `makeRenderer` lets the caller create a matching one.
 */
export async function startImmersiveVr(
  device: GPUDevice,
  makeRenderer: (format: GPUTextureFormat) => SplatRenderer,
  onEnd: () => void,
): Promise<() => void> {
  const xr = xrSystem();
  if (!xr || !hasBinding()) throw new Error('WebXR with WebGPU binding is not available in this browser');
  const session = await xr.requestSession('immersive-vr', { requiredFeatures: ['webgpu'] });
  const Binding = (globalThis as unknown as { XRGPUBinding: new (s: unknown, d: GPUDevice) => XRBindingLike }).XRGPUBinding;
  const binding = new Binding(session, device);
  const format = binding.getPreferredColorFormat();
  const renderer = makeRenderer(format);
  const layer = binding.createProjectionLayer({ colorFormat: format });
  session.updateRenderState({ layers: [layer] });
  const space = await session.requestReferenceSpace('local');
  let ended = false;
  session.addEventListener('end', () => {
    ended = true;
    onEnd();
  });

  const onFrame = (_t: number, frame: XRFrameLike) => {
    if (ended) return;
    session.requestAnimationFrame(onFrame);
    const pose = frame.getViewerPose(space);
    if (!pose) return;
    const enc = device.createCommandEncoder({ label: 'splatweb xr frame' });
    pose.views.forEach((view, i) => {
      const sub = binding.getViewSubImage(layer, view);
      const vp = sub.viewport;
      const colorView = sub.colorTexture.createView(sub.getViewDescriptor?.());
      renderer.encode(
        enc,
        { colorView, viewportRect: [vp.x, vp.y, vp.width, vp.height], clear: { r: 0, g: 0, b: 0, a: 1 } },
        {
          view: new Float32Array(view.transform.inverse.matrix),
          proj: new Float32Array(view.projectionMatrix),
          viewport: [vp.width, vp.height],
          near: session.renderState.depthNear,
        },
        i,
      );
    });
    device.queue.submit([enc.finish()]);
    renderer.afterSubmit();
  };
  session.requestAnimationFrame(onFrame);
  return () => {
    if (!ended) void session.end();
  };
}
