# Embedding splatweb in a host engine (Vanta integration contract)

splatweb's renderer core is engine-agnostic: it never creates a canvas, a device, a frame loop
or input handling. A host — Vanta, or anything else — provides those, and splatweb records its
GPU work into the host's command encoder.

> This document describes what **splatweb requires from a host**. It deliberately does not
> describe Vanta's own API, which is not part of this repository. Whoever integrates Vanta
> writes a thin adapter that maps Vanta's device, camera and render-target objects onto the
> interface below.

## What the host must supply

| # | Item | Type | Notes |
|---|------|------|-------|
| 1 | GPU device | `GPUDevice` | Any WebGPU device (browser, Dawn, wgpu-native through a JS binding). Large scenes need a higher `maxStorageBufferBindingSize` (≈96 B × splats for SH degree 3); request the adapter's limits when you create the device. Add the `timestamp-query` feature if you want GPU timings. |
| 2 | Colour target | `GPUTextureView` + its `GPUTextureFormat` | Needs `RENDER_ATTACHMENT`. splatweb blends premultiplied colour onto it (see "Compositing"). |
| 3 | Viewport | `[w, h]` px, optional `[x, y, w, h]` rect | The pixel size of the region being drawn (one eye of a stereo target, a split-screen pane, ...). |
| 4 | View matrix | `Float32Array(16)`, column-major | World → view. Right-handed, camera looks down **−Z**, **+Y** up (OpenGL/WebXR convention). |
| 5 | Projection matrix | `Float32Array(16)`, column-major | View → clip. x, y in [−1, 1] with y up. Depth convention only matters when you enable depth testing; splatweb's helpers produce WebGPU's [0, 1]. Asymmetric (off-axis/XR) frusta are fine. |
| 6 | Near distance | `number` | View-space distance below which splats are culled. |
| 7 | (optional) depth target | `GPUTextureView` + format + compare | If supplied, splats are depth-tested **read-only** against the host's depth buffer so opaque host geometry occludes them. Use `depthCompare: 'greater'` for reversed-Z. |
| 8 | Command encoder + submit | `GPUCommandEncoder` | splatweb records passes; the host submits. Call `renderer.afterSubmit()` after the submit if you use stats. |

## API

```ts
import { SplatRenderer, loadFromUrl } from 'splatweb';

const splats = new SplatRenderer({
  device,                       // (1)
  colorFormat: 'bgra8unorm',    // (2) must match the target you pass to encode()
  depthFormat: 'depth24plus',   // (7) optional
  depthCompare: 'less',         // (7) optional, 'greater' for reversed-Z
  sort: 'gpu',                  // or 'cpu' fallback
  blend: 'back-to-front',       // or 'front-to-back'
});

const scene = await loadFromUrl(url, {
  onProgress: (p) => {                       // progressive streaming
    if (!p.data) return;
    if (first) { splats.setScene(p.data); first = false; } else splats.updateScene(p.data);
  },
});

splats.modelMatrix = sceneToWorld;           // place the scan in the host's world (Float32Array(16))

// every frame, inside the host's render loop:
const enc = device.createCommandEncoder();
hostRenderOpaque(enc);                       // host's own passes first (writes depth)
splats.encode(
  enc,
  { colorView, depthView, viewportRect: [x, y, w, h], clear: null },   // (2)(3)(7) load, don't clear
  { view, proj, viewport: [w, h], near },                               // (4)(5)(6)
  /* slot */ 0,
);
hostRenderTransparentOrUI(enc);
device.queue.submit([enc.finish()]);
splats.afterSubmit();
```

### Multiple views per frame (stereo, split screen, shadow-less mirrors)

Call `encode()` once per view with a **distinct `slot`** (0, 1, ...). Each slot owns its own
uniform buffers; that is required because uniforms are uploaded with `queue.writeBuffer`, which
lands before the submit. Every view re-runs preprocess + sort (the splat order is view
dependent), so cost scales linearly with the number of views.

## Compositing rules

* The fragment shader outputs **premultiplied** colour `(rgb·α, α)`.
* `back-to-front` uses the "over" operator (`src=one, dst=one-minus-src-alpha`) — correct on top
  of an existing opaque frame.
* `front-to-back` uses the "under" operator (`src=one-minus-dst-alpha, dst=one`). It is only
  correct when the target starts **transparent** (alpha 0) where splats should show; composite the
  result over the host frame afterwards. Prefer `back-to-front` when drawing directly onto a
  host frame.
* splatweb never writes depth. Splats are depth-tested against host geometry per splat centre
  (all fragments of a splat share its centre depth), which is the usual approximation.

## Coordinate frames

Trained 3DGS scenes usually come in COLMAP's convention (Y down, Z forward). Put the
correction into `modelMatrix`, e.g. `mat4.fromScaling([1, -1, -1])`, combined with whatever
placement the host needs (scan-to-world registration from a handheld scanner, for example).

## Threading / lifetime

* `setScene()` allocates GPU buffers sized for `data.count` and uploads what is loaded.
  `updateScene()` uploads only newly decoded splats — safe to call every progress event.
* `destroy()` frees everything. The host owns and destroys the device.
* The CPU sort fallback runs on the calling thread and only re-sorts when the view changes.

## Live scans (streamed splats)

A host can also show a scan **while it is being captured** (for example from splatscan running on
a Jetson). Nothing in the contract above changes; the host adds one call per frame before
`encode()`:

```ts
import { connectLive, flushLiveScene, upAxisModelMatrix } from 'splatweb';

const live = connectLive(scannerUrl, {
  onMessage: (m) => { if (m.type === 'hello') splats.modelMatrix = mul(scanToHostWorld, upAxisModelMatrix(m.hello.upAxis)); },
});
const liveState = {};
// every frame, before splats.encode(...):
flushLiveScene(live.scene, splats, liveState);   // in-place GPU updates, no reload
```

`live.scene.lastFrame` carries the scanner's camera pose (camera → world, computer-vision axes),
which a host can draw as a frustum or use to follow the device. The wire format is in
[live-stream.md](live-stream.md).

## Checklist for a host adapter

- [ ] Map host device → `GPUDevice` (request higher storage-binding limits for big scans).
- [ ] Map host camera → column-major view (−Z forward, +Y up) and projection matrices.
- [ ] Map host render target → `GPUTextureView` + format; viewport rect in pixels.
- [ ] Decide depth integration: none, or host depth view + compare function.
- [ ] Call `encode()` after opaque geometry, before UI; one slot per view.
- [ ] Call `afterSubmit()` after the host submits (only needed for stats).
- [ ] (Live scans) call `flushLiveScene()` once per frame before `encode()`.
