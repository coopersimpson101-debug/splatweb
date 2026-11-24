# Capture pipeline: from a phone scan of the lab to splatweb

splatweb is a **renderer**. It does not train Gaussian splats. Training uses existing tools
(COLMAP, nerfstudio/gsplat, or a commercial app). This page lists the routes that end in a file
splatweb can load: a standard 3DGS `.ply` (preferred, keeps view-dependent SH) or a `.splat`.

```
 phone video / photos
        │
        ├── Route A: scanning app (Polycam, Luma AI, Scaniverse, ...)
        │        └── export "Gaussian splat" .ply ─────────────────────────────┐
        │                                                                       │
        └── Route B: open-source pipeline (scripts/train_from_video.sh)        │
                 ffmpeg frames → COLMAP poses → nerfstudio splatfacto (gsplat)  │
                 → ns-export gaussian-splat → splat.ply ───────────────────────┤
                                                                                ▼
                                       splatweb: drag & drop / ?url= / SplatRenderer API
```

## Route A — a scanning app

Several phone apps capture and train splats in the cloud or on device and can export a `.ply`
in the standard 3DGS layout. Menu names change between versions; look for an export option
named like "Gaussian Splat", "3DGS" or "PLY (splat)". Not every app's export includes SH bands
above degree 0 — splatweb handles degree 0 to 3 and reports the degree it found after loading.

If an app only exports a mesh or a plain point cloud, you can still load a point-cloud `.ply`
(x, y, z, red, green, blue): splatweb turns each point into a small isotropic splat, which is
useful for checking alignment but does not look like a trained splat scene.

## Route B — open source: `scripts/train_from_video.sh`

Needs a CUDA GPU, [ffmpeg](https://ffmpeg.org), [COLMAP](https://colmap.github.io) and
[nerfstudio](https://docs.nerf.studio) (`pip install nerfstudio`; splatfacto is built on
[gsplat](https://github.com/nerfstudio-project/gsplat)).

```bash
scripts/train_from_video.sh lab.mp4 colmap_work/lab 2 30000
# -> colmap_work/lab/export/splat.ply
```

Steps it runs:

1. `ffmpeg` extracts frames (default 2 fps, max 1920 px wide).
2. `ns-process-data images` runs COLMAP feature extraction, sequential matching and mapping to
   recover camera poses and a sparse point cloud.
3. `ns-train splatfacto` optimises the Gaussians.
4. `ns-export gaussian-splat` writes the standard `.ply`.

Alternatives for step 3: the original Inria `gaussian-splatting` `train.py`, or gsplat's own
example trainer. Both write the same `.ply` layout.

## Capture tips for a lab / room

* Walk slowly. Motion blur ruins COLMAP matching more than anything else.
* Lock exposure and white balance if the camera app allows it.
* Orbit objects of interest at two or three heights. Do a perimeter loop facing inward and one
  facing outward for the room itself.
* Avoid large featureless surfaces (white walls) filling the frame, and avoid mirrors and screens
  where you can.
* 150–400 sharp frames is a good range for a room. More frames means slower COLMAP, not
  necessarily a better result.

## Viewing the result

* Local: `npm run dev`, then drag `splat.ply` onto the page.
* Hosted: put the file somewhere that sends CORS headers and open
  `https://coopersimpson101-debug.github.io/splatweb/?url=<encoded file URL>`.
* Keep **Flip Y (COLMAP)** on for COLMAP-oriented scenes (the default for loaded files).

## Connection to a handheld 3D scanner

The renderer only needs positions, covariances (scale + rotation), opacity and SH colour. A
handheld scanner pipeline that produces posed RGB frames (e.g. from its own tracking instead
of COLMAP) can skip step 2: write the poses in nerfstudio's `transforms.json` format and train
from there. Placing the scan in a host world is a `modelMatrix` on the renderer (see
[vanta-integration.md](./vanta-integration.md)).
