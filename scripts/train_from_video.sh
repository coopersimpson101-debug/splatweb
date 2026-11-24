#!/usr/bin/env bash
# Phone video -> trained 3D Gaussian Splatting .ply that splatweb can render.
#
# IMPORTANT: this script does NOT train anything "from scratch" in this repo. It orchestrates
# existing, well-known tools:  ffmpeg (frames) -> COLMAP (camera poses, via nerfstudio's
# ns-process-data) -> nerfstudio `splatfacto` (Gaussian training, built on gsplat) -> ns-export.
# splatweb's own contribution is the renderer.
#
# Requirements (install yourself; a CUDA GPU is needed for training):
#   - ffmpeg            https://ffmpeg.org
#   - COLMAP            https://colmap.github.io        (on PATH as `colmap`)
#   - nerfstudio        pip install nerfstudio           (provides ns-process-data, ns-train, ns-export)
#
# Usage:
#   scripts/train_from_video.sh <video.mp4> [work_dir] [fps] [max_iterations]
# Example:
#   scripts/train_from_video.sh lab_walkthrough.mp4 colmap_work/lab 2 30000
#
# Output: <work_dir>/export/splat.ply  — drag it into the splatweb page, or serve it and open
#         index.html?url=<served path>. Trained scenes are usually COLMAP-oriented: keep
#         "Flip Y (COLMAP)" enabled in the viewer.
set -euo pipefail

VIDEO="${1:?usage: train_from_video.sh <video> [work_dir] [fps] [max_iterations]}"
WORK="${2:-colmap_work/$(basename "${VIDEO%.*}")}"
FPS="${3:-2}"
ITERS="${4:-30000}"

for tool in ffmpeg colmap ns-process-data ns-train ns-export; do
  command -v "$tool" >/dev/null 2>&1 || { echo "error: '$tool' not found on PATH" >&2; exit 1; }
done

mkdir -p "$WORK/frames"

echo "==> 1/4 extracting frames at ${FPS} fps (sharp, evenly spaced frames matter more than many frames)"
ffmpeg -hide_banner -loglevel error -i "$VIDEO" -vf "fps=${FPS},scale='min(1920,iw)':-2" -q:v 2 "$WORK/frames/frame_%05d.jpg"
echo "    $(ls "$WORK/frames" | wc -l) frames"

echo "==> 2/4 structure-from-motion with COLMAP (via ns-process-data)"
ns-process-data images --data "$WORK/frames" --output-dir "$WORK/processed" --matching-method sequential

echo "==> 3/4 training Gaussians with nerfstudio splatfacto (${ITERS} iterations)"
ns-train splatfacto \
  --data "$WORK/processed" \
  --output-dir "$WORK/outputs" \
  --max-num-iterations "$ITERS" \
  --viewer.quit-on-train-completion True

CONFIG="$(ls -t "$WORK"/outputs/*/splatfacto/*/config.yml | head -n 1)"
echo "    config: $CONFIG"

echo "==> 4/4 exporting standard 3DGS .ply"
ns-export gaussian-splat --load-config "$CONFIG" --output-dir "$WORK/export"

echo "done: $WORK/export/splat.ply"
