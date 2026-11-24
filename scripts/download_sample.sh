#!/usr/bin/env bash
# Downloads a public sample scene for local testing into scenes/ (gitignored).
# Nothing is downloaded by the build or tests; run this yourself.
#
#   scripts/download_sample.sh [url] [output_name]
#
# The default is the "train" scene in .splat format that the antimatter15/splat web viewer
# demo uses (hosted on Hugging Face, roughly 60 MB). Check the dataset's license before
# redistributing it. Any URL to a 3DGS .ply or .splat works.
set -euo pipefail

URL="${1:-https://huggingface.co/cakewalk/splat-data/resolve/main/train.splat}"
NAME="${2:-$(basename "${URL%%\?*}")}"
mkdir -p scenes
echo "downloading $URL -> scenes/$NAME"
curl -L --fail --progress-bar -o "scenes/$NAME" "$URL"
ls -lh "scenes/$NAME"
echo "open: npm run dev, then drag scenes/$NAME onto the page"
echo "  or: npx vite-node scripts/headless.ts -- --scene scenes/$NAME --flip --out shot.png"
