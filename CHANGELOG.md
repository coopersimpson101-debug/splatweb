# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-11-25

First public release of a from-scratch Gaussian splatting renderer in WebGPU.

### Added
- Streaming .ply (SH degree 0-3) and .splat loaders
- Compute preprocessing: culling, EWA projection, SH colour; hand-written 4-pass GPU radix sort
- Premultiplied alpha blending, orbit/fly/touch cameras, WebXR session support
- Headless timing on an RTX 2060: 2.6 ms at 150k splats, 12.5 ms at 1M (1080p, not browser FPS)

[0.1.0]: https://github.com/coopersimpson101-debug/splatweb/releases/tag/v0.1.0
