// splatweb — splat rasterization.
//
// One instance per visible splat, drawn in sorted order as a 4-vertex triangle strip. The quad
// is aligned with the eigenvectors of the 2D covariance and extends 3σ along each axis, so in
// quad coordinates uv ∈ [-1, 1]² the Mahalanobis distance is simply |3·uv|² and the Gaussian
// weight is exp(-0.5 · 9 · |uv|²). This is algebraically identical to evaluating the conic
// Σ'⁻¹ at the pixel offset (tested in tests/covariance.test.ts) but needs no per-pixel matrix.
//
// Output colour is premultiplied: (rgb · α, α). The pipeline blend state chooses the operator:
//   back-to-front ("over"):  C = Cs + (1 - αs) · Cd      (src: one, dst: one-minus-src-alpha)
//   front-to-back ("under"): C = (1 - αd) · Cs + Cd      (src: one-minus-dst-alpha, dst: one)

struct Splat2D {
  center: vec2<f32>,
  axis1: vec2<f32>,
  axis2: vec2<f32>,
  color: vec2<u32>,
  depth: f32,
  _pad: f32,
};

struct RenderParams {
  viewport: vec2<f32>,
  _pad: vec2<f32>,
};

@group(0) @binding(0) var<uniform> rp: RenderParams;
@group(0) @binding(1) var<storage, read> splats2d: array<Splat2D>;
@group(0) @binding(2) var<storage, read> order: array<u32>;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
  var out: VSOut;
  let s = splats2d[order[ii]];
  if (s.axis1.x == 0.0 && s.axis1.y == 0.0) {
    // Culled: emit a degenerate primitive outside the clip volume.
    out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    out.uv = vec2<f32>(0.0);
    out.color = vec4<f32>(0.0);
    return out;
  }
  let corner = vec2<f32>(f32(vi & 1u), f32(vi >> 1u)) * 2.0 - vec2<f32>(1.0);
  let pix = s.center + corner.x * s.axis1 + corner.y * s.axis2;
  let ndc = pix / rp.viewport * 2.0 - vec2<f32>(1.0);
  out.position = vec4<f32>(ndc, s.depth, 1.0);
  out.uv = corner;
  let rg = unpack2x16float(s.color.x);
  let ba = unpack2x16float(s.color.y);
  out.color = vec4<f32>(rg, ba);
  return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let r2 = dot(in.uv, in.uv);
  if (r2 > 1.0) { discard; }
  let alpha = min(0.99, in.color.a * exp(-4.5 * r2));
  if (alpha < 1.0 / 255.0) { discard; }
  return vec4<f32>(in.color.rgb * alpha, alpha);
}
