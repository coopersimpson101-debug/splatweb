// splatweb — per-splat preprocessing (one thread per Gaussian).
//
// For every splat:
//   1. cull by opacity, near plane and screen rectangle,
//   2. build the 3D covariance  Σ = R S Sᵀ Rᵀ  from the f16 scale + rotation,
//   3. EWA-project it to a 2D covariance  Σ' = J W Σ Wᵀ Jᵀ + 0.3 I  (pixels²),
//   4. eigen-decompose Σ' into an oriented quad (3σ along each principal axis),
//   5. evaluate spherical harmonics (degree 0..3) for the view-dependent colour,
//   6. emit a depth sort key + index, and count visible splats into the indirect draw args.
//
// The CPU mirror of this file is src/core/covariance.ts (projectSplat) and is unit tested.

struct Uniforms {
  view: mat4x4<f32>,      // model-view (data space -> view space), camera looks down -Z
  proj: mat4x4<f32>,      // projection, clip z in [0, 1]
  cam_pos: vec4<f32>,     // camera position in data space (xyz)
  viewport: vec2<f32>,    // render target size in pixels
  focal: vec2<f32>,       // fx, fy in pixels
  splat_count: u32,
  sh_degree: u32,         // degree to evaluate (<= degree stored in the buffer)
  sh_stride: u32,         // u32 words per splat in the SH buffer
  flags: u32,             // bit 0: front-to-back ordering
  scale_mod: f32,
  near: f32,
  _pad: vec2<f32>,
};

struct Splat2D {
  center: vec2<f32>,      // pixels, y up, origin at bottom-left of the viewport
  axis1: vec2<f32>,       // major half-axis (3σ) in pixels; (0,0) means culled
  axis2: vec2<f32>,       // minor half-axis (3σ) in pixels
  color: vec2<u32>,       // pack2x16float(r, g), pack2x16float(b, opacity)
  depth: f32,             // clip z / w, used when depth-testing against host geometry
  _pad: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> splats: array<u32>;
@group(0) @binding(2) var<storage, read> sh: array<u32>;
@group(0) @binding(3) var<storage, read_write> out2d: array<Splat2D>;
@group(0) @binding(4) var<storage, read_write> sort_keys: array<u32>;
@group(0) @binding(5) var<storage, read_write> sort_vals: array<u32>;
// drawIndirect args: [vertexCount, instanceCount, firstVertex, firstInstance]
@group(0) @binding(6) var<storage, read_write> draw_args: array<atomic<u32>, 4>;

const SH_C0: f32 = 0.28209479177387814;
const SH_C1: f32 = 0.4886025119029199;
const SH_C2 = array<f32, 5>(1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396);
const SH_C3 = array<f32, 7>(-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435);
const ALPHA_CUTOFF: f32 = 1.0 / 255.0;
const LOW_PASS: f32 = 0.3;
const SIGMA_EXTENT: f32 = 3.0;
const CULLED_KEY: u32 = 0xffffffffu;
const WG: u32 = 256u;
const MAX_WG_X: u32 = 65535u;

fn half_at(base: u32, h: u32) -> f32 {
  let w = unpack2x16float(sh[base + (h >> 1u)]);
  return select(w.x, w.y, (h & 1u) == 1u);
}

fn sh_coef(base: u32, k: u32) -> vec3<f32> {
  let h = k * 3u;
  return vec3<f32>(half_at(base, h), half_at(base, h + 1u), half_at(base, h + 2u));
}

fn eval_sh(base: u32, degree: u32, dir_in: vec3<f32>) -> vec3<f32> {
  let d = normalize(dir_in);
  let x = d.x; let y = d.y; let z = d.z;
  var c = SH_C0 * sh_coef(base, 0u);
  if (degree > 0u) {
    c += -SH_C1 * y * sh_coef(base, 1u) + SH_C1 * z * sh_coef(base, 2u) - SH_C1 * x * sh_coef(base, 3u);
    if (degree > 1u) {
      let xx = x * x; let yy = y * y; let zz = z * z;
      c += SH_C2[0] * x * y * sh_coef(base, 4u)
         + SH_C2[1] * y * z * sh_coef(base, 5u)
         + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(base, 6u)
         + SH_C2[3] * x * z * sh_coef(base, 7u)
         + SH_C2[4] * (xx - yy) * sh_coef(base, 8u);
      if (degree > 2u) {
        c += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(base, 9u)
           + SH_C3[1] * x * y * z * sh_coef(base, 10u)
           + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(base, 11u)
           + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(base, 12u)
           + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(base, 13u)
           + SH_C3[5] * z * (xx - yy) * sh_coef(base, 14u)
           + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(base, 15u);
      }
    }
  }
  return max(c + vec3<f32>(0.5), vec3<f32>(0.0));
}

// Rotation matrix of quaternion (w, x, y, z). WGSL matrices are column-major.
fn quat_to_mat3(q_in: vec4<f32>) -> mat3x3<f32> {
  let q = normalize(q_in);
  let w = q.x; let x = q.y; let y = q.z; let z = q.w;
  return mat3x3<f32>(
    vec3<f32>(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)),
    vec3<f32>(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)),
    vec3<f32>(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)),
  );
}

fn cull(i: u32) {
  var s: Splat2D;
  s.center = vec2<f32>(0.0);
  s.axis1 = vec2<f32>(0.0);
  s.axis2 = vec2<f32>(0.0);
  s.color = vec2<u32>(0u);
  s.depth = 0.0;
  s._pad = 0.0;
  out2d[i] = s;
  sort_keys[i] = CULLED_KEY;
  sort_vals[i] = i;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let i = (wid.y * MAX_WG_X + wid.x) * WG + lid;
  if (i >= u.splat_count) { return; }

  let b = i * 8u;
  let pos = vec3<f32>(bitcast<f32>(splats[b]), bitcast<f32>(splats[b + 1u]), bitcast<f32>(splats[b + 2u]));
  let s01 = unpack2x16float(splats[b + 3u]);
  let s2o = unpack2x16float(splats[b + 4u]);
  let q01 = unpack2x16float(splats[b + 5u]);
  let q23 = unpack2x16float(splats[b + 6u]);
  let scale = vec3<f32>(s01.x, s01.y, s2o.x) * u.scale_mod;
  let opacity = s2o.y;
  let quat = vec4<f32>(q01.x, q01.y, q23.x, q23.y);

  if (opacity < ALPHA_CUTOFF) { cull(i); return; }

  let t = (u.view * vec4<f32>(pos, 1.0)).xyz;
  let d = -t.z;
  if (d < u.near) { cull(i); return; }

  let clip = u.proj * vec4<f32>(t, 1.0);
  let ndc = clip.xyz / clip.w;
  let center = (ndc.xy * 0.5 + vec2<f32>(0.5)) * u.viewport;

  // 3D covariance Σ = M Mᵀ with M = R S.
  let R = quat_to_mat3(quat);
  let M = mat3x3<f32>(R[0] * scale.x, R[1] * scale.y, R[2] * scale.z);
  let Sigma = M * transpose(M);

  // Jacobian of the perspective projection at the (frustum-clamped) splat centre.
  let fx = u.focal.x;
  let fy = u.focal.y;
  let lim = 1.3 * u.viewport / (2.0 * u.focal);
  let txy = clamp(t.xy / d, -lim, lim) * d;
  let J = mat3x3<f32>(
    vec3<f32>(fx / d, 0.0, 0.0),
    vec3<f32>(0.0, fy / d, 0.0),
    vec3<f32>(fx * txy.x / (d * d), fy * txy.y / (d * d), 0.0),
  );
  let W = mat3x3<f32>(u.view[0].xyz, u.view[1].xyz, u.view[2].xyz);
  let T = J * W;
  let cov = T * Sigma * transpose(T);
  let a = cov[0][0] + LOW_PASS;
  let bb = cov[0][1];
  let c = cov[1][1] + LOW_PASS;

  let det = a * c - bb * bb;
  if (det <= 0.0) { cull(i); return; }

  // Eigen-decomposition of the 2x2 covariance -> oriented quad.
  let mid = 0.5 * (a + c);
  let disc = sqrt(max(0.0, mid * mid - det));
  let l1 = mid + disc;
  let l2 = max(mid - disc, 0.0);
  var v1 = select(vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), a >= c);
  if (abs(bb) >= 1e-9) { v1 = normalize(vec2<f32>(bb, l1 - a)); }
  let r1 = SIGMA_EXTENT * sqrt(l1);
  let r2 = SIGMA_EXTENT * sqrt(l2);
  let radius = ceil(r1);
  if (center.x + radius < 0.0 || center.x - radius > u.viewport.x ||
      center.y + radius < 0.0 || center.y - radius > u.viewport.y) { cull(i); return; }

  let color = eval_sh(i * u.sh_stride, u.sh_degree, pos - u.cam_pos.xyz);

  var o: Splat2D;
  o.center = center;
  o.axis1 = v1 * r1;
  o.axis2 = vec2<f32>(-v1.y, v1.x) * r2;
  o.color = vec2<u32>(pack2x16float(color.rg), pack2x16float(vec2<f32>(color.b, opacity)));
  o.depth = ndc.z;
  o._pad = 0.0;
  out2d[i] = o;

  // Depth key: positive float bits are monotonic. Invert for back-to-front.
  let bits = bitcast<u32>(d);
  sort_keys[i] = select(~bits, bits, (u.flags & 1u) == 1u);
  sort_vals[i] = i;
  atomicAdd(&draw_args[1], 1u);
}
