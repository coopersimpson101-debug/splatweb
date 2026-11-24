// splatweb GPU radix sort — pass 2/3: multi-level exclusive prefix sum.
//
// scan_block: each workgroup scans 512 elements (256 threads x 2) in shared memory with the
//             work-efficient Blelloch up-sweep / down-sweep, writes the scanned values back and
//             stores the block total in block_sums[block].
// add_sums:   adds the (already scanned) block_sums[block] to every element of the block.
//
// The host scans level 0 (the histogram), then the block sums, then their block sums ... until a
// level fits in one workgroup, and then walks back down with add_sums.

struct ScanParams {
  count: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
};

const WG: u32 = 256u;
const N: u32 = 512u;

@group(0) @binding(0) var<uniform> params: ScanParams;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read_write> block_sums: array<u32>;

var<workgroup> temp: array<u32, 512>;

@compute @workgroup_size(256)
fn scan_block(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let base = wid.x * N;
  let i0 = base + 2u * lid;
  let i1 = i0 + 1u;
  temp[2u * lid] = select(0u, data[min(i0, params.count - 1u)], i0 < params.count);
  temp[2u * lid + 1u] = select(0u, data[min(i1, params.count - 1u)], i1 < params.count);

  // Up-sweep (reduce).
  var offset = 1u;
  for (var d = N >> 1u; d > 0u; d = d >> 1u) {
    workgroupBarrier();
    if (lid < d) {
      let ai = offset * (2u * lid + 1u) - 1u;
      let bi = offset * (2u * lid + 2u) - 1u;
      temp[bi] += temp[ai];
    }
    offset = offset * 2u;
  }
  workgroupBarrier();
  if (lid == 0u) {
    block_sums[wid.x] = temp[N - 1u];
    temp[N - 1u] = 0u;
  }
  // Down-sweep.
  for (var d = 1u; d < N; d = d * 2u) {
    offset = offset >> 1u;
    workgroupBarrier();
    if (lid < d) {
      let ai = offset * (2u * lid + 1u) - 1u;
      let bi = offset * (2u * lid + 2u) - 1u;
      let t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  workgroupBarrier();
  if (i0 < params.count) { data[i0] = temp[2u * lid]; }
  if (i1 < params.count) { data[i1] = temp[2u * lid + 1u]; }
}

@compute @workgroup_size(256)
fn add_sums(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let base = wid.x * N;
  let add = block_sums[wid.x];
  let i0 = base + 2u * lid;
  let i1 = i0 + 1u;
  if (i0 < params.count) { data[i0] += add; }
  if (i1 < params.count) { data[i1] += add; }
}
