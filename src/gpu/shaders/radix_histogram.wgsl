// splatweb GPU radix sort — pass 1/3: per-block digit histogram.
//
// Each workgroup owns a block of BLOCK = 256 threads x 8 items = 2048 keys and counts how many
// keys in the block have each 8-bit digit (bits [shift, shift+8)). The result is written
// digit-major:  histogram[digit * num_blocks + block], so a single exclusive scan over the whole
// array yields, for every (digit, block), the global output offset of that block's first key
// with that digit. That is what makes the subsequent scatter stable across blocks.

struct SortParams {
  count: u32,
  shift: u32,
  num_blocks: u32,
  _pad: u32,
};

const WG: u32 = 256u;
const ITEMS: u32 = 8u;
const BLOCK: u32 = WG * ITEMS;

@group(0) @binding(0) var<uniform> params: SortParams;
@group(0) @binding(1) var<storage, read> keys_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> histogram: array<u32>;

var<workgroup> local_hist: array<atomic<u32>, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  atomicStore(&local_hist[lid], 0u);
  workgroupBarrier();
  let base = wid.x * BLOCK;
  for (var c = 0u; c < ITEMS; c++) {
    let idx = base + c * WG + lid;
    if (idx < params.count) {
      let digit = (keys_in[idx] >> params.shift) & 0xffu;
      atomicAdd(&local_hist[digit], 1u);
    }
  }
  workgroupBarrier();
  histogram[lid * params.num_blocks + wid.x] = atomicLoad(&local_hist[lid]);
}
