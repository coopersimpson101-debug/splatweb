// splatweb GPU radix sort — pass 3/3: stable scatter.
//
// The workgroup walks its 2048-key block in 8 chunks of 256 keys (one key per thread), in order.
// For each chunk:
//   rank   = number of earlier threads in this chunk with the same digit   (stable local rank)
//   dst    = offsets[digit] + rank                                        (global position)
//   offsets[digit] += number of keys with that digit in this chunk
// offsets[] starts as the scanned global histogram for this block, so keys keep their relative
// order across chunks, across blocks and therefore across all 4 LSD passes -> a correct sort.
//
// The local rank is computed by a simple O(256) scan of the chunk's digits in shared memory.
// This is not the fastest possible scheme (see README roadmap: subgroup match / Onesweep) but it
// is simple, portable to every WebGPU implementation and needs no subgroup extension.

struct SortParams {
  count: u32,
  shift: u32,
  num_blocks: u32,
  _pad: u32,
};

const WG: u32 = 256u;
const ITEMS: u32 = 8u;
const BLOCK: u32 = WG * ITEMS;
const NO_DIGIT: u32 = 256u;

@group(0) @binding(0) var<uniform> params: SortParams;
@group(0) @binding(1) var<storage, read> keys_in: array<u32>;
@group(0) @binding(2) var<storage, read> vals_in: array<u32>;
@group(0) @binding(3) var<storage, read_write> keys_out: array<u32>;
@group(0) @binding(4) var<storage, read_write> vals_out: array<u32>;
@group(0) @binding(5) var<storage, read> histogram: array<u32>;

var<workgroup> offsets: array<u32, 256>;
var<workgroup> chunk_digits: array<u32, 256>;
var<workgroup> chunk_hist: array<atomic<u32>, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  offsets[lid] = histogram[lid * params.num_blocks + wid.x];
  atomicStore(&chunk_hist[lid], 0u);
  workgroupBarrier();

  let base = wid.x * BLOCK;
  for (var c = 0u; c < ITEMS; c++) {
    let idx = base + c * WG + lid;
    let valid = idx < params.count;
    var key = 0u;
    var val = 0u;
    var digit = NO_DIGIT;
    if (valid) {
      key = keys_in[idx];
      val = vals_in[idx];
      digit = (key >> params.shift) & 0xffu;
      atomicAdd(&chunk_hist[digit], 1u);
    }
    chunk_digits[lid] = digit;
    workgroupBarrier();

    if (valid) {
      var rank = 0u;
      for (var j = 0u; j < lid; j++) {
        rank += select(0u, 1u, chunk_digits[j] == digit);
      }
      let dst = offsets[digit] + rank;
      keys_out[dst] = key;
      vals_out[dst] = val;
    }
    workgroupBarrier();

    offsets[lid] += atomicLoad(&chunk_hist[lid]);
    atomicStore(&chunk_hist[lid], 0u);
    workgroupBarrier();
  }
}
