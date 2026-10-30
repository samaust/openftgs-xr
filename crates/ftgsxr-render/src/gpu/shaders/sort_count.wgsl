// SPDX-License-Identifier: Apache-2.0

// sort_count.wgsl: §1.4 stage 4, the histogram of one radix pass (E8, D-33). Each
// workgroup of 256 threads takes one block of SORT_BLOCK consecutive keys of the visible
// count, counts the pass's 8-bit digit in a workgroup-local histogram and writes all 256
// bins (zeros included) to `histograms[block * 256 + digit]`. No subgroup operations.
//
// The bindings are `sort.rs`'s one layout, shared by the three sort kernels.

// The pass: the digit is (key >> shift) & 0xff.
struct Params {
    shift: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// The keys per block, `sort.rs`'s `SORT_BLOCK` (four per thread).
override SORT_BLOCK: u32 = 1024u;

const WORKGROUP: u32 = 256u;
const RADIX: u32 = 256u;
// The counters' word holding the visible splat count after stage 3's clamp (`cull.wgsl`).
const SPLAT_COUNT: u32 = 1u;

@group(0) @binding(0) var<storage, read> keys_in: array<u32>;
@group(0) @binding(4) var<storage, read_write> histograms: array<u32>;
@group(0) @binding(5) var<storage, read> counters: array<u32, 12>;
@group(0) @binding(6) var<uniform> params: Params;

var<workgroup> local_histogram: array<atomic<u32>, 256>;

@compute @workgroup_size(256)
fn count(
    @builtin(workgroup_id) block: vec3<u32>,
    @builtin(local_invocation_index) t: u32,
) {
    atomicStore(&local_histogram[t], 0u);
    workgroupBarrier();
    let n = counters[SPLAT_COUNT];
    let first = block.x * SORT_BLOCK;
    for (var i = t; i < SORT_BLOCK; i += WORKGROUP) {
        let e = first + i;
        if e < n {
            let digit = (keys_in[e] >> params.shift) & (RADIX - 1u);
            atomicAdd(&local_histogram[digit], 1u);
        }
    }
    workgroupBarrier();
    histograms[block.x * RADIX + t] = atomicLoad(&local_histogram[t]);
}
