// SPDX-License-Identifier: Apache-2.0

// sort_scatter.wgsl: §1.4 stage 4, the stable scatter of one radix pass (E8, D-33). Each
// workgroup of 256 threads takes the block `sort_count.wgsl` counted, in four rounds of
// 256 consecutive keys, one per thread. In a round every thread sets its bit in its
// digit's 256-bit mask in workgroup memory; a key's rank among the round's keys of its
// digit is the number of bits set below its own, so keys of equal digit keep their order
// (the sort is stable). The key and its payload go to the digit's offset from
// `sort_scan.wgsl`, plus the keys of that digit in the block's earlier rounds, plus the
// rank. No subgroup operations, no decoupled look-back.

struct Params {
    shift: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

override SORT_BLOCK: u32 = 1024u;

const WORKGROUP: u32 = 256u;
const RADIX: u32 = 256u;
// A digit's mask: 256 bits, one per thread of the round.
const MASK_WORDS: u32 = 8u;
const SPLAT_COUNT: u32 = 1u;

@group(0) @binding(0) var<storage, read> keys_in: array<u32>;
@group(0) @binding(1) var<storage, read> payloads_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> keys_out: array<u32>;
@group(0) @binding(3) var<storage, read_write> payloads_out: array<u32>;
@group(0) @binding(4) var<storage, read_write> histograms: array<u32>;
@group(0) @binding(5) var<storage, read> counters: array<u32, 12>;
@group(0) @binding(6) var<uniform> params: Params;

var<workgroup> masks: array<atomic<u32>, 2048>;
// Digit d's next output position in this block.
var<workgroup> offsets: array<u32, 256>;

@compute @workgroup_size(256)
fn scatter(
    @builtin(workgroup_id) block: vec3<u32>,
    @builtin(local_invocation_index) t: u32,
) {
    let n = counters[SPLAT_COUNT];
    offsets[t] = histograms[block.x * RADIX + t];
    let word = t / 32u;
    let below = (1u << (t % 32u)) - 1u;
    for (var round = 0u; round < SORT_BLOCK / WORKGROUP; round++) {
        for (var w = 0u; w < MASK_WORDS; w++) {
            atomicStore(&masks[t * MASK_WORDS + w], 0u);
        }
        workgroupBarrier();
        let e = block.x * SORT_BLOCK + round * WORKGROUP + t;
        let live = e < n;
        var key = 0u;
        var digit = 0u;
        if live {
            key = keys_in[e];
            digit = (key >> params.shift) & (RADIX - 1u);
            atomicOr(&masks[digit * MASK_WORDS + word], 1u << (t % 32u));
        }
        workgroupBarrier();
        if live {
            var rank = countOneBits(atomicLoad(&masks[digit * MASK_WORDS + word]) & below);
            for (var w = 0u; w < word; w++) {
                rank += countOneBits(atomicLoad(&masks[digit * MASK_WORDS + w]));
            }
            let dst = offsets[digit] + rank;
            keys_out[dst] = key;
            payloads_out[dst] = payloads_in[e];
        }
        workgroupBarrier();
        // Thread t moves digit t's offset past this round's keys of digit t.
        var count = 0u;
        for (var w = 0u; w < MASK_WORDS; w++) {
            count += countOneBits(atomicLoad(&masks[t * MASK_WORDS + w]));
        }
        offsets[t] += count;
        workgroupBarrier();
    }
}
