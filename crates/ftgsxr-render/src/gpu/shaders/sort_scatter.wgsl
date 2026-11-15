// SPDX-License-Identifier: Apache-2.0

// sort_scatter.wgsl: §1.4 stage 4, the stable scatter of one radix pass (E8, D-33). Each
// workgroup of 256 threads takes the block `sort_count.wgsl` counted, four consecutive keys
// per thread (Q-85's block of 1024): thread t holds keys 4t to 4t + 3 and writes their four
// 8-bit digits as one word of `packed` in workgroup memory. After one barrier a key's rank
// among the block's keys of its digit is the number of equal digits before it: in its own
// word's earlier bytes, and in the words of threads 0 to t − 1, counted four bytes at a time
// (`zero_bytes`). Every thread of a wave reads the same word of `packed` at once. Ranks
// follow key order, so keys of equal digit keep their order (the sort is stable). The key
// and its payload go to the digit's base and the block's column prefix from
// `sort_scan.wgsl`, plus the rank. No atomics, no subgroup operations, no decoupled
// look-back (#123).
//
// Keys past the visible count are never read: their digits are 0, but they sit after
// every live key of the last block, so no live key's rank counts them, and they are not
// written.

struct Params {
    shift: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

override SORT_BLOCK: u32 = 1024u;

const RADIX: u32 = 256u;
const SPLAT_COUNT: u32 = 1u;

@group(0) @binding(0) var<storage, read> keys_in: array<u32>;
@group(0) @binding(1) var<storage, read> payloads_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> keys_out: array<u32>;
@group(0) @binding(3) var<storage, read_write> payloads_out: array<u32>;
@group(0) @binding(4) var<storage, read_write> histograms: array<u32>;
@group(0) @binding(5) var<storage, read> counters: array<u32, 12>;
@group(0) @binding(6) var<uniform> params: Params;

// Digit d's first output position in this block: its base (block 0's row) plus, past
// block 0, the keys of digit d in the earlier blocks.
var<workgroup> offsets: array<u32, 256>;
// Thread t's four digits, one byte each, keys 4t to 4t + 3 of the block.
var<workgroup> packed: array<u32, 256>;

// The number of zero bytes in x, exact (no carries between bytes).
fn zero_bytes(x: u32) -> u32 {
    let y = (x & 0x7f7f7f7fu) + 0x7f7f7f7fu;
    return countOneBits(~(y | x | 0x7f7f7f7fu));
}

@compute @workgroup_size(256)
fn scatter(
    @builtin(workgroup_id) block: vec3<u32>,
    @builtin(local_invocation_index) t: u32,
) {
    let n = counters[SPLAT_COUNT];
    let first = block.x * SORT_BLOCK + 4u * t;
    var keys: array<u32, 4>;
    var digits: array<u32, 4>;
    var word = 0u;
    for (var k = 0u; k < 4u; k++) {
        if first + k < n {
            keys[k] = keys_in[first + k];
        }
        digits[k] = (keys[k] >> params.shift) & (RADIX - 1u);
        word |= digits[k] << (8u * k);
    }
    packed[t] = word;
    var start = histograms[t];
    if block.x > 0u {
        start += histograms[block.x * RADIX + t];
    }
    offsets[t] = start;
    workgroupBarrier();
    let p0 = digits[0] * 0x01010101u;
    let p1 = digits[1] * 0x01010101u;
    let p2 = digits[2] * 0x01010101u;
    let p3 = digits[3] * 0x01010101u;
    var r0 = 0u;
    var r1 = select(0u, 1u, digits[1] == digits[0]);
    var r2 = select(0u, 1u, digits[2] == digits[0]) + select(0u, 1u, digits[2] == digits[1]);
    var r3 = select(0u, 1u, digits[3] == digits[0]) + select(0u, 1u, digits[3] == digits[1])
        + select(0u, 1u, digits[3] == digits[2]);
    for (var w = 0u; w < t; w++) {
        let x = packed[w];
        r0 += zero_bytes(x ^ p0);
        r1 += zero_bytes(x ^ p1);
        r2 += zero_bytes(x ^ p2);
        r3 += zero_bytes(x ^ p3);
    }
    let ranks = array<u32, 4>(r0, r1, r2, r3);
    for (var k = 0u; k < 4u; k++) {
        let e = first + k;
        if e < n {
            let dst = offsets[digits[k]] + ranks[k];
            keys_out[dst] = keys[k];
            payloads_out[dst] = payloads_in[e];
        }
    }
}
