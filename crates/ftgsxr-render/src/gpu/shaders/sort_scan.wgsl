// SPDX-License-Identifier: Apache-2.0

// sort_scan.wgsl: §1.4 stage 4, the scan of one radix pass (E8, D-33). A single workgroup
// (§2.4) turns the blocks' histograms (`histograms[block * 256 + digit]`, written by
// `sort_count.wgsl`) in place into scatter offsets, in one walk: thread d walks digit d's
// column and leaves, for every block b ≥ 1, the keys of digit d in blocks 0 to b − 1; the
// 256 digit totals are scanned in workgroup memory, and block 0's row, whose column prefix
// is always 0, receives each digit's base (the keys of smaller digits). `sort_scatter.wgsl`
// adds the base to its block's entry, so (block, digit)'s first output position is
// `histograms[digit] + histograms[block * 256 + digit]` for block ≥ 1 and
// `histograms[digit]` for block 0 (#123). No subgroup operations, no decoupled look-back.
//
// The walk loads UNROLL blocks' entries before it adds them, so the column's loads are in
// flight together instead of one after another: a single workgroup's walk is bound by the
// memory's latency (#123).

struct Params {
    shift: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

override SORT_BLOCK: u32 = 1024u;

const RADIX: u32 = 256u;
// The blocks whose entries the walk loads at once; the loads below are written out.
const UNROLL: u32 = 16u;
const SPLAT_COUNT: u32 = 1u;

@group(0) @binding(4) var<storage, read_write> histograms: array<u32>;
@group(0) @binding(5) var<storage, read> counters: array<u32, 12>;

var<workgroup> totals: array<u32, 256>;

// The scan's dispatch is one workgroup only when a key is visible (Q-85), so `blocks` ≥ 1.
@compute @workgroup_size(256)
fn scan(@builtin(local_invocation_index) d: u32) {
    let blocks = (counters[SPLAT_COUNT] + SORT_BLOCK - 1u) / SORT_BLOCK;
    // Digit d's column from block 1 on: each entry becomes the keys of digit d in the
    // blocks before it, and `total` ends as the column's sum.
    var total = histograms[d];
    let unrolled = 1u + (blocks - 1u) / UNROLL * UNROLL;
    for (var b = 1u; b < unrolled; b += UNROLL) {
        let c0 = histograms[(b + 0u) * RADIX + d];
        let c1 = histograms[(b + 1u) * RADIX + d];
        let c2 = histograms[(b + 2u) * RADIX + d];
        let c3 = histograms[(b + 3u) * RADIX + d];
        let c4 = histograms[(b + 4u) * RADIX + d];
        let c5 = histograms[(b + 5u) * RADIX + d];
        let c6 = histograms[(b + 6u) * RADIX + d];
        let c7 = histograms[(b + 7u) * RADIX + d];
        let c8 = histograms[(b + 8u) * RADIX + d];
        let c9 = histograms[(b + 9u) * RADIX + d];
        let c10 = histograms[(b + 10u) * RADIX + d];
        let c11 = histograms[(b + 11u) * RADIX + d];
        let c12 = histograms[(b + 12u) * RADIX + d];
        let c13 = histograms[(b + 13u) * RADIX + d];
        let c14 = histograms[(b + 14u) * RADIX + d];
        let c15 = histograms[(b + 15u) * RADIX + d];
        histograms[(b + 0u) * RADIX + d] = total;
        total += c0;
        histograms[(b + 1u) * RADIX + d] = total;
        total += c1;
        histograms[(b + 2u) * RADIX + d] = total;
        total += c2;
        histograms[(b + 3u) * RADIX + d] = total;
        total += c3;
        histograms[(b + 4u) * RADIX + d] = total;
        total += c4;
        histograms[(b + 5u) * RADIX + d] = total;
        total += c5;
        histograms[(b + 6u) * RADIX + d] = total;
        total += c6;
        histograms[(b + 7u) * RADIX + d] = total;
        total += c7;
        histograms[(b + 8u) * RADIX + d] = total;
        total += c8;
        histograms[(b + 9u) * RADIX + d] = total;
        total += c9;
        histograms[(b + 10u) * RADIX + d] = total;
        total += c10;
        histograms[(b + 11u) * RADIX + d] = total;
        total += c11;
        histograms[(b + 12u) * RADIX + d] = total;
        total += c12;
        histograms[(b + 13u) * RADIX + d] = total;
        total += c13;
        histograms[(b + 14u) * RADIX + d] = total;
        total += c14;
        histograms[(b + 15u) * RADIX + d] = total;
        total += c15;
    }
    for (var b = unrolled; b < blocks; b++) {
        let i = b * RADIX + d;
        let c = histograms[i];
        histograms[i] = total;
        total += c;
    }
    totals[d] = total;
    workgroupBarrier();
    // An inclusive scan of the 256 totals (Hillis–Steele, in workgroup memory).
    for (var step = 1u; step < RADIX; step <<= 1u) {
        var v = totals[d];
        if d >= step {
            v += totals[d - step];
        }
        workgroupBarrier();
        totals[d] = v;
        workgroupBarrier();
    }
    // Digit d's base, in block 0's row.
    histograms[d] = totals[d] - total;
}
