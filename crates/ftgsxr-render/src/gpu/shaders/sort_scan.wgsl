// SPDX-License-Identifier: Apache-2.0

// sort_scan.wgsl: §1.4 stage 4, the scan of one radix pass (E8, D-33). A single workgroup
// turns the blocks' histograms (`histograms[block * 256 + digit]`, written by
// `sort_count.wgsl`) in place into each (block, digit)'s first output position: the keys
// of smaller digits, plus the keys of the same digit in earlier blocks. Thread d walks
// digit d's column, the 256 digit totals are scanned in workgroup memory, and each column
// gets its digit's base. No subgroup operations, no decoupled look-back.

struct Params {
    shift: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

override SORT_BLOCK: u32 = 1024u;

const RADIX: u32 = 256u;
const SPLAT_COUNT: u32 = 1u;

@group(0) @binding(4) var<storage, read_write> histograms: array<u32>;
@group(0) @binding(5) var<storage, read> counters: array<u32, 12>;

var<workgroup> totals: array<u32, 256>;

@compute @workgroup_size(256)
fn scan(@builtin(local_invocation_index) d: u32) {
    let blocks = (counters[SPLAT_COUNT] + SORT_BLOCK - 1u) / SORT_BLOCK;
    // Digit d's column: the exclusive prefix over the blocks, and its total.
    var total = 0u;
    for (var b = 0u; b < blocks; b++) {
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
    let base = totals[d] - total;
    for (var b = 0u; b < blocks; b++) {
        histograms[b * RADIX + d] += base;
    }
}
