// SPDX-License-Identifier: Apache-2.0

// probe.wgsl: the layer probe of E14's exposure match (§2.1 `LayerSums`, R-21).
//
// One workgroup per cell of the 64 × 48 grid over the layer: cell (i, j) holds the pixels
// with ⌊64 x / W⌋ = i and ⌊48 y / H⌋ = j, so x ∈ [⌈i W / 64⌉, ⌈(i + 1) W / 64⌉) and likewise
// in y. Each of the 256 invocations sums a strided share of the cell's pixels; the
// partial sums are reduced in workgroup memory and the cell written row-major from the
// top-left. A pixel with α > 0 adds (α lin(C/α), α): the layer's premultiplied
// file-encoded color un-premultiplied and decoded by the resolve's sRGB curve (D-6,
// `resolve.wgsl`'s passthrough form), weighted by α. After the cells, workgroup (0, 0)
// writes the mark (1, W, H, 0): a read back without it was not probed (the Bevy adapter's
// one-shot readback, `bevy/probe.rs`).

@group(0) @binding(0) var layer: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> cells: array<vec4<f32>>;

const GRID: vec2<u32> = vec2<u32>(64u, 48u);
const THREADS: u32 = 256u;

var<workgroup> partial: array<vec4<f32>, 256>;

// IEC 61966-2-1's sRGB decoding per channel; negative values are taken as 0 (as the
// resolve's).
fn srgb_to_linear(c_raw: vec3<f32>) -> vec3<f32> {
    let c = max(c_raw, vec3<f32>(0.0));
    let lo = c / 12.92;
    let hi = pow((c + 0.055) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

// ⌈n s / g⌉, the first pixel of cell n of g over s pixels.
fn first(n: u32, s: u32, g: u32) -> u32 {
    return (n * s + g - 1u) / g;
}

@compute @workgroup_size(256)
fn cs_probe(@builtin(workgroup_id) cell: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let size = textureDimensions(layer);
    let x0 = first(cell.x, size.x, GRID.x);
    let x1 = first(cell.x + 1u, size.x, GRID.x);
    let y0 = first(cell.y, size.y, GRID.y);
    let y1 = first(cell.y + 1u, size.y, GRID.y);
    let w = x1 - x0;
    let count = w * (y1 - y0);
    var sum = vec4<f32>(0.0);
    for (var k = t; k < count; k += THREADS) {
        let p = vec2<u32>(x0 + k % w, y0 + k / w);
        let texel = textureLoad(layer, p, 0);
        let a = texel.a;
        if a > 0.0 {
            sum += vec4<f32>(srgb_to_linear(texel.rgb / a) * a, a);
        }
    }
    partial[t] = sum;
    workgroupBarrier();
    for (var s = THREADS / 2u; s > 0u; s /= 2u) {
        if t < s {
            partial[t] += partial[t + s];
        }
        workgroupBarrier();
    }
    if t == 0u {
        cells[cell.y * GRID.x + cell.x] = partial[0];
        if cell.x == 0u && cell.y == 0u {
            cells[GRID.x * GRID.y] = vec4<f32>(1.0, f32(size.x), f32(size.y), 0.0);
        }
    }
}
