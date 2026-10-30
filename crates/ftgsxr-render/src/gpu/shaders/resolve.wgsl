// SPDX-License-Identifier: Apache-2.0

// resolve.wgsl: §1.4 stage 7, `ftgsxr_render::gpu::SplatPasses::encode_resolve` (D-6, D-14).
//
// One full-screen triangle over the engine's view target. The layer holds file-encoded
// premultiplied values (C, A) (E7, §1.14 Color). Each pixel's color is un-premultiplied,
// decoded from sRGB to linear light (IEC 61966-2-1) and re-premultiplied, (lin(C/A)·A, A),
// and composited over the target with the over operator (`One, OneMinusSrcAlpha` on color
// and alpha), so an sRGB target shows the file's values unchanged and keeps the layer's
// alpha for the compositor (passthrough, D-14). With a background (VR mode) E7's
// C_final = C + (1 − A)·c_bg is formed in the file's space first and decoded, opaque:
// (lin(C_final), 1) (Q-94). The layer is sampled at the target pixel's center through a
// linear sampler, so a layer at another render scale is resampled (Q-95).

struct Resolve {
    background: vec3<f32>,
    // 1 with a background (VR mode), 0 without (passthrough).
    has_background: u32,
}

@group(0) @binding(0) var layer: texture_2d<f32>;
@group(0) @binding(1) var layer_sampler: sampler;
@group(0) @binding(2) var<uniform> resolve: Resolve;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// The full-screen triangle (−1, −1), (3, −1), (−1, 3); uv (0, 0) at the target's top left.
@vertex
fn vs_resolve(@builtin(vertex_index) vid: u32) -> VertexOut {
    let xy = vec2<f32>(f32((vid << 1u) & 2u), f32(vid & 2u));
    var out: VertexOut;
    out.position = vec4<f32>(xy * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(xy.x, 1.0 - xy.y);
    return out;
}

// IEC 61966-2-1's sRGB decoding per channel; negative values are taken as 0.
fn srgb_to_linear(c_raw: vec3<f32>) -> vec3<f32> {
    let c = max(c_raw, vec3<f32>(0.0));
    let lo = c / 12.92;
    let hi = pow((c + 0.055) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

@fragment
fn fs_resolve(in: VertexOut) -> @location(0) vec4<f32> {
    let texel = textureSample(layer, layer_sampler, in.uv);
    let a = texel.a;
    if resolve.has_background != 0u {
        return vec4<f32>(srgb_to_linear(texel.rgb + (1.0 - a) * resolve.background), 1.0);
    }
    if !(a > 0.0) {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(srgb_to_linear(texel.rgb / a) * a, a);
}
