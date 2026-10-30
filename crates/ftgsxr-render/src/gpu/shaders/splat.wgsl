// SPDX-License-Identifier: Apache-2.0

// splat.wgsl: §1.4 stage 5, the splat part of `ftgsxr_render::gpu::SplatPasses::encode_draw`.
//
// One indirect instanced draw of four-vertex strips in the sorted order (D-3, D-48): instance
// i draws the record in slot `payloads[i]` (the sort's result, pair 0). The vertex stage
// places the quad at μ′ ± r in the camera's pixels (E6, continuous pixel convention; image
// row 0 at the top). The fragment stage evaluates E7's α at the pixel center with the conic,
// skips a negative (or NaN) quadratic form and α < α_skip, clamps at α_max, then multiplies
// by E13's weight w_occ (mirroring `ftgsxr_core::occlusion_weight`, Q-63; 1 when occlusion
// is off), and writes the premultiplied (c·α, α) for the `One, OneMinusSrcAlpha` blend.
// E7's constants are the instance's model's (D-39), from the per-instance block that
// `encode_prepare` writes.
//
// The per-view record (`cull.wgsl`, 12 words): μ′ (0–1), the conic (a, b, c) (2–4), z_c (5),
// the radii (6–7), c·σ_eff and σ_eff as four half floats (8–9), the instance (10), 0 (11).

// E7's constants of one instance's model (D-39).
struct InstanceConstants {
    alpha_max: f32,
    alpha_skip: f32,
    pixel_center_offset: f32,
    pad: f32,
}

// The view (E6): the camera's image W × H, the target's size, the depth mapping and E13's
// switches.
struct DrawView {
    camera_size: vec2<f32>,
    target_size: vec2<f32>,
    // The engine's near plane for the reverse-Z depth test, 0 without one (Q-87).
    depth_near: f32,
    // 1 when E13's environment depth is sampled (M-6), else 0: w_occ = 1.
    occlusion: u32,
    soft: u32,
    delta_m: f32,
}

@group(0) @binding(0) var<storage, read> records: array<u32>;
@group(0) @binding(1) var<storage, read> payloads: array<u32>;
@group(0) @binding(2) var<storage, read> instances: array<InstanceConstants>;
@group(0) @binding(3) var<uniform> view: DrawView;

const RECORD_WORDS: u32 = 12u;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) mean: vec2<f32>,
    @location(1) @interpolate(flat) conic: vec3<f32>,
    // c·σ_eff and σ_eff.
    @location(2) @interpolate(flat) color: vec4<f32>,
    // α_max, α_skip, pixel_center_offset, z_c.
    @location(3) @interpolate(flat) constants: vec4<f32>,
}

@vertex
fn vs_splat(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VertexOut {
    let b = payloads[iid] * RECORD_WORDS;
    let mean = vec2<f32>(bitcast<f32>(records[b]), bitcast<f32>(records[b + 1u]));
    let conic = vec3<f32>(bitcast<f32>(records[b + 2u]), bitcast<f32>(records[b + 3u]), bitcast<f32>(records[b + 4u]));
    let depth = bitcast<f32>(records[b + 5u]);
    let radii = vec2<f32>(bitcast<f32>(records[b + 6u]), bitcast<f32>(records[b + 7u]));
    let color = vec4<f32>(unpack2x16float(records[b + 8u]), unpack2x16float(records[b + 9u]));
    let k = instances[records[b + 10u]];

    // The strip's corners (−,−), (+,−), (−,+), (+,+).
    let corner = vec2<f32>(f32(vid & 1u) * 2.0 - 1.0, f32((vid >> 1u) & 1u) * 2.0 - 1.0);
    let px = mean + corner * radii;
    let ndc = vec2<f32>(2.0 * px.x / view.camera_size.x - 1.0, 1.0 - 2.0 * px.y / view.camera_size.y);
    var z = 0.0;
    if view.depth_near > 0.0 {
        // Q-87: reverse Z with an infinite far plane, z_ndc = near / z_c.
        z = min(view.depth_near / depth, 1.0);
    }
    var out: VertexOut;
    out.position = vec4<f32>(ndc, z, 1.0);
    out.mean = mean;
    out.conic = conic;
    out.color = color;
    out.constants = vec4<f32>(k.alpha_max, k.alpha_skip, k.pixel_center_offset, depth);
    return out;
}

// E13's weight (`ftgsxr_core::occlusion_weight`, Q-63): hard, 1 up to z_e + δ; soft,
// 1 − smoothstep(z_e − δ, z_e + δ, z_f), the hard step at δ ≤ 0; 1 without a valid sample.
fn occlusion_weight(z_f: f32, z_e: f32, valid: bool, delta_m: f32, soft: bool) -> f32 {
    if !valid {
        return 1.0;
    }
    if soft && delta_m > 0.0 {
        let t = clamp((z_f - (z_e - delta_m)) / (2.0 * delta_m), 0.0, 1.0);
        return 1.0 - t * t * (3.0 - 2.0 * t);
    }
    if z_f <= z_e + delta_m {
        return 1.0;
    }
    return 0.0;
}

@fragment
fn fs_splat(in: VertexOut) -> @location(0) vec4<f32> {
    let alpha_max = in.constants.x;
    let alpha_skip = in.constants.y;
    // The pixel's sample point in the camera's pixels: `position.xy` is the target pixel's
    // center (i + 0.5, j + 0.5); the model's offset replaces the 0.5 (D-39).
    let scale = view.camera_size / view.target_size;
    let p = (in.position.xy - vec2<f32>(0.5) + vec2<f32>(in.constants.z)) * scale;
    let d = p - in.mean;
    let form = in.conic.x * d.x * d.x + 2.0 * in.conic.y * d.x * d.y + in.conic.z * d.y * d.y;
    // E7's skips: a negative or NaN form, then α < α_skip, before the occlusion weight.
    if !(form >= 0.0) {
        discard;
    }
    let sigma_eff = in.color.w;
    let raw = sigma_eff * exp(-0.5 * form);
    if !(raw >= alpha_skip) {
        discard;
    }
    var alpha = min(alpha_max, raw);
    if alpha < alpha_skip {
        discard;
    }
    // M-2 samples no environment depth (occlusion is off in every M-2 tier): w_occ = 1.
    alpha = alpha * occlusion_weight(in.constants.w, 0.0, false, view.delta_m, view.soft != 0u);
    // (c·σ_eff)·(α/σ_eff) = c·α, premultiplied.
    return vec4<f32>(in.color.xyz * (alpha / sigma_eff), alpha);
}
