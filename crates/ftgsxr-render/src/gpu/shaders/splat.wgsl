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
// E13 (D-45): p_f is the point on the fragment's own view ray at the record's depth z_c,
// projected into the environment depth camera; the weight is filtered over the 2 × 2 nearest
// texels, each decoded to metres (`occlusion.wgsl`, prepended to this file, Q-340, #601); a
// fragment behind the depth camera or outside its image gets w_occ = 1, and an invalid texel
// weighs 1.
// E7's constants are the instance's model's (D-39), from the per-instance block that
// `encode_prepare` writes.
//
// The per-view record (`cull.wgsl`, 12 words): μ′ (0–1), the conic (a, b, c) (2–4), z_c (5),
// the radii (6–7), c·σ_eff and σ_eff as four half floats (8–9), the instance (10), 0 (11).
//
// Shared-center mode (D-4, Q-400): the vertex stage projects the record's μ^S and Σ^S for
// this eye with E6 in full (`projection.wgsl`, prepended to this file, the cull's own
// arithmetic): the Jacobian clamp, Σ′ + εI, the determinant guard, the opacity-aware
// extent and its cap, the near and far culls and the box cull, with the instance's model's
// constants (D-39). A record E6 culls in this eye gets a degenerate quad, its four corners
// at one point outside the clip volume. The color and the draw order are the midpoint's,
// as the cull wrote them; the depth test and E13 use this eye's z_c. The shared-center
// records (`cull.wgsl`): μ^S (0–2), the key (3), then Σ^S as six f32 (4–9), the color c
// and σ_eff as four half floats (10–11) and the instance (12) in the 16-word layout, or
// Σ^S as six half floats (4–6), the color (7–8) and the instance (9) in the 12-word one.

// E7's constants of one instance's model (D-39), then the centers' (`centers.wgsl`, read
// by stage 6 only).
struct InstanceConstants {
    alpha_max: f32,
    alpha_skip: f32,
    pixel_center_offset: f32,
    colormap: u32,
    pixel_radius: f32,
    inv_span: f32,
    inv_speed: f32,
    pad: f32,
    // E6's constants (D-39), read by the shared-center vertex stage (Q-400).
    near: f32,
    far: f32,
    eps2d: f32,
    jacobian_clamp_margin: f32,
    screen_extent_max: f32,
    pad1: f32,
    pad2: f32,
    pad3: f32,
}

// The view (E6): the camera's image W × H, the target's size, the depth mapping and E13's
// switches.
struct DrawView {
    camera_size: vec2<f32>,
    target_size: vec2<f32>,
    // The engine's near plane for the reverse-Z depth test, 0 without one (Q-87).
    depth_near: f32,
    // 1 when E13's environment depth is sampled, 2 when it is and its decode folds into
    // the weight (Q-446), else 0: w_occ = 1.
    occlusion: u32,
    soft: u32,
    delta_m: f32,
    // The view camera's f_x, f_y, c_x, c_y (E6).
    view_k: vec4<f32>,
    // The depth camera's f_x, f_y, c_x, c_y (E6), unread since #790 (`env_projection`
    // folds them), and its image W × H.
    env_k: vec4<f32>,
    env_size: vec2<f32>,
    // The depth image's nearZ and farZ in metres (farZ < nearZ: infinite).
    env_near: f32,
    env_far: f32,
    // The view's pixels to the depth camera's (Q-446, #790): columns G₀, G₁, G₂ and g of
    // G = K_d·M·K_v⁻¹ and g = K_d·t, with M = R_d·R_vᵀ and t the view's camera space to
    // the depth camera's.
    env_projection: mat4x4<f32>,
    // The view's world-to-camera transform [R_w t_w; 0 1] (E6), read by the shared-center
    // vertex stage (Q-400).
    world_to_camera: mat4x4<f32>,
    // Q-340's decode terms (A, B, C) of the depth image's planes, for Z = A / (B − d·C)
    // (`occlusion.wgsl`'s `decode_terms`, #790), and 0.
    env_decode: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> records: array<u32>;
@group(0) @binding(1) var<storage, read> payloads: array<u32>;
@group(0) @binding(2) var<storage, read> instances: array<InstanceConstants>;
@group(0) @binding(3) var<uniform> view: DrawView;
// E13's environment depth (a 1 × 1 placeholder when `view.occlusion` is 0), read texel by
// texel with `textureLoad` at the 2 × 2 nearest texels (Q-340, #601). The non-filtering
// sampler stays in the layout, unread since #601.
@group(0) @binding(4) var env_depth: texture_2d<f32>;
@group(0) @binding(5) var env_sampler: sampler;

const RECORD_WORDS: u32 = 12u;

// 0 per view (12-word records), 1 shared center (16), 2 shared center with half Σ^S (12).
override RECORD_MODE: u32 = 0u;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) mean: vec2<f32>,
    @location(1) @interpolate(flat) conic: vec3<f32>,
    // c·σ_eff and σ_eff.
    @location(2) @interpolate(flat) color: vec4<f32>,
    // α_max, α_skip, pixel_center_offset, z_c.
    @location(3) @interpolate(flat) constants: vec4<f32>,
}

// The strip's corner (−,−), (+,−), (−,+), (+,+) of vertex `vid`.
fn strip_corner(vid: u32) -> vec2<f32> {
    return vec2<f32>(f32(vid & 1u) * 2.0 - 1.0, f32((vid >> 1u) & 1u) * 2.0 - 1.0);
}

// The quad's corner at μ′ ± r in the camera's pixels, with the depth test's z.
fn quad_position(vid: u32, mean: vec2<f32>, radii: vec2<f32>, depth: f32) -> vec4<f32> {
    let px = mean + strip_corner(vid) * radii;
    let ndc = vec2<f32>(2.0 * px.x / view.camera_size.x - 1.0, 1.0 - 2.0 * px.y / view.camera_size.y);
    var z = 0.0;
    if view.depth_near > 0.0 {
        // Q-87: reverse Z with an infinite far plane, z_ndc = near / z_c.
        z = min(view.depth_near / depth, 1.0);
    }
    return vec4<f32>(ndc, z, 1.0);
}

// This eye's camera (E6) from the view block.
fn eye_view() -> View {
    var v: View;
    let m = view.world_to_camera;
    v.rotation = mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz);
    v.translation = m[3].xyz;
    v.width = view.camera_size.x;
    v.height = view.camera_size.y;
    v.focal = view.view_k.xy;
    v.principal = view.view_k.zw;
    return v;
}

// Shared-center mode (Q-400): the record in slot `slot` projected for this eye.
fn vs_shared(vid: u32, slot: u32) -> VertexOut {
    let b = slot * select(12u, 16u, RECORD_MODE == 1u);
    let mu_s = vec3<f32>(bitcast<f32>(records[b]), bitcast<f32>(records[b + 1u]), bitcast<f32>(records[b + 2u]));
    // Σ^S as (xx, xy, xz, yy, yz, zz).
    var s: array<f32, 6>;
    var color: vec4<f32>;
    var instance: u32;
    if RECORD_MODE == 1u {
        for (var i = 0u; i < 6u; i++) {
            s[i] = bitcast<f32>(records[b + 4u + i]);
        }
        color = vec4<f32>(unpack2x16float(records[b + 10u]), unpack2x16float(records[b + 11u]));
        instance = records[b + 12u];
    } else {
        let s01 = unpack2x16float(records[b + 4u]);
        let s23 = unpack2x16float(records[b + 5u]);
        let s45 = unpack2x16float(records[b + 6u]);
        s = array<f32, 6>(s01.x, s01.y, s23.x, s23.y, s45.x, s45.y);
        color = vec4<f32>(unpack2x16float(records[b + 7u]), unpack2x16float(records[b + 8u]));
        instance = records[b + 9u];
    }
    let cov_s = mat3x3<f32>(
        vec3<f32>(s[0], s[1], s[2]),
        vec3<f32>(s[1], s[3], s[4]),
        vec3<f32>(s[2], s[4], s[5]),
    );
    let k = instances[instance];
    var e6: E6Constants;
    e6.near = k.near;
    e6.far = k.far;
    e6.eps2d = k.eps2d;
    e6.jacobian_clamp_margin = k.jacobian_clamp_margin;
    e6.screen_extent_max = k.screen_extent_max;
    e6.alpha_skip = k.alpha_skip;
    let sigma_eff = color.w;
    let p = project_view(eye_view(), e6, mu_s, cov_s, sigma_eff);
    var out: VertexOut;
    if !p.visible {
        // Culled in this eye: a degenerate quad outside the clip volume.
        out.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return out;
    }
    out.position = quad_position(vid, p.mean, p.radii, p.depth);
    out.mean = p.mean;
    out.conic = p.conic;
    // c·σ_eff and σ_eff, as the per-view record holds them.
    out.color = vec4<f32>(color.xyz * sigma_eff, sigma_eff);
    out.constants = vec4<f32>(k.alpha_max, k.alpha_skip, k.pixel_center_offset, p.depth);
    return out;
}

@vertex
fn vs_splat(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VertexOut {
    if RECORD_MODE != 0u {
        return vs_shared(vid, payloads[iid]);
    }
    let b = payloads[iid] * RECORD_WORDS;
    let mean = vec2<f32>(bitcast<f32>(records[b]), bitcast<f32>(records[b + 1u]));
    let conic = vec3<f32>(bitcast<f32>(records[b + 2u]), bitcast<f32>(records[b + 3u]), bitcast<f32>(records[b + 4u]));
    let depth = bitcast<f32>(records[b + 5u]);
    let radii = vec2<f32>(bitcast<f32>(records[b + 6u]), bitcast<f32>(records[b + 7u]));
    let color = vec4<f32>(unpack2x16float(records[b + 8u]), unpack2x16float(records[b + 9u]));
    let k = instances[records[b + 10u]];

    var out: VertexOut;
    out.position = quad_position(vid, mean, radii, depth);
    out.mean = mean;
    out.conic = conic;
    out.color = color;
    out.constants = vec4<f32>(k.alpha_max, k.alpha_skip, k.pixel_center_offset, depth);
    return out;
}

// E13's weight against one texel of the depth image: its stored depth decoded to metres
// (Q-340), weighed at the depth camera's z of p_f; an invalid texel weighs 1.
fn texel_weight(texel: vec2<i32>, z_f: f32) -> f32 {
    let d = textureLoad(env_depth, texel, 0).x;
    if view.occlusion == 2u {
        return texel_weight_folded(d, z_f);
    }
    let e = depth_metres_terms(d, view.env_decode.xyz);
    return occlusion_weight(z_f, e.metres, e.valid, view.delta_m, view.soft != 0u);
}

// Q-446 (#790): `texel_weight`'s decode folded into the weight, for a draw whose decode
// term A = `env_decode.x` is positive and finite and whose δ is finite (`occlusion` 2, the
// CPU's check). With D = B − d·C the decode is z_e = A / D, so z_e > 0 exactly when D > 0
// (D = +0 gives z_e = +∞, weight 1 either way), and an invalid texel weighs 1 as before.
// Multiplying through by D > 0: the hard step z_f ≤ z_e + δ is (z_f − δ)·D ≤ A, and the
// soft ramp t = (z_f − (z_e − δ)) / (2δ) is ((z_f + δ)·D − A) / (2δ·D), clamped by
// comparing the numerator with 0 and the denominator, so the weight is exactly 1 or 0
// off the ramp and only a texel on it divides. The same weight up to rounding.
// Q-446
fn texel_weight_folded(d: f32, z_f: f32) -> f32 {
    let dd = view.env_decode.y - d * view.env_decode.z;
    if !(d >= 0.0 && d < 1.0 && dd > 0.0) {
        return 1.0;
    }
    let a = view.env_decode.x;
    let delta = max(view.delta_m, 0.0);
    if view.soft != 0u && delta > 0.0 {
        let num = (z_f + delta) * dd - a;
        let den = 2.0 * delta * dd;
        if num <= 0.0 {
            return 1.0;
        }
        if num >= den {
            return 0.0;
        }
        let t = num / den;
        return 1.0 - t * t * (3.0 - 2.0 * t);
    }
    return select(0.0, 1.0, (z_f - delta) * dd <= a);
}

// E13's w_occ for the fragment at camera pixel `p` of a record at depth `z_c` (D-45).
// Q-340 (#601): the weight is filtered over the 2 × 2 nearest texels (percentage-closer
// filtering): E13's weight against each of the four texels whose centres surround p_f's
// projection q, clamped to the image, blended bilinearly by q's position among their
// centres, so an occlusion edge ramps over about one depth texel (D-16's smooth edges).
fn environment_weight(p: vec2<f32>, z_c: f32) -> f32 {
    if view.occlusion == 0u {
        return 1.0;
    }
    // p_f on the fragment's own view ray at the record's depth, x_v = z_c·K_v⁻¹·(p, 1),
    // in the depth camera's space, x_d = M·x_v + t, and its pixel q = (h_x, h_y) / h_z
    // with h = K_d·x_d. Q-446 (#790): the CPU folds G = K_d·M·K_v⁻¹ and g = K_d·t
    // (`env_projection`), so h = z_c·G·(p, 1) + g; h_z is x_d's z, since K_d's last row
    // is (0, 0, 1). The same point up to rounding.
    // Q-446
    let m = view.env_projection;
    let h = z_c * (m[0].xyz * p.x + m[1].xyz * p.y + m[2].xyz) + m[3].xyz;
    if !(h.z > 0.0) {
        return 1.0;
    }
    let q = h.xy / h.z;
    if !(all(q >= vec2<f32>(0.0)) && all(q < view.env_size)) {
        return 1.0;
    }
    // The texel centres are at i + ½: t = q − ½ lies between i₀ and i₀ + 1, f along.
    let t = q - vec2<f32>(0.5);
    let i0 = vec2<i32>(floor(t));
    let f = fract(t);
    let last = vec2<i32>(view.env_size) - vec2<i32>(1);
    let a = clamp(i0, vec2<i32>(0), last);
    let b = clamp(i0 + vec2<i32>(1), vec2<i32>(0), last);
    let w00 = texel_weight(a, h.z);
    let w10 = texel_weight(vec2<i32>(b.x, a.y), h.z);
    let w01 = texel_weight(vec2<i32>(a.x, b.y), h.z);
    let w11 = texel_weight(b, h.z);
    // w + f·(w′ − w), not (1 − f)·w + f·w′: equal weights blend to themselves exactly.
    let top = w00 + f.x * (w10 - w00);
    let bottom = w01 + f.x * (w11 - w01);
    return top + f.y * (bottom - top);
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
    alpha = alpha * environment_weight(p, in.constants.w);
    // (c·σ_eff)·(α/σ_eff) = c·α, premultiplied.
    return vec4<f32>(in.color.xyz * (alpha / sigma_eff), alpha);
}
