// SPDX-License-Identifier: Apache-2.0

// cull.wgsl: §1.4 stages 2 (cull, per instance, per dispatch range) and 3 (size, one
// workgroup) of `ftgsxr_render::gpu::SplatPasses::encode_prepare`.
//
// Equations: E1 (the moved position), E2 (temporal and effective opacity, from the rebased
// float32 values; σt ≡ 1 with a single timestamp, D-9), E3 (the exact test σ_eff < α_skip
// on every row visited, D-10), E4 (the SH color at the eye in model space; a zero
// direction stays zero, Q-27), E5 (the pose and Σ^S), E6 (camera space, the near and far
// culls, the Jacobian with its clamp, Σ′ + εI, the determinant guard, the opacity-aware
// extent and its cap, whole-pixel radii, the box cull, the conic as (a, b, c), Q-21, and
// the non-finite cull), E8 (the key k̄ = 2³² − 1 − bits(z)) and E14 (the gains in linear
// light). Every E6 and E7 constant comes from the model's uniform block (D-39).
//
// Records (§1.4 stage 2, words of 32 bits, little-endian):
// - per view (12 words): μ′ (2 f32), the conic (3 f32), z_c (f32), the radii (2 f32),
//   the premultiplied color c·σ_eff and σ_eff as four half floats (2 words), the
//   instance (u32), 0;
// - shared center (16 words): μ^S (3 f32), the key (u32), Σ^S (xx, xy, xz, yy, yz, zz:
//   6 f32), the color c and σ_eff as four half floats (2 words), the instance, 0, 0, 0;
// - shared center with half Σ^S (12 words): μ^S, the key, Σ^S as six half floats (3
//   words), the color (2 words), the instance, 0, 0;
// - a center (Q-78), in the same buffer after the splats, at most 12 words: μ^S (3
//   f32), its depth (f32), μ′ in view 0 (2 f32), time_center − τ₀ (f32), ‖velocity‖ (f32),
//   the DC color C₀c₀₀ + 0.5 and σ_eff as four half floats (2 words), the instance, 0.

// The model's uniform block (`model.rs`, D-39).
struct Model {
    alpha_max: f32,
    eps2d: f32,
    near: f32,
    far: f32,
    alpha_skip: f32,
    pixel_center_offset: f32,
    jacobian_clamp_margin: f32,
    screen_extent_max: f32,
    n: u32,
    sh_degree: u32,
    half_flags: u32,
    single: u32,
}

// One instance (§1.4 stage 1).
struct Instance {
    rotation: mat3x3<f32>,
    translation: vec3<f32>,
    scale: f32,
    eye: vec3<f32>,
    tau: f32,
    gain: vec3<f32>,
    flags: u32,
    persistent: u32,
    transient_start: u32,
    rows: u32,
    sh_degree: u32,
    instance: u32,
    pixel_radius: f32,
    pad0: u32,
    pad1: u32,
}

// A cull view (E6): R_w, t_w, W × H, f, c.
struct ViewCam {
    rotation: mat3x3<f32>,
    translation: vec3<f32>,
    width: f32,
    focal: vec2<f32>,
    principal: vec2<f32>,
    height: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

// The frame: the cull views, C, the shared sort pose (D-4).
struct Frame {
    views: array<ViewCam, 2>,
    view_count: u32,
    capacity: u32,
    pad0: u32,
    pad1: u32,
    sort_origin: vec3<f32>,
    pad2: f32,
    sort_forward: vec3<f32>,
    pad3: f32,
}

// 0 per view (48-byte records), 1 shared center (64), 2 shared center with half Σ^S (48).
override RECORD_MODE: u32 = 0u;

const FLAG_TRANSFORM: u32 = 1u;
const FLAG_COUNT_ACTIVE: u32 = 2u;

// The counters' words (the indirect arguments of `frame.rs`): the splat draw (0–3), the
// centers draw (4–7), the overflow flag (8), the rows that passed E3's test (9).
const SPLAT_COUNT: u32 = 1u;
const CENTER_COUNT: u32 = 5u;
const OVERFLOW: u32 = 8u;
const ACTIVE: u32 = 9u;

// The sort's block, elements per workgroup of its count pass (§1.4 stage 3), set from
// `passes.rs`'s `SORT_BLOCK`.
override SORT_BLOCK: u32 = 256u;

const WORKGROUP: u32 = 256u;

const SH_C0: f32 = 0.28209479177387814;
const SH_C1: f32 = 0.4886025119029199;
const SH_C20: f32 = 0.31539156525252005;
const SH_C21: f32 = 1.0925484305920792;
const SH_C22: f32 = 0.5462742152960396;
const SH_C30: f32 = 0.3731763325901154;
const SH_C31: f32 = 0.4570457994644658;
const SH_C32: f32 = 1.445305721320277;
const SH_C33: f32 = 0.5900435899266435;

@group(0) @binding(0) var<uniform> model: Model;
@group(0) @binding(1) var<storage, read> position: array<f32>;
@group(0) @binding(2) var<storage, read> time_center: array<f32>;
@group(0) @binding(3) var<storage, read> duration: array<f32>;
@group(0) @binding(4) var<storage, read> velocity: array<u32>;
@group(0) @binding(5) var<storage, read> scale: array<u32>;
@group(0) @binding(6) var<storage, read> rotation: array<u32>;
@group(0) @binding(7) var<storage, read> opacity: array<u32>;
@group(0) @binding(8) var<storage, read> sh: array<u32>;

@group(1) @binding(0) var<storage, read_write> records: array<u32>;
@group(1) @binding(1) var<storage, read_write> keys: array<u32>;
@group(1) @binding(2) var<storage, read_write> payloads: array<u32>;
@group(1) @binding(3) var<storage, read_write> counters: array<atomic<u32>, 12>;
@group(1) @binding(4) var<uniform> frame: Frame;

@group(2) @binding(0) var<uniform> inst: Instance;

// Stage 3's own bindings (a pipeline layout of its own).
@group(0) @binding(10) var<storage, read_write> size_counters: array<atomic<u32>, 12>;
@group(0) @binding(11) var<storage, read_write> size_dispatch: array<u32, 3>;
@group(0) @binding(12) var<storage, read_write> size_stats: array<u32, 16>;
@group(0) @binding(13) var<uniform> size_frame: Frame;

fn record_words() -> u32 {
    if RECORD_MODE == 1u {
        return 16u;
    }
    return 12u;
}

// Value m of a packed field (Q-22): float32 words, or half floats two to a word.
fn unpack_value(word: u32, m: u32, half: bool) -> f32 {
    if half {
        return unpack2x16float(word)[m & 1u];
    }
    return bitcast<f32>(word);
}

fn field_index(m: u32, half: bool) -> u32 {
    if half {
        return m >> 1u;
    }
    return m;
}

fn velocity_at(m: u32) -> f32 {
    let half = (model.half_flags & 1u) != 0u;
    return unpack_value(velocity[field_index(m, half)], m, half);
}

fn scale_at(m: u32) -> f32 {
    let half = (model.half_flags & 2u) != 0u;
    return unpack_value(scale[field_index(m, half)], m, half);
}

fn rotation_at(m: u32) -> f32 {
    let half = (model.half_flags & 4u) != 0u;
    return unpack_value(rotation[field_index(m, half)], m, half);
}

fn opacity_at(m: u32) -> f32 {
    let half = (model.half_flags & 8u) != 0u;
    return unpack_value(opacity[field_index(m, half)], m, half);
}

fn sh_at(m: u32) -> f32 {
    let half = (model.half_flags & 16u) != 0u;
    return unpack_value(sh[field_index(m, half)], m, half);
}

fn finite(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}

// The row this invocation visits: the persistent rows 0..P, then the transient run (E3).
fn row_of(wid: vec3<u32>, nwg: vec3<u32>, lid: u32) -> u32 {
    let k = (wid.y * nwg.x + wid.x) * WORKGROUP + lid;
    if k >= inst.rows {
        return 0xffffffffu;
    }
    if k < inst.persistent {
        return k;
    }
    return inst.transient_start + (k - inst.persistent);
}

struct Temporal {
    mu: vec3<f32>,
    sigma_eff: f32,
}

// E1 and E2 from the rebased float32 values (τ − τ₀, time_center − τ₀).
fn temporal(row: u32) -> Temporal {
    var t: Temporal;
    let p = 3u * row;
    t.mu = vec3<f32>(position[p], position[p + 1u], position[p + 2u]);
    var sigma_t = 1.0;
    if model.single == 0u {
        let dt = inst.tau - time_center[row];
        let x = dt / duration[row];
        sigma_t = exp(-0.5 * x * x);
        let v = vec3<f32>(velocity_at(p), velocity_at(p + 1u), velocity_at(p + 2u));
        t.mu = t.mu + v * dt;
    }
    t.sigma_eff = sigma_t * opacity_at(row);
    return t;
}

// R(q) for q = (w, x, y, z), normalized (E5).
fn quat_matrix(q_raw: vec4<f32>) -> mat3x3<f32> {
    let q = normalize(q_raw);
    let w = q.x;
    let x = q.y;
    let y = q.z;
    let z = q.w;
    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)),
        vec3<f32>(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)),
        vec3<f32>(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)),
    );
}

// Σ^S = s² R_m R(q) diag(a)² R(q)ᵀ R_mᵀ (E5).
fn stage_covariance(row: u32) -> mat3x3<f32> {
    let r = 4u * row;
    let rq = quat_matrix(vec4<f32>(rotation_at(r), rotation_at(r + 1u), rotation_at(r + 2u), rotation_at(r + 3u)));
    let s = 3u * row;
    let a = vec3<f32>(scale_at(s), scale_at(s + 1u), scale_at(s + 2u));
    let m = inst.rotation * rq * mat3x3<f32>(
        vec3<f32>(a.x, 0.0, 0.0),
        vec3<f32>(0.0, a.y, 0.0),
        vec3<f32>(0.0, 0.0, a.z),
    );
    let cov = m * transpose(m);
    return cov * (inst.scale * inst.scale);
}

struct Proj {
    visible: bool,
    mean: vec2<f32>,
    conic: vec3<f32>,
    radii: vec2<f32>,
    depth: f32,
}

// E6 for view v, in the order of `ftgsxr_core::project`.
fn project(v: ViewCam, mu_s: vec3<f32>, cov_s: mat3x3<f32>, sigma_eff: f32) -> Proj {
    var p: Proj;
    p.visible = false;
    let xc = v.rotation * mu_s + v.translation;
    let z = xc.z;
    if !finite(z) || z < model.near || z > model.far {
        return p;
    }
    let fx = v.focal.x;
    let fy = v.focal.y;
    let cx = v.principal.x;
    let cy = v.principal.y;
    let u = xc.x / z;
    let w = xc.y / z;
    let mean = vec2<f32>(fx * u + cx, fy * w + cy);
    // The Jacobian at x_c, its ratios clamped m·W/f_x (m·H/f_y) beyond the image; μ′ is not.
    let mx = model.jacobian_clamp_margin * v.width / fx;
    let my = model.jacobian_clamp_margin * v.height / fy;
    let tx = clamp(u, -(cx / fx + mx), (v.width - cx) / fx + mx);
    let ty = clamp(w, -(cy / fy + my), (v.height - cy) / fy + my);
    let j0 = vec3<f32>(fx / z, 0.0, -fx * tx / z);
    let j1 = vec3<f32>(0.0, fy / z, -fy * ty / z);
    let sc = v.rotation * cov_s * transpose(v.rotation);
    let s00 = dot(j0, sc * j0) + model.eps2d;
    let s01 = dot(j0, sc * j1);
    let s11 = dot(j1, sc * j1) + model.eps2d;
    let det = s00 * s11 - s01 * s01;
    if !finite(det) || det <= 0.0 {
        return p;
    }
    // The opacity-aware extent, capped; σ_eff < α_skip has none.
    let e2 = 2.0 * log(sigma_eff / model.alpha_skip);
    if !finite(e2) || e2 < 0.0 {
        return p;
    }
    let extent = min(sqrt(e2), model.screen_extent_max);
    let rx = ceil(extent * sqrt(s00));
    let ry = ceil(extent * sqrt(s11));
    if !finite(rx) || !finite(ry) || rx <= 0.0 || ry <= 0.0 {
        return p;
    }
    if mean.x + rx <= 0.0 || mean.x - rx >= v.width || mean.y + ry <= 0.0 || mean.y - ry >= v.height {
        return p;
    }
    let conic = vec3<f32>(s11 / det, -s01 / det, s00 / det);
    if !finite(mean.x) || !finite(mean.y) || !finite(conic.x) || !finite(conic.y) || !finite(conic.z) {
        return p;
    }
    p.visible = true;
    p.mean = mean;
    p.conic = conic;
    p.radii = vec2<f32>(rx, ry);
    p.depth = z;
    return p;
}

// Coefficient k, channel ch of `row` in the loaded layout (L′, Q-22).
fn coeff(row: u32, k: u32, ch: u32) -> f32 {
    let per_row = (model.sh_degree + 1u) * (model.sh_degree + 1u) * 3u;
    return sh_at(row * per_row + 3u * k + ch);
}

// E4 for one channel at the unit (or zero, Q-27) direction d.
fn sh_channel(row: u32, ch: u32, d: vec3<f32>) -> f32 {
    let x = d.x;
    let y = d.y;
    let z = d.z;
    var c = SH_C0 * coeff(row, 0u, ch);
    let degree = min(inst.sh_degree, model.sh_degree);
    if degree >= 1u {
        c += -SH_C1 * y * coeff(row, 1u, ch) + SH_C1 * z * coeff(row, 2u, ch) - SH_C1 * x * coeff(row, 3u, ch);
    }
    if degree >= 2u {
        c += SH_C21 * x * y * coeff(row, 4u, ch)
            - SH_C21 * y * z * coeff(row, 5u, ch)
            + SH_C20 * (3.0 * z * z - 1.0) * coeff(row, 6u, ch)
            - SH_C21 * x * z * coeff(row, 7u, ch)
            + SH_C22 * (x * x - y * y) * coeff(row, 8u, ch);
    }
    if degree >= 3u {
        c += -SH_C33 * y * (3.0 * x * x - y * y) * coeff(row, 9u, ch)
            + 2.0 * SH_C32 * x * y * z * coeff(row, 10u, ch)
            - SH_C31 * y * (5.0 * z * z - 1.0) * coeff(row, 11u, ch)
            + SH_C30 * z * (5.0 * z * z - 3.0) * coeff(row, 12u, ch)
            - SH_C31 * x * (5.0 * z * z - 1.0) * coeff(row, 13u, ch)
            + SH_C32 * z * (x * x - y * y) * coeff(row, 14u, ch)
            - SH_C33 * x * (x * x - 3.0 * y * y) * coeff(row, 15u, ch);
    }
    return max(0.0, c + 0.5);
}

// IEC 61966-2-1's sRGB decoding and encoding, per channel (§1.14 Color, E14).
fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + 0.055) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

fn linear_to_srgb(l: vec3<f32>) -> vec3<f32> {
    let lo = l * 12.92;
    let hi = 1.055 * pow(l, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, l <= vec3<f32>(0.0031308));
}

// E4 at the eye in model space (o_{v,m}), then E14 when the transform is not the identity.
fn color(row: u32, mu: vec3<f32>) -> vec3<f32> {
    let off = mu - inst.eye;
    let len = length(off);
    var d = vec3<f32>(0.0);
    if len > 0.0 {
        d = off / len;
    }
    var c = vec3<f32>(sh_channel(row, 0u, d), sh_channel(row, 1u, d), sh_channel(row, 2u, d));
    if (inst.flags & FLAG_TRANSFORM) != 0u {
        c = linear_to_srgb(clamp(inst.gain * srgb_to_linear(c), vec3<f32>(0.0), vec3<f32>(1.0)));
    }
    return c;
}

// E8's key from a depth; a depth that is not positive sorts nearest.
fn sort_key(depth: f32) -> u32 {
    if !(depth > 0.0) {
        return 0xffffffffu;
    }
    return 0xffffffffu - bitcast<u32>(depth);
}

fn mark_active() {
    if (inst.flags & FLAG_COUNT_ACTIVE) != 0u {
        atomicAdd(&counters[ACTIVE], 1u);
    }
}

@compute @workgroup_size(256)
fn cull_splats(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let row = row_of(wid, nwg, lid);
    if row == 0xffffffffu {
        return;
    }
    let t = temporal(row);
    // E3's exact test (D-10); a NaN σ_eff falls to the projection's checks.
    if t.sigma_eff < model.alpha_skip {
        return;
    }
    mark_active();
    let mu_s = inst.scale * (inst.rotation * t.mu) + inst.translation;
    let cov_s = stage_covariance(row);

    if RECORD_MODE == 0u {
        let p = project(frame.views[0], mu_s, cov_s, t.sigma_eff);
        if !p.visible {
            return;
        }
        let c = color(row, t.mu);
        let slot = atomicAdd(&counters[SPLAT_COUNT], 1u);
        if slot >= frame.capacity {
            atomicStore(&counters[OVERFLOW], 1u);
            return;
        }
        let b = slot * record_words();
        records[b] = bitcast<u32>(p.mean.x);
        records[b + 1u] = bitcast<u32>(p.mean.y);
        records[b + 2u] = bitcast<u32>(p.conic.x);
        records[b + 3u] = bitcast<u32>(p.conic.y);
        records[b + 4u] = bitcast<u32>(p.conic.z);
        records[b + 5u] = bitcast<u32>(p.depth);
        records[b + 6u] = bitcast<u32>(p.radii.x);
        records[b + 7u] = bitcast<u32>(p.radii.y);
        records[b + 8u] = pack2x16float(c.xy * t.sigma_eff);
        records[b + 9u] = pack2x16float(vec2<f32>(c.z * t.sigma_eff, t.sigma_eff));
        records[b + 10u] = inst.instance;
        records[b + 11u] = 0u;
        keys[slot] = sort_key(p.depth);
        payloads[slot] = slot;
        return;
    }

    // Shared center (D-4): visible when inside either eye's frustum.
    var visible = false;
    for (var i = 0u; i < frame.view_count; i++) {
        if project(frame.views[i], mu_s, cov_s, t.sigma_eff).visible {
            visible = true;
        }
    }
    if !visible {
        return;
    }
    let c = color(row, t.mu);
    let key = sort_key(dot(mu_s - frame.sort_origin, frame.sort_forward));
    let slot = atomicAdd(&counters[SPLAT_COUNT], 1u);
    if slot >= frame.capacity {
        atomicStore(&counters[OVERFLOW], 1u);
        return;
    }
    let b = slot * record_words();
    records[b] = bitcast<u32>(mu_s.x);
    records[b + 1u] = bitcast<u32>(mu_s.y);
    records[b + 2u] = bitcast<u32>(mu_s.z);
    records[b + 3u] = key;
    if RECORD_MODE == 1u {
        records[b + 4u] = bitcast<u32>(cov_s[0].x);
        records[b + 5u] = bitcast<u32>(cov_s[1].x);
        records[b + 6u] = bitcast<u32>(cov_s[2].x);
        records[b + 7u] = bitcast<u32>(cov_s[1].y);
        records[b + 8u] = bitcast<u32>(cov_s[2].y);
        records[b + 9u] = bitcast<u32>(cov_s[2].z);
        records[b + 10u] = pack2x16float(c.xy);
        records[b + 11u] = pack2x16float(vec2<f32>(c.z, t.sigma_eff));
        records[b + 12u] = inst.instance;
        records[b + 13u] = 0u;
        records[b + 14u] = 0u;
        records[b + 15u] = 0u;
    } else {
        records[b + 4u] = pack2x16float(vec2<f32>(cov_s[0].x, cov_s[1].x));
        records[b + 5u] = pack2x16float(vec2<f32>(cov_s[2].x, cov_s[1].y));
        records[b + 6u] = pack2x16float(vec2<f32>(cov_s[2].y, cov_s[2].z));
        records[b + 7u] = pack2x16float(c.xy);
        records[b + 8u] = pack2x16float(vec2<f32>(c.z, t.sigma_eff));
        records[b + 9u] = inst.instance;
        records[b + 10u] = 0u;
        records[b + 11u] = 0u;
    }
    keys[slot] = key;
    payloads[slot] = slot;
}

// Centers mode (Q-78): the rows that pass E3's test, at their moved positions, kept
// when the near and far culls pass and the disc's box μ′ ± ρ is not wholly off the image
// in any cull view; written after every splat, from slot min(splats, C) on.
@compute @workgroup_size(256)
fn cull_centers(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let row = row_of(wid, nwg, lid);
    if row == 0xffffffffu {
        return;
    }
    let t = temporal(row);
    if t.sigma_eff < model.alpha_skip {
        return;
    }
    mark_active();
    let mu_s = inst.scale * (inst.rotation * t.mu) + inst.translation;
    if !finite(mu_s.x) || !finite(mu_s.y) || !finite(mu_s.z) {
        return;
    }
    var visible = false;
    var mean0 = vec2<f32>(0.0);
    var depth0 = 0.0;
    for (var i = 0u; i < frame.view_count; i++) {
        let v = frame.views[i];
        let xc = v.rotation * mu_s + v.translation;
        let z = xc.z;
        let mean = vec2<f32>(v.focal.x * xc.x / z + v.principal.x, v.focal.y * xc.y / z + v.principal.y);
        if i == 0u {
            mean0 = mean;
            depth0 = z;
        }
        let r = inst.pixel_radius;
        let inside = z >= model.near && z <= model.far && finite(mean.x) && finite(mean.y)
            && mean.x + r > 0.0 && mean.x - r < v.width && mean.y + r > 0.0 && mean.y - r < v.height;
        if inside {
            visible = true;
        }
    }
    if !visible {
        return;
    }
    if RECORD_MODE != 0u {
        depth0 = dot(mu_s - frame.sort_origin, frame.sort_forward);
    }
    let j = atomicAdd(&counters[CENTER_COUNT], 1u);
    let slot = min(atomicLoad(&counters[SPLAT_COUNT]), frame.capacity) + j;
    if slot >= frame.capacity {
        atomicStore(&counters[OVERFLOW], 1u);
        return;
    }
    let p = 3u * row;
    let speed = length(vec3<f32>(velocity_at(p), velocity_at(p + 1u), velocity_at(p + 2u)));
    let dc = vec3<f32>(coeff(row, 0u, 0u), coeff(row, 0u, 1u), coeff(row, 0u, 2u)) * SH_C0 + 0.5;
    let b = slot * record_words();
    records[b] = bitcast<u32>(mu_s.x);
    records[b + 1u] = bitcast<u32>(mu_s.y);
    records[b + 2u] = bitcast<u32>(mu_s.z);
    records[b + 3u] = bitcast<u32>(depth0);
    records[b + 4u] = bitcast<u32>(mean0.x);
    records[b + 5u] = bitcast<u32>(mean0.y);
    records[b + 6u] = bitcast<u32>(time_center[row]);
    records[b + 7u] = bitcast<u32>(speed);
    records[b + 8u] = pack2x16float(dc.xy);
    records[b + 9u] = pack2x16float(vec2<f32>(dc.z, t.sigma_eff));
    records[b + 10u] = inst.instance;
    records[b + 11u] = 0u;
}

// Stage 3: the counts clamped to C (splats first, then centers in what remains), the draw
// arguments completed, the sort's dispatch ⌈count / SORT_BLOCK⌉, and the statistics
// block's counts, overflow flag and capacity, its timestamps zeroed (§1.4 stage 8).
@compute @workgroup_size(1)
fn size() {
    let cap = size_frame.capacity;
    let raw_s = atomicLoad(&size_counters[SPLAT_COUNT]);
    let s = min(raw_s, cap);
    let raw_c = atomicLoad(&size_counters[CENTER_COUNT]);
    let c = min(raw_c, cap - s);
    var over = atomicLoad(&size_counters[OVERFLOW]);
    if raw_s > s || raw_c > c {
        over = 1u;
    }
    atomicStore(&size_counters[0], 4u);
    atomicStore(&size_counters[SPLAT_COUNT], s);
    atomicStore(&size_counters[2], 0u);
    atomicStore(&size_counters[3], 0u);
    atomicStore(&size_counters[4], 4u);
    atomicStore(&size_counters[CENTER_COUNT], c);
    atomicStore(&size_counters[6], 0u);
    atomicStore(&size_counters[7], 0u);
    atomicStore(&size_counters[OVERFLOW], over);
    size_dispatch[0] = (s + SORT_BLOCK - 1u) / SORT_BLOCK;
    size_dispatch[1] = 1u;
    size_dispatch[2] = 1u;
    size_stats[0] = s;
    size_stats[1] = c;
    size_stats[2] = over;
    size_stats[3] = cap;
    for (var i = 4u; i < 16u; i++) {
        size_stats[i] = 0u;
    }
}
