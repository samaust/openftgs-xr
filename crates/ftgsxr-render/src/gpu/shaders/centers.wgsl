// SPDX-License-Identifier: Apache-2.0

// centers.wgsl: §1.4 stage 6, the centers part of `ftgsxr_render::gpu::SplatPasses::encode_draw`.
//
// One indirect instanced draw of four-vertex strips, one disc per center record (Q-78):
// instance i draws the record in slot `visible_splats + i`, where `visible_splats` is the
// splat draw's instance count after stage 3's clamp (the draw's first instance is 0).
// Centers need no sort. The vertex stage places the quad at μ′ ± ρ in the camera's pixels
// (E6's projection of the moved position μx(τ), E1 and E5, done by the cull); the fragment
// stage keeps the pixels whose sample point (the splat draw's, D-39's pixel center offset)
// lies within ρ of μ′ and writes the instance's colormap color, opaque: (c, 1), with the
// depth written (reverse Z, Q-87). Colormaps (Q-92): the DC color C₀c₀₀ + 0.5 as the
// record holds it; the time center over [τ₀, τ₁], the speed over [0, its 99th percentile]
// and σ_eff over [0, 1], each clamped to [0, 1] and mapped through viridis.
//
// A center record (`cull.wgsl`, Q-78), at the record layout's stride: μ^S (0–2), the depth
// (3), μ′ in view 0 (4–5), time_center − τ₀ (6), ‖velocity‖ (7), the DC color and σ_eff as
// four half floats (8–9), the instance (10).

// One instance's draw constants (`splat.wgsl`'s `InstanceConstants`, Q-87, Q-92).
struct InstanceConstants {
    alpha_max: f32,
    alpha_skip: f32,
    pixel_center_offset: f32,
    // 0 DC, 1 time center, 2 speed, 3 opacity.
    colormap: u32,
    // ρ in the camera's pixels.
    pixel_radius: f32,
    // 1 / (τ₁ − τ₀), 0 for a single timestamp (Q-97).
    inv_span: f32,
    // 1 / the speeds' 99th percentile, 0 when it is 0.
    inv_speed: f32,
    pad: f32,
}

// The view (`splat.wgsl`'s `DrawView`).
struct DrawView {
    camera_size: vec2<f32>,
    target_size: vec2<f32>,
    depth_near: f32,
    occlusion: u32,
    soft: u32,
    delta_m: f32,
}

// 0 per view (12-word records), 1 shared center (16), 2 shared center with half Σ^S (12).
override RECORD_MODE: u32 = 0u;

@group(0) @binding(0) var<storage, read> records: array<u32>;
// The counters (`frame.rs`): word 1 is the splat draw's clamped instance count.
@group(0) @binding(1) var<storage, read> counters: array<u32, 12>;
@group(0) @binding(2) var<storage, read> instances: array<InstanceConstants>;
@group(0) @binding(3) var<uniform> view: DrawView;

fn record_words() -> u32 {
    if RECORD_MODE == 1u {
        return 16u;
    }
    return 12u;
}

// Viridis at 17 evenly spaced points, entries round(k · 255 / 16) of matplotlib's 256-entry
// `_viridis_data` (matplotlib v3.9.2, lib/matplotlib/_cm_listed.py; matplotlib license, PSF-
// style, NOTICE), sRGB-encoded display values like the file's colors (§1.14 Color).
const VIRIDIS = array<vec3<f32>, 17>(
    vec3<f32>(0.267004, 0.004874, 0.329415),
    vec3<f32>(0.282327, 0.094955, 0.417331),
    vec3<f32>(0.278826, 0.175490, 0.483397),
    vec3<f32>(0.258965, 0.251537, 0.524736),
    vec3<f32>(0.229739, 0.322361, 0.545706),
    vec3<f32>(0.199430, 0.387607, 0.554642),
    vec3<f32>(0.172719, 0.448791, 0.557885),
    vec3<f32>(0.149039, 0.508051, 0.557250),
    vec3<f32>(0.127568, 0.566949, 0.550556),
    vec3<f32>(0.120081, 0.622161, 0.534946),
    vec3<f32>(0.153894, 0.680203, 0.504172),
    vec3<f32>(0.239374, 0.735588, 0.455688),
    vec3<f32>(0.360741, 0.785964, 0.387814),
    vec3<f32>(0.506271, 0.828786, 0.300362),
    vec3<f32>(0.668054, 0.861999, 0.196293),
    vec3<f32>(0.835270, 0.886029, 0.102646),
    vec3<f32>(0.993248, 0.906157, 0.143936),
);

// Viridis at t ∈ [0, 1] (clamped; NaN reads as 0), linear between the 17 points.
fn viridis(t_raw: f32) -> vec3<f32> {
    var t = 0.0;
    if t_raw > 0.0 {
        t = min(t_raw, 1.0);
    }
    let x = t * 16.0;
    let i = min(u32(floor(x)), 15u);
    var table = VIRIDIS;
    return mix(table[i], table[i + 1u], x - f32(i));
}

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) mean: vec2<f32>,
    @location(1) @interpolate(flat) color: vec3<f32>,
    // ρ, pixel_center_offset.
    @location(2) @interpolate(flat) disc: vec2<f32>,
}

@vertex
fn vs_center(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VertexOut {
    let b = (counters[1] + iid) * record_words();
    let depth = bitcast<f32>(records[b + 3u]);
    let mean = vec2<f32>(bitcast<f32>(records[b + 4u]), bitcast<f32>(records[b + 5u]));
    let time_center = bitcast<f32>(records[b + 6u]);
    let speed = bitcast<f32>(records[b + 7u]);
    let dc_rg = unpack2x16float(records[b + 8u]);
    let dc_b_sigma = unpack2x16float(records[b + 9u]);
    let k = instances[records[b + 10u]];

    var color = vec3<f32>(dc_rg, dc_b_sigma.x);
    if k.colormap == 1u {
        color = viridis(time_center * k.inv_span);
    } else if k.colormap == 2u {
        color = viridis(speed * k.inv_speed);
    } else if k.colormap == 3u {
        color = viridis(dc_b_sigma.y);
    }

    let corner = vec2<f32>(f32(vid & 1u) * 2.0 - 1.0, f32((vid >> 1u) & 1u) * 2.0 - 1.0);
    let px = mean + corner * k.pixel_radius;
    let ndc = vec2<f32>(2.0 * px.x / view.camera_size.x - 1.0, 1.0 - 2.0 * px.y / view.camera_size.y);
    var z = 0.0;
    if view.depth_near > 0.0 {
        // Q-87: reverse Z with an infinite far plane, z_ndc = near / z_c.
        z = min(view.depth_near / depth, 1.0);
    }
    var out: VertexOut;
    out.position = vec4<f32>(ndc, z, 1.0);
    out.mean = mean;
    out.color = color;
    out.disc = vec2<f32>(k.pixel_radius, k.pixel_center_offset);
    return out;
}

@fragment
fn fs_center(in: VertexOut) -> @location(0) vec4<f32> {
    // The splat draw's sample point (`splat.wgsl`): the target pixel's (i, j) plus the
    // model's pixel center offset, in the camera's pixels.
    let scale = view.camera_size / view.target_size;
    let p = (in.position.xy - vec2<f32>(0.5) + vec2<f32>(in.disc.y)) * scale;
    let d = p - in.mean;
    if dot(d, d) > in.disc.x * in.disc.x {
        discard;
    }
    return vec4<f32>(in.color, 1.0);
}
