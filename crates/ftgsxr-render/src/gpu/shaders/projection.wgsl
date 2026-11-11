// SPDX-License-Identifier: Apache-2.0

// projection.wgsl: E6's projection of one stage-space primitive for one view, prepended to
// `cull.wgsl` (stage 2, per view and the shared-center cull) and to `splat.wgsl` (stage 5's
// shared-center vertex stage, which projects each record for its eye, Q-400), so both run
// the same arithmetic in the same order (`passes.rs`'s `cull_source` and `splat_source`).
// E6 in the order of `ftgsxr_core::project`: camera space, the near and far culls, the
// Jacobian with its clamp, Σ′ + εI, the determinant guard, the opacity-aware extent and its
// cap, whole-pixel radii, the box cull, the conic as (a, b, c) (Q-21), and the non-finite
// cull, plus the float32 bound on μ′ (Q-118).

// The largest finite float32.
const F32_MAX: f32 = 3.40282347e38;

fn finite(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}

// A view's fields by name: R_w, t_w, W, f, c, H (E6).
struct View {
    rotation: mat3x3<f32>,
    translation: vec3<f32>,
    width: f32,
    focal: vec2<f32>,
    principal: vec2<f32>,
    height: f32,
}

// The model's E6 constants (D-39).
struct E6Constants {
    near: f32,
    far: f32,
    eps2d: f32,
    jacobian_clamp_margin: f32,
    screen_extent_max: f32,
    alpha_skip: f32,
}

struct Proj {
    visible: bool,
    // Culled for a non-finite value in the projection (E6).
    non_finite: bool,
    mean: vec2<f32>,
    conic: vec3<f32>,
    radii: vec2<f32>,
    depth: f32,
}

// The bound L on |u| beyond which f·u + c leaves float32's range, for f and L taken as
// at least 1 (Q-118).
fn mean_bound(f: f32, c: f32) -> f32 {
    return max((F32_MAX - abs(c)) / max(f, 1.0), 1.0);
}

// E6 for view `v` with the model's constants `k`.
fn project_view(v: View, k: E6Constants, mu_s: vec3<f32>, cov_s: mat3x3<f32>, sigma_eff: f32) -> Proj {
    var p: Proj;
    p.visible = false;
    p.non_finite = false;
    let xc = v.rotation * mu_s + v.translation;
    let z = xc.z;
    if !finite(z) {
        p.non_finite = true;
        return p;
    }
    if z < k.near || z > k.far {
        return p;
    }
    let fx = v.focal.x;
    let fy = v.focal.y;
    let cx = v.principal.x;
    let cy = v.principal.y;
    // Q-118: a μ′ beyond float32's range is non-finite, decided on x_c before μ′ is
    // computed, since a float32 overflow on the GPU need not give ∞: |f·x/z + c| fits
    // when |x| / L ≤ z with L = (F32_MAX − |c|) / f. f is taken as at least 1, where u
    // itself is the bound, and L as at least 1, so neither quotient overflows.
    if abs(xc.x) / mean_bound(fx, cx) > z || abs(xc.y) / mean_bound(fy, cy) > z {
        p.non_finite = true;
        return p;
    }
    let u = xc.x / z;
    let w = xc.y / z;
    let mean = vec2<f32>(fx * u + cx, fy * w + cy);
    // The Jacobian at x_c, its ratios clamped m·W/f_x (m·H/f_y) beyond the image; μ′ is not.
    let mx = k.jacobian_clamp_margin * v.width / fx;
    let my = k.jacobian_clamp_margin * v.height / fy;
    let tx = clamp(u, -(cx / fx + mx), (v.width - cx) / fx + mx);
    let ty = clamp(w, -(cy / fy + my), (v.height - cy) / fy + my);
    let j0 = vec3<f32>(fx / z, 0.0, -fx * tx / z);
    let j1 = vec3<f32>(0.0, fy / z, -fy * ty / z);
    let sc = v.rotation * cov_s * transpose(v.rotation);
    let s00 = dot(j0, sc * j0) + k.eps2d;
    let s01 = dot(j0, sc * j1);
    let s11 = dot(j1, sc * j1) + k.eps2d;
    let det = s00 * s11 - s01 * s01;
    if !finite(det) {
        p.non_finite = true;
        return p;
    }
    if det <= 0.0 {
        return p;
    }
    // The opacity-aware extent, capped; σ_eff < α_skip has none.
    let e2 = 2.0 * log(sigma_eff / k.alpha_skip);
    if !finite(e2) {
        p.non_finite = true;
        return p;
    }
    if e2 < 0.0 {
        return p;
    }
    let extent = min(sqrt(e2), k.screen_extent_max);
    let rx = ceil(extent * sqrt(s00));
    let ry = ceil(extent * sqrt(s11));
    // Q-118: the mean's finiteness before the box test, so an infinite μ′ is counted
    // rather than culled as off the image.
    if !finite(rx) || !finite(ry) || !finite(mean.x) || !finite(mean.y) {
        p.non_finite = true;
        return p;
    }
    if rx <= 0.0 || ry <= 0.0 {
        return p;
    }
    if mean.x + rx <= 0.0 || mean.x - rx >= v.width || mean.y + ry <= 0.0 || mean.y - ry >= v.height {
        return p;
    }
    let conic = vec3<f32>(s11 / det, -s01 / det, s00 / det);
    if !finite(conic.x) || !finite(conic.y) || !finite(conic.z) {
        p.non_finite = true;
        return p;
    }
    p.visible = true;
    p.mean = mean;
    p.conic = conic;
    p.radii = vec2<f32>(rx, ry);
    p.depth = z;
    return p;
}
