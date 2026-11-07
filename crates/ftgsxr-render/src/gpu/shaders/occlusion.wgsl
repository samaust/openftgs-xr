// SPDX-License-Identifier: Apache-2.0

// occlusion.wgsl: E13's per-fragment functions (D-45), prepended to `splat.wgsl` when the
// splat pipeline is built (`passes.rs`'s `splat_source`): the stored environment depth to
// metres and the occlusion weight.

// One environment depth sample in metres along the depth camera's z axis, or invalid.
struct EnvDepth {
    metres: f32,
    valid: bool,
}

// Q-340: the stored depth d as an OpenXR projection writes it (XR_KHR_composition_layer_depth's
// transform with minDepth 0 and maxDepth 1, zero at near and one at far), inverted:
// Z = f·n / (f − d·(f − n)); with an infinite far plane (farZ < nearZ, XR_META_environment_depth,
// or farZ = +∞) Z = n / (1 − d). A sample is valid when 0 ≤ d < 1: NaN fails the test, and
// d = 1 is the far plane, no surface in range.
fn depth_metres(d: f32, near: f32, far: f32) -> EnvDepth {
    var out: EnvDepth;
    out.valid = d >= 0.0 && d < 1.0;
    // +∞ fails `far <= 3.4e38`, so it takes the infinite form.
    if far < near || !(far <= 3.4e38) {
        out.metres = near / (1.0 - d);
    } else {
        out.metres = far * near / (far - d * (far - near));
    }
    if !(out.metres > 0.0) {
        out.valid = false;
    }
    return out;
}

// E13's weight (`ftgsxr_core::occlusion_weight`, Q-63): hard, 1 up to z_e + δ; soft,
// 1 − smoothstep(z_e − δ, z_e + δ, z_f); at δ ≤ 0, in both modes, the hard step at z_e;
// 1 without a valid sample.
fn occlusion_weight(z_f: f32, z_e: f32, valid: bool, delta_m: f32, soft: bool) -> f32 {
    if !valid {
        return 1.0;
    }
    let delta = max(delta_m, 0.0);
    if soft && delta > 0.0 {
        let t = clamp((z_f - (z_e - delta)) / (2.0 * delta), 0.0, 1.0);
        return 1.0 - t * t * (3.0 - 2.0 * t);
    }
    if z_f <= z_e + delta {
        return 1.0;
    }
    return 0.0;
}
