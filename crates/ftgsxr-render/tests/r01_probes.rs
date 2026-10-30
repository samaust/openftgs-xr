// SPDX-License-Identifier: Apache-2.0

//! R-1, E6 and E7 on the probes P1–P10 (§3.2): every item of each probe, rendered by the
//! headless renderer in the `conformance` tier, passes E15's tier against openftgs's
//! reference render (PSNR ≥ 45 dB, s ≤ 0.5 %, s_α ≤ 0.5 %); the per-probe minimum PSNR is
//! reported on standard error.

mod probe_support;

use ftgsxr_testkit::layer_messages;
use probe_support::check_probe;

#[test]
#[ignore = "gpu"]
fn gpu_r01_probe_p1_static_isotropic_subpixel_shifts() {
    let _layer = layer_messages(); // R-16, Q-T121.1
    check_probe("P1");
}

#[test]
#[ignore = "gpu"]
fn gpu_r01_probe_p2_moving_primitive_over_tau() {
    let _layer = layer_messages(); // R-16, Q-T121.1
    check_probe("P2");
}

#[test]
#[ignore = "gpu"]
fn gpu_r01_probe_p3_short_duration_over_tau() {
    let _layer = layer_messages(); // R-16, Q-T121.1
    check_probe("P3");
}

#[test]
#[ignore = "gpu"]
fn gpu_r01_probe_p4_near_opaque_swept_to_alpha_max() {
    let _layer = layer_messages(); // R-16, Q-T121.1
    check_probe("P4");
}

#[test]
#[ignore = "gpu"]
fn gpu_r01_probe_p5_degree_3_sh_from_a_ring() {
    let _layer = layer_messages(); // R-16, Q-T121.1
    check_probe("P5");
}

#[test]
#[ignore = "gpu"]
fn gpu_r01_probe_p6_crossing_the_near_plane() {
    let _layer = layer_messages(); // R-16, Q-T121.1
    check_probe("P6");
}

#[test]
#[ignore = "gpu"]
fn gpu_r01_probe_p7_two_primitives_swapping_depth() {
    let _layer = layer_messages(); // R-16, Q-T121.1
    check_probe("P7");
}

#[test]
#[ignore = "gpu"]
fn gpu_r01_probe_p8_effective_opacity_around_the_skip() {
    let _layer = layer_messages(); // R-16, Q-T121.1
    check_probe("P8");
}

#[test]
#[ignore = "gpu"]
fn gpu_r01_probe_p9_swept_off_screen() {
    let _layer = layer_messages(); // R-16, Q-T121.1
    check_probe("P9");
}

#[test]
#[ignore = "gpu"]
fn gpu_r01_probe_p10_times_outside_the_span() {
    let _layer = layer_messages(); // R-16, Q-T121.1
    check_probe("P10");
}
