// SPDX-License-Identifier: Apache-2.0

//! R-2, the player's additional probes (§3.2, Q-10, Q-11): two overlapping degree-3
//! primitives; a model whose `render_conventions` background is white rendered over the
//! request's black; a single-timestamp model at τ ≠ τ₀; a model with τ₀ = 10⁹ s; a random
//! model at three times. Each item passes E15's tier against openftgs's reference render.

mod probe_support;

use ftgsxr_testkit::layer_messages;
use probe_support::check_probe;

#[test]
#[ignore = "gpu"]
fn gpu_r02_two_overlapping_degree_3_primitives() {
    let _layer = layer_messages(); // R-16, Q-T121.1
    check_probe("R2_overlap_deg3");
}

#[test]
#[ignore = "gpu"]
fn gpu_r02_request_background_not_the_models() {
    let _layer = layer_messages(); // R-16, Q-T121.1
    check_probe("R2_white_background");
}

#[test]
#[ignore = "gpu"]
fn gpu_r02_single_timestamp_at_other_times() {
    let _layer = layer_messages(); // R-16, Q-T121.1
    check_probe("R2_single_timestamp");
}

#[test]
#[ignore = "gpu"]
fn gpu_r02_epoch_time_origin() {
    let _layer = layer_messages(); // R-16, Q-T121.1
    check_probe("R2_epoch_time");
}

#[test]
#[ignore = "gpu"]
fn gpu_r02_random_model_at_three_times() {
    let _layer = layer_messages(); // R-16, Q-T121.1
    check_probe("R2_random");
}
