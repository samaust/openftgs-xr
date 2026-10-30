// SPDX-License-Identifier: Apache-2.0

//! R-8, the pose (§3.2, E5, E4's eye o_{v,m} = P_m⁻¹(o_v)): one instance of the degree-3
//! probe `R2_overlap_deg3` rotated 90° about y and scaled 2, against openftgs's reference
//! render of the merged pre-posed model, whose SH coefficients are rotated with it (Q-12:
//! the pose in `merged/R8_poses.json`, the reference `probes/R8_merged`, its 32-item ring).
//! E15's tier.

mod probe_support;

use ftgsxr_render::{HeadlessRenderer, QualityTier};
use probe_support::{check_items, posed_instances, probe, render_request};

#[test]
#[ignore = "gpu"]
fn gpu_r08_rotated_and_scaled_instance_with_degree_3_sh() {
    let instances = posed_instances("R8_poses.json");
    assert_eq!(instances.len(), 1);
    let (model, pose) = &instances[0];
    assert_eq!(model.sh_degree, 3, "degree-3 SH");
    assert_eq!(pose.scale, 2.0);
    let (_, request) = probe("R8_merged");
    let mut renderer = HeadlessRenderer::new(QualityTier::conformance()).expect("an adapter");
    let id = renderer
        .load(model, request.sh_degree)
        .expect("the instance loads");
    let items = render_request(&mut renderer, &[(id, *pose)], "R8_merged");
    check_items("R8_merged", &items, false);
}
