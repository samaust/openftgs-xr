// SPDX-License-Identifier: Apache-2.0

//! R-7, several models (§3.2, §1.4 Several models, D-5): P7's two primitives as two
//! instances, each at its pose (E5), overlapping and swapping depth over τ, drawn from one
//! record buffer under one count and one sort, against openftgs's reference render of the
//! pre-posed merged model at eight times (Q-12: `merged/R7_instance_a`, `R7_instance_b`,
//! their poses in `merged/R7_poses.json`, the reference `probes/R7_merged`). E15's tier.

mod probe_support;

use ftgsxr_render::{HeadlessRenderer, QualityTier};
use ftgsxr_testkit::layer_messages;
use probe_support::{check_items, posed_instances, probe, render_request};

#[test]
#[ignore = "gpu"]
fn gpu_r07_two_instances_against_the_merged_model() {
    let _layer = layer_messages(); // R-16, Q-T121.1
    let instances = posed_instances("R7_poses.json");
    assert_eq!(instances.len(), 2, "P7's two rows as two instances");
    let (merged, request) = probe("R7_merged");
    assert_eq!(merged.len(), 2);
    assert_eq!(request.items.len(), 8, "eight times");
    let mut renderer = HeadlessRenderer::new(QualityTier::conformance()).expect("an adapter");
    let posed: Vec<_> = instances
        .iter()
        .map(|(model, pose)| {
            let id = renderer
                .load(model, request.sh_degree)
                .expect("the instance loads");
            (id, *pose)
        })
        .collect();
    let items = render_request(&mut renderer, &posed, "R7_merged");
    check_items("R7_merged", &items, false);
}
