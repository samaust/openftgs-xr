// SPDX-License-Identifier: Apache-2.0

//! Shared-center records are built (§1.4 stage 2 and Stereo, D-4; #112): with two eyes a
//! primitive is kept when E6 keeps it in either eye, under the 64-byte records and the
//! 48-byte half-Σ^S records alike. Their contents are measured from M-8 (R-9); here the
//! count against the oracle and a clean validation run.

mod cull_support;

use cull_support::{draw, look_at, oracle, passes, prepare, test_pose, upload, view};
use ftgsxr_core::{UVec2, Vec3};
use ftgsxr_render::{FrameBuffers, QualityTier, RenderMode, SortMode, SortPose};
use ftgsxr_testkit::{gpu_context, random_model};

#[test]
#[ignore = "gpu"]
fn gpu_shared_center_records_keep_rows_visible_in_either_eye() {
    let dev = gpu_context();
    let model = random_model(0x4_04, 20_000, 2);
    let pose = test_pose();
    let center = pose.translation;
    let head = center + Vec3::new(0.0, 0.2, -1.0);
    // Two eyes 64 mm apart, each with a narrow, offset image so the two frusta differ.
    let size = UVec2::new(160, 120);
    let left = look_at(head - Vec3::X * 0.032, center, 140.0, size, [100.0, 60.0]);
    let right = look_at(head + Vec3::X * 0.032, center, 140.0, size, [60.0, 60.0]);
    let tau = 0.9;
    let mut either: Vec<usize> = oracle(&model, &pose, tau, &left)
        .into_iter()
        .chain(oracle(&model, &pose, tau, &right))
        .map(|(i, _)| i)
        .collect();
    either.sort_unstable();
    either.dedup();
    assert!(either.len() > 100);
    for half_records in [false, true] {
        let tier = QualityTier {
            sort_mode: SortMode::SharedCenter,
            half_records,
            ..QualityTier::conformance()
        };
        let gpu_model = upload(&dev, &model, &tier);
        let passes = passes(&dev, &tier);
        let frame = FrameBuffers::new(&dev, &tier, model.len() as u32);
        let sort = SortPose {
            origin: head,
            forward: (center - head).normalize(),
        };
        let draws = [draw(&gpu_model, pose, tau, RenderMode::Splats)];
        let got = prepare(
            &dev,
            &passes,
            &frame,
            &draws,
            &sort,
            &[view(left), view(right)],
        );
        assert_eq!(
            got.visible_splats as usize,
            either.len(),
            "half Σ^S: {half_records}"
        );
        assert!(!got.overflow);
        assert!(
            got.records.is_empty(),
            "shared-center records are not decoded before M-8"
        );
        assert_eq!(got.sorted.len(), either.len());
    }
    assert!(dev.take_validation_messages().is_empty());
}
