// SPDX-License-Identifier: Apache-2.0

//! Shared-center records are built (§1.4 stage 2 and Stereo, D-4; #112): with two eyes a
//! primitive is kept when E6 keeps it in either eye, under the 64-byte records and the
//! 48-byte half-Σ^S records alike: the count against the oracle and a clean validation
//! run. From M-8 (#721, Q-400) their contents too, read back as `FrameContents.shared`:
//! μ^S, Σ^S, the color at the midpoint and the key, against the float64 oracle.

use crate::cull_support;

use cull_support::{draw, look_at, oracle, passes, prepare, row_at, test_pose, upload, view};
use ftgsxr_core::{Model, Pose, UVec2, Vec3, moved_position, sh_color, sort_key};
use ftgsxr_render::{FrameBuffers, QualityTier, RenderMode, SharedRecordView, SortMode, SortPose};
use ftgsxr_testkit::{gpu_context, layer_messages, random_model};

#[test]
#[ignore = "gpu"]
fn gpu_shared_center_records_keep_rows_visible_in_either_eye() {
    let _layer = layer_messages(); // R-16, Q-105
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
            "no per-view records in shared-center mode"
        );
        assert_eq!(got.shared.len(), either.len(), "Q-400: one per slot");
        assert_eq!(got.sorted.len(), either.len());
    }
    assert!(dev.take_validation_messages().is_empty());
}

/// R-5's tolerances (§3.2): positions and depths 10⁻⁵, covariances 10⁻⁴ relative, with a
/// floor of 10⁻⁴ of the largest component as R-5's conics (Q-112, Q-T721.2).
const POSITION: f32 = 1e-5;
const DEPTH: f32 = 1e-5;
// Q-T721.2
const COV_REL: f32 = 1e-4;
/// D-12's (§1.12) half-precision bound, 2⁻¹¹ relative: Σ^S under `half_records`, added to
/// `COV_REL`, and the color and σ_eff, which every shared-center record holds as half
/// floats (§1.4 stage 2), with 10⁻⁵ for float32's evaluation (Q-T721.2).
// Q-T721.2
const HALF_REL: f32 = 1.0 / 2048.0;
// Q-T721.2
const COLOR_ABS: f32 = 1e-5;

/// One visible row's oracle values (float64, rounded once): μ^S and Σ^S (E1, E5), the
/// color at the sort pose's origin (E4, D-4) and σ_eff (E2), and its forward depth along
/// the sort pose's axis (E8, D-4).
struct Expected {
    mean: Vec3,
    cov: [f32; 6],
    color: [f32; 4],
    depth: f32,
}

fn expected(model: &Model, row: usize, pose: &Pose, tau: f64, sort: &SortPose) -> Expected {
    let (sigma_eff, mean, cov) = row_at(model, row, pose, tau);
    let mu = moved_position(
        Vec3::from(model.position[row]),
        Vec3::from(model.velocity[row]),
        model.time_center[row],
        tau,
    )
    .as_dvec3();
    // E4's eye in model space: the pose's inverse at the midpoint (D-4).
    let r = pose.rotation.as_dquat();
    let eye = r.inverse() * (sort.origin.as_dvec3() - pose.translation.as_dvec3())
        / f64::from(pose.scale);
    let k = (usize::from(model.sh_degree) + 1).pow(2) * 3;
    let c = sh_color(
        model.sh_degree,
        &model.sh[row * k..(row + 1) * k],
        (mu - eye).as_vec3(),
    );
    let depth = (mean.as_dvec3() - sort.origin.as_dvec3()).dot(sort.forward.as_dvec3()) as f32;
    Expected {
        mean,
        cov: [
            cov.x_axis.x,
            cov.y_axis.x,
            cov.z_axis.x,
            cov.y_axis.y,
            cov.z_axis.y,
            cov.z_axis.z,
        ],
        color: [c[0], c[1], c[2], sigma_eff],
        depth,
    }
}

/// The record whose μ^S lies within `POSITION` of `want` per axis, among those not yet
/// taken: records carry no row, and the slots' order is the atomics'.
fn find(records: &[SharedRecordView], used: &mut [bool], want: Vec3) -> Option<usize> {
    let k = (0..records.len())
        .find(|&k| !used[k] && (records[k].mean_stage - want).abs().max_element() <= POSITION)?;
    used[k] = true;
    Some(k)
}

/// Q-400 (#721): `FrameContents.shared` for a two-eye frame against the CPU oracle, in both
/// shared-center record layouts. Each row visible in either eye has one record, whose μ^S
/// is the oracle's (E1, E5) within 10⁻⁵, whose Σ^S is within 10⁻⁴ relative (2⁻¹¹ more
/// under `half_records`, D-12), whose color and σ_eff are the midpoint's (E4, E2) within
/// half precision's 2⁻¹¹ relative plus float32's 10⁻⁵, and whose key is `sort_key` (Q-2)
/// of a depth within 10⁻⁵ of the oracle's forward depth at the sort pose. In per-view
/// mode `shared` is empty.
#[test]
#[ignore = "gpu"]
fn gpu_frame_contents_shared_holds_the_shared_records_against_the_oracle() {
    let _layer = layer_messages(); // R-16, Q-105
    let dev = gpu_context();
    let model = random_model(0x7_21, 4_000, 2);
    let pose = test_pose();
    let center = pose.translation;
    let head = center + Vec3::new(0.0, 0.2, -1.0);
    let size = UVec2::new(160, 120);
    let left = look_at(head - Vec3::X * 0.032, center, 140.0, size, [100.0, 60.0]);
    let right = look_at(head + Vec3::X * 0.032, center, 140.0, size, [60.0, 60.0]);
    let tau = 0.9;
    let sort = SortPose {
        origin: head,
        forward: (center - head).normalize(),
    };
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
        let frame = FrameBuffers::new(&dev, &tier, model.len() as u32);
        let got = prepare(
            &dev,
            &passes(&dev, &tier),
            &frame,
            &[draw(&gpu_model, pose, tau, RenderMode::Splats)],
            &sort,
            &[view(left), view(right)],
        );
        assert_eq!(got.shared.len(), either.len(), "half Σ^S: {half_records}");
        let mut used = vec![false; got.shared.len()];
        for &row in &either {
            let want = expected(&model, row, &pose, tau, &sort);
            let k = find(&got.shared, &mut used, want.mean)
                .unwrap_or_else(|| panic!("row {row}: no record at μ^S {}", want.mean));
            let r = got.shared[k];
            let largest = want.cov.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let cov_rel = if half_records {
                COV_REL + HALF_REL
            } else {
                COV_REL
            };
            for (i, (&g, &w)) in r.cov_stage.iter().zip(&want.cov).enumerate() {
                assert!(
                    (g - w).abs() <= cov_rel * w.abs().max(largest),
                    "row {row} Σ^S[{i}]: {g} against {w} (half Σ^S: {half_records})"
                );
            }
            for (i, (&g, &w)) in r.color.iter().zip(&want.color).enumerate() {
                assert!(
                    (g - w).abs() <= HALF_REL * w.abs() + COLOR_ABS,
                    "row {row} color[{i}]: {g} against {w}"
                );
            }
            let depth = f32::from_bits(u32::MAX - r.key);
            assert_eq!(sort_key(depth), Some(r.key), "row {row}: a key of a depth");
            assert!(
                (depth - want.depth).abs() <= DEPTH,
                "row {row}: the key's depth {depth} against {}",
                want.depth
            );
        }
        assert!(got.records.is_empty());
    }

    // Per view: `records` filled, `shared` empty (Q-400).
    let tier = QualityTier::conformance();
    let gpu_model = upload(&dev, &model, &tier);
    let frame = FrameBuffers::new(&dev, &tier, model.len() as u32);
    let got = prepare(
        &dev,
        &passes(&dev, &tier),
        &frame,
        &[draw(&gpu_model, pose, tau, RenderMode::Splats)],
        &sort,
        &[view(left)],
    );
    assert!(!got.records.is_empty());
    assert!(got.shared.is_empty(), "Q-400: no shared records per view");
    assert!(dev.take_validation_messages().is_empty());
}
