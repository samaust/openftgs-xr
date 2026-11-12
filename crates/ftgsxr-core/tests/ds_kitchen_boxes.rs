// SPDX-License-Identifier: Apache-2.0

//! E10's two boxes on a trained room-scale model (`ds` lane, #622): on `kitchen`'s
//! exchange file, B_m (`Model::bounds`) and the content box B_c (`Model::content_bounds`)
//! have the sizes #622's triage measured, within 1 % (the agent brief's criterion, from
//! a script kept outside the repository). B_m is the brief's figure; B_c is this
//! implementation's E10 on that file (5.246 × 2.763 × 2.344), since the brief's
//! 5.25 × 2.82 × 2.38 could not be reproduced under any reading of the midpoint (owner
//! decision, 2026-11-07, #622). The figures are those of the file
//! the triage measured, openftgs-m17's `kitchen/default-395` (N = 3,000,000, τ in
//! [0, 9.97] s); a `kitchen` trained again gives other figures. The test reads it at
//! `$OPENFTGS_DATASETS/xr-fixtures/kitchen.safetensors`, #622's openftgs-m17 file (M-8's
//! budget model is `kitchen_1m.safetensors` beside it, Q-410), and fails with the reason
//! when it is absent (§2.7 Lanes).

use std::path::PathBuf;

use ftgsxr_core::{Vec3, open_model};

/// `$OPENFTGS_DATASETS/xr-fixtures/kitchen.safetensors`.
///
/// # Panics
///
/// When the variable is unset or the file is absent, with the reason (§2.7 Lanes).
fn kitchen() -> PathBuf {
    let root = std::env::var_os("OPENFTGS_DATASETS").unwrap_or_else(|| {
        panic!(
            "#622: $OPENFTGS_DATASETS is unset; the ds lane reads xr-fixtures/kitchen.safetensors"
        )
    });
    let path = PathBuf::from(root).join("xr-fixtures/kitchen.safetensors");
    assert!(
        path.is_file(),
        "#622: {} is absent: the measured model is openftgs-m17's kitchen/default-395",
        path.display()
    );
    path
}

fn assert_size(what: &str, got: Vec3, want: [f32; 3]) {
    for a in 0..3 {
        let rel = (got[a] - want[a]).abs() / want[a];
        assert!(rel <= 0.01, "{what}: size {got} vs {want:?} (axis {a})");
    }
}

#[test]
#[ignore = "ds"]
fn ds_kitchen_boxes_have_the_measured_sizes() {
    let _layer = ftgsxr_testkit::layer_messages(); // R-16, Q-105
    let path = kitchen();
    let model = open_model(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let whole = model.bounds();
    let content = model.content_bounds();
    eprintln!("kitchen: B_m {whole:?}, B_c {content:?}");
    assert_size("B_m", whole.size(), [24.2, 30.2, 23.0]);
    assert_size("B_c", content.size(), [5.25, 2.76, 2.34]);
}
