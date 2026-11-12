// SPDX-License-Identifier: Apache-2.0

//! The budget model (`ds` lane, #726, Q-410): `kitchen_1m.safetensors`, openftgs-synth's
//! `kitchen` trained by `tools/fixtures/train_tiny.sh kitchen_1m` into
//! `$OPENFTGS_DATASETS/xr-fixtures/` (§2.7 Fixtures, D-29), opens through `open_model` as
//! the model P-1 names, "1 M primitives at degree 2" (§1.12): SH degree 2 and at most
//! 1,000,000 primitives, the run's `growth.n_max`. Without the variable or the file the
//! test fails with the reason (§2.7 Lanes). #622's `kitchen.safetensors` is another file
//! (`ds_kitchen_boxes`).

use std::path::PathBuf;

use ftgsxr_core::open_model;

/// P-1's count, the budget model's `growth.n_max` (Q-410).
const CAP: usize = 1_000_000;

/// `$OPENFTGS_DATASETS/xr-fixtures/kitchen_1m.safetensors`.
///
/// # Panics
///
/// When the variable is unset or the file is absent, with the reason (§2.7 Lanes).
fn kitchen_1m() -> PathBuf {
    let root = std::env::var_os("OPENFTGS_DATASETS").unwrap_or_else(|| {
        panic!(
            "#726: $OPENFTGS_DATASETS is unset; the ds lane reads \
             xr-fixtures/kitchen_1m.safetensors"
        )
    });
    let path = PathBuf::from(root).join("xr-fixtures/kitchen_1m.safetensors");
    assert!(
        path.is_file(),
        "#726: {} is absent; run tools/fixtures/train_tiny.sh kitchen_1m to write it",
        path.display()
    );
    path
}

#[test]
#[ignore = "ds"]
fn ds_kitchen_1m_opens_at_degree_2_within_the_cap() {
    let _layer = ftgsxr_testkit::layer_messages(); // R-16, Q-105
    let path = kitchen_1m();
    let model = open_model(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    eprintln!(
        "kitchen_1m: N = {}, degree {}",
        model.len(),
        model.sh_degree
    );
    assert_eq!(model.sh_degree, 2, "P-1: degree 2");
    assert!(model.len() <= CAP, "P-1: N = {} above {CAP}", model.len());
}
