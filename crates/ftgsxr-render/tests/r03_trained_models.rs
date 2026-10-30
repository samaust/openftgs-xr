// SPDX-License-Identifier: Apache-2.0

//! R-3, the trained models (§3.2, `ds` lane): openftgs's `tiny` and `tiny_unsync` models,
//! trained by `tools/fixtures/train_tiny.sh` into `$OPENFTGS_DATASETS/xr-fixtures/` (§2.7
//! Fixtures, D-29, Q-84), rendered at their request's 20 items (Q-83) by the headless
//! renderer in the `conformance` tier, each item passing E15's tier against openftgs's
//! reference render (PSNR ≥ 45 dB, s ≤ 0.5 %, s_α ≤ 0.5 %). Every item's figures go to
//! standard error: D-7 leaves the transmittance stop out on the ground that the affected
//! pixels are rare in trained models, which R-3 measures. In a debug build the renderer's
//! device fails the test on any GPU validation message (§1.13, R-16, Q-74).
//!
//! Without `$OPENFTGS_DATASETS`, or without its `xr-fixtures/` directory, the tests fail with
//! the reason: running the `ds` lane is the request for the dataset and the trained models
//! (§2.7 Lanes), so a missing one is a fault, not a pass. A directory that holds only some of
//! a model's three files fails the same way.

mod probe_support;

use std::path::PathBuf;

use ftgsxr_core::{open_model, read_renders, read_request};
use ftgsxr_testkit::layer_messages;
use probe_support::{check_items, render_items};

/// R-3's item count per model (§3.2).
const ITEMS: usize = 20;

/// `$OPENFTGS_DATASETS/xr-fixtures/`.
///
/// # Panics
///
/// When the variable is unset or the directory is absent, with the reason (§2.7 Lanes).
fn xr_fixtures() -> PathBuf {
    let root = std::env::var_os("OPENFTGS_DATASETS").unwrap_or_else(|| {
        panic!(
            "R-3: $OPENFTGS_DATASETS is unset; the ds lane reads the trained models that \
             tools/fixtures/train_tiny.sh writes into $OPENFTGS_DATASETS/xr-fixtures/"
        )
    });
    let dir = PathBuf::from(root).join("xr-fixtures");
    assert!(
        dir.is_dir(),
        "R-3: {} is absent; run tools/fixtures/train_tiny.sh to write it",
        dir.display()
    );
    dir
}

/// R-3 on the trained model `name`: `NAME.safetensors` read by `open_model`, its
/// `NAME.request.json` and the reference renders `NAME.renders.safetensors`, checked to be
/// made from that request (its SHA-256), every item through the tier.
fn check_trained(name: &str) {
    let dir = xr_fixtures();
    let file = |suffix: &str| dir.join(format!("{name}.{suffix}"));
    let model_path = file("safetensors");
    let model = open_model(&model_path).unwrap_or_else(|e| panic!("{}: {e}", model_path.display()));
    let request_path = file("request.json");
    let request_bytes =
        std::fs::read(&request_path).unwrap_or_else(|e| panic!("{}: {e}", request_path.display()));
    let request =
        read_request(&request_bytes).unwrap_or_else(|e| panic!("{}: {e}", request_path.display()));
    let renders_path = file("renders.safetensors");
    let renders_bytes =
        std::fs::read(&renders_path).unwrap_or_else(|e| panic!("{}: {e}", renders_path.display()));
    let references = read_renders(&renders_bytes, Some(request.sha256))
        .unwrap_or_else(|e| panic!("{}: {e}", renders_path.display()));
    // Q-83
    assert_eq!(
        request.items.len(),
        ITEMS,
        "{name}: R-3 renders {ITEMS} items per model"
    );
    let items = render_items(name, &model, &request, &references);
    check_items(name, &items, true);
}

#[test]
#[ignore = "ds"]
fn ds_r03_trained_tiny_passes_the_tier() {
    let _layer = layer_messages(); // R-16, Q-105
    check_trained("tiny");
}

#[test]
#[ignore = "ds"]
fn ds_r03_trained_tiny_unsync_passes_the_tier() {
    let _layer = layer_messages(); // R-16, Q-105
    check_trained("tiny_unsync");
}
