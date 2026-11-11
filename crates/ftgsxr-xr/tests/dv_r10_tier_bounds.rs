// SPDX-License-Identifier: Apache-2.0

//! R-10 on the Quest's GPU (§3.2 R-10, D-12, `dv` lane, Q-122, Q-70, Q-T728.2): the
//! driver writes the desktop R-10's degree-2 random model (`random_model(0x10_10, 50_000,
//! 2)`) as an exchange file with `write_exchange` (Q-65) and pushes it with R-2's
//! random-model request, and `R10_deg2`'s model and request (Q-70). The device renders the
//! random model through `cmd_render` in `conformance` and in `quest-balanced`, and
//! `R10_deg2` in `conformance` at its request's degree 2. The driver judges: the
//! `quest-balanced` storage against `conformance` with PSNR ≥ 50 dB on every item (finite,
//! so the half path ran; the model in view), and `R10_deg2` E15's tier against openftgs's
//! degree-2 reference. Every minimum is printed. The bundled validation layer is on and
//! must stay silent (§1.13, Q-123). Prerequisites: `dv_support`'s.

use crate::dv_support;

use std::path::Path;
use std::time::Duration;

use dv_support::{
    DebugLayers, Validation, apk, clear, device, install, launch, pull, push, stop, verdict,
};
use ftgsxr_core::{RenderImage, read_renders, read_request, write_exchange};
use ftgsxr_testkit::{fixtures_dir, metrics, random_model, reference_render};

/// R-10's bound between the two storages.
const MIN_STORAGE_PSNR: f64 = 50.0;
/// E15's tier (§3.2): PSNR ≥ 45 dB, s ≤ 0.5 %, s_α ≤ 0.5 % per item.
const MIN_PSNR: f64 = 45.0;
const MAX_SHARE: f64 = 0.005;

/// The device's file `name`, pulled through `tmp` and read against `sha256`.
fn pulled(tmp: &Path, name: &str, sha256: [u8; 32]) -> Vec<(String, RenderImage)> {
    let local = tmp.join(name);
    pull(name, &local);
    let bytes = std::fs::read(&local).expect("the pulled renders");
    read_renders(&bytes, Some(sha256)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

#[test]
#[ignore = "dv"]
fn dv_r10_tier_bounds_on_the_quest() {
    let apk = apk();
    let tmp = std::env::temp_dir().join(format!("ftgsxr-dv-r10-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("a scratch directory");
    let random = tmp.join("r10-random.safetensors");
    // The desktop R-10's model (`ftgsxr-render`'s `r10_tier_bounds.rs`).
    write_exchange(&random, &random_model(0x10_10, 50_000, 2)).expect("the model writes");
    let probes = fixtures_dir().join("probes");
    let random_request = probes.join("R2_random.request.json");
    let deg2_request = probes.join("R10_deg2.request.json");

    device();
    install(&apk);
    stop();
    clear();
    push(&random, "r10-random.safetensors");
    push(&random_request, "r10-random.request.json");
    push(&probes.join("R10_deg2.safetensors"), "R10_deg2.safetensors");
    push(&deg2_request, "R10_deg2.request.json");
    // §1.13: the debug build runs with the validation layer (Q-123).
    let layers = DebugLayers::enable();
    let start = launch("r10");
    let result = verdict("r10", start, Duration::from_secs(300));
    stop();
    let validation = Validation::read();
    drop(layers);
    let (pass, detail, _) = result.expect("an r10 verdict line in logcat");
    validation.assert_clean();
    assert!(pass, "r10 FAIL {detail}");

    let request = |path: &Path| {
        read_request(&std::fs::read(path).expect("the request"))
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    };
    let random_req = request(&random_request);
    let full = pulled(
        &tmp,
        "r10-random.conformance.renders.safetensors",
        random_req.sha256,
    );
    let half = pulled(
        &tmp,
        "r10-random.quest-balanced.renders.safetensors",
        random_req.sha256,
    );
    let deg2_req = request(&deg2_request);
    let deg2 = pulled(&tmp, "R10_deg2.renders.safetensors", deg2_req.sha256);
    let _ = std::fs::remove_dir_all(&tmp);

    let mut failures = Vec::new();
    assert_eq!(full.len(), random_req.items.len(), "one render per item");
    let mut min_psnr = f64::INFINITY;
    for (id, a) in &full {
        let (_, b) = half
            .iter()
            .find(|(oid, _)| oid == id)
            .unwrap_or_else(|| panic!("no quest-balanced render for {id}"));
        assert!(
            a.alpha.iter().any(|&v| v > 0.1),
            "{id}: the model is in view"
        );
        let psnr = metrics(a, b).psnr;
        min_psnr = min_psnr.min(psnr);
        // The tier stores in half precision (§1.12): an infinite PSNR means the half path
        // did not run.
        if !psnr.is_finite() || psnr < MIN_STORAGE_PSNR {
            failures.push(format!("r10-random {id}: PSNR {psnr:.2} dB"));
        }
    }
    eprintln!("R-10 quest-balanced storage on the Quest: minimum PSNR {min_psnr:.2} dB");

    let references = reference_render("R10_deg2");
    assert_eq!(
        deg2.len(),
        references.len(),
        "R10_deg2: one render per item"
    );
    let mut min_psnr = f64::INFINITY;
    for (id, reference) in &references {
        let (_, image) = deg2
            .iter()
            .find(|(oid, _)| oid == id)
            .unwrap_or_else(|| panic!("R10_deg2: no device render for {id}"));
        let m = metrics(reference, image);
        min_psnr = min_psnr.min(m.psnr);
        // A NaN PSNR fails.
        if m.psnr.is_nan() || m.psnr < MIN_PSNR || m.share > MAX_SHARE || m.alpha > MAX_SHARE {
            failures.push(format!(
                "R10_deg2 {id}: PSNR {:.2} dB, s {:.4} %, s_α {:.4} %",
                m.psnr,
                100.0 * m.share,
                100.0 * m.alpha
            ));
        }
    }
    eprintln!("R10_deg2 on the Quest: minimum PSNR {min_psnr:.2} dB");
    assert!(
        failures.is_empty(),
        "R-10 on the Quest:\n{}",
        failures.join("\n")
    );
}
