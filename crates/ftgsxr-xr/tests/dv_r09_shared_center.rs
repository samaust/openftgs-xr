// SPDX-License-Identifier: Apache-2.0

//! R-9 on the Quest's GPU (§3.2 R-9, D-4, `dv` lane, Q-122, Q-414, Q-T728.2): the driver
//! pushes P1–P10 and their `R9_<P>` requests (Q-402, Q-415) into `dv/`; the device renders
//! each request through `cmd_render_stereo` per view and shared-center in `conformance`,
//! and shared-center once more with `half_records` (§1.12), and writes
//! `dv/R9_<P>.<mode>.renders.safetensors`. The driver pulls them and judges with E15's
//! `metrics`: the per-view renders pass the tier against the committed `R9_<P>` references,
//! and the PSNR between the modes, the minimum over items with the per-view render as the
//! reference, is ≥ 40 dB (calibrated), with `half_records` too (between the modes only,
//! since the per-view render ignores the flag, Q-109). Every minimum is printed, so a miss
//! comes back with its figures. The bundled validation layer is on and must stay silent
//! (§1.13, Q-123). Prerequisites: `dv_support`'s.

use crate::dv_support;

use std::path::Path;
use std::time::Duration;

use dv_support::{
    DebugLayers, Validation, apk, clear, device, install, launch, pull, push, stop, verdict,
};
use ftgsxr_core::{RenderImage, read_renders, read_request};
use ftgsxr_testkit::{fixtures_dir, metrics, reference_render};

/// E15's tier (§3.2): PSNR ≥ 45 dB, s ≤ 0.5 %, s_α ≤ 0.5 % per item.
const MIN_PSNR: f64 = 45.0;
const MAX_SHARE: f64 = 0.005;
/// R-9's calibrated bound on the PSNR between the two modes.
const MIN_MODE_PSNR: f64 = 40.0;

const PROBES: [&str; 10] = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"];

/// The device's renders of `R9_<p>` in `mode`, pulled through `tmp` and read against the
/// request's SHA-256.
fn pulled(tmp: &Path, p: &str, mode: &str, sha256: [u8; 32]) -> Vec<(String, RenderImage)> {
    let name = format!("R9_{p}.{mode}.renders.safetensors");
    let local = tmp.join(&name);
    pull(&name, &local);
    let bytes = std::fs::read(&local).expect("the pulled renders");
    read_renders(&bytes, Some(sha256)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// The minimum PSNR of `other` against `per_view` over the items, with the item.
fn mode_psnr(per_view: &[(String, RenderImage)], other: &[(String, RenderImage)]) -> (f64, String) {
    let mut min = (f64::INFINITY, String::new());
    for (id, a) in per_view {
        let (_, b) = other
            .iter()
            .find(|(oid, _)| oid == id)
            .unwrap_or_else(|| panic!("no render for {id}"));
        let psnr = metrics(a, b).psnr;
        // A NaN is the minimum, so it fails.
        if psnr.is_nan() || psnr < min.0 {
            min = (psnr, id.clone());
        }
    }
    min
}

#[test]
#[ignore = "dv"]
fn dv_r09_shared_center_stereo_on_the_quest() {
    let apk = apk();
    device();
    install(&apk);
    stop();
    clear();
    let probes = fixtures_dir().join("probes");
    for p in PROBES {
        push(
            &probes.join(format!("{p}.safetensors")),
            &format!("{p}.safetensors"),
        );
        push(
            &probes.join(format!("R9_{p}.request.json")),
            &format!("R9_{p}.request.json"),
        );
    }
    // §1.13: the debug build runs with the validation layer (Q-123).
    let layers = DebugLayers::enable();
    let start = launch("r9");
    // Three runs of each probe, each pair rendered by its own renderer: about four
    // times Dv-4's renders.
    let result = verdict("r9", start, Duration::from_secs(1200));
    stop();
    let validation = Validation::read();
    drop(layers);
    let (pass, detail, took) = result.expect("an r9 verdict line in logcat");
    validation.assert_clean();
    assert!(pass, "r9 FAIL {detail}");
    eprintln!("r9 PASS {detail} in {:.0} s", took.as_secs_f64());

    let tmp = std::env::temp_dir().join(format!("ftgsxr-dv-r9-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("a scratch directory");
    let mut failures = Vec::new();
    for p in PROBES {
        let path = probes.join(format!("R9_{p}.request.json"));
        let request = read_request(&std::fs::read(&path).expect("the request"))
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let per_view = pulled(&tmp, p, "per-view", request.sha256);
        let shared = pulled(&tmp, p, "shared", request.sha256);
        let half = pulled(&tmp, p, "shared-half", request.sha256);
        let references = reference_render(&format!("R9_{p}"));
        assert_eq!(
            per_view.len(),
            references.len(),
            "R9_{p}: one render per item"
        );
        // The per-view renders against openftgs's references of the same eyes (Q-402).
        let mut min_psnr = f64::INFINITY;
        for (id, reference) in &references {
            let (_, image) = per_view
                .iter()
                .find(|(oid, _)| oid == id)
                .unwrap_or_else(|| panic!("R9_{p}: no device render for {id}"));
            let m = metrics(reference, image);
            min_psnr = min_psnr.min(m.psnr);
            // A NaN PSNR fails.
            if m.psnr.is_nan() || m.psnr < MIN_PSNR || m.share > MAX_SHARE || m.alpha > MAX_SHARE {
                failures.push(format!(
                    "R9_{p} {id} per view: PSNR {:.2} dB, s {:.4} %, s_α {:.4} %",
                    m.psnr,
                    100.0 * m.share,
                    100.0 * m.alpha
                ));
            }
        }
        eprintln!("R9_{p} per view against the references: minimum PSNR {min_psnr:.2} dB");
        for (label, other) in [("shared center", &shared), ("half records", &half)] {
            let (psnr, id) = mode_psnr(&per_view, other);
            eprintln!("R9_{p} per view against {label}: minimum PSNR {psnr:.2} dB ({id})");
            if psnr.is_nan() || psnr < MIN_MODE_PSNR {
                failures.push(format!(
                    "R9_{p} per view against {label}: {psnr:.2} dB ({id}), under {MIN_MODE_PSNR} dB"
                ));
            }
        }
    }
    let _ = std::fs::remove_dir_all(&tmp);
    assert!(
        failures.is_empty(),
        "R-9 on the Quest:\n{}",
        failures.join("\n")
    );
}
