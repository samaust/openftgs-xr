// SPDX-License-Identifier: Apache-2.0

//! Dv-4 (§3.4, `dv` lane, Q-122): the probes P1–P10 rendered by the headless renderer on
//! the Quest's GPU meet E15's tier against the committed reference renders, as R-1 does
//! (§3.2): the driver pushes each probe's model and request into `dv/`, the device runs
//! `cmd_render` on each in the `conformance` tier and writes `dv/<P>.renders.safetensors`,
//! and the driver pulls them and applies `metrics` against `reference_render` (D-46).
//! The bundled validation layer is enabled for the run (§1.13, Q-123), as for Dv-2, and
//! the run fails on any of its messages. Prerequisites: `dv_support`'s.

mod dv_support;

use std::time::Duration;

use dv_support::{
    DebugLayers, Validation, apk, clear, device, install, launch, pull, push, stop, verdict,
};
use ftgsxr_core::read_renders;
use ftgsxr_testkit::{fixtures_dir, metrics, probe, reference_render};

/// E15's tier (§3.2): PSNR ≥ 45 dB, s ≤ 0.5 %, s_α ≤ 0.5 % per item.
const MIN_PSNR: f64 = 45.0;
const MAX_SHARE: f64 = 0.005;

const PROBES: [&str; 10] = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"];

#[test]
#[ignore = "dv"]
fn dv_dv4_probes_rendered_on_the_quest_meet_the_tier() {
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
            &probes.join(format!("{p}.request.json")),
            &format!("{p}.request.json"),
        );
    }
    // §1.13: the debug build runs with the validation layer (Q-123).
    let layers = DebugLayers::enable();
    let start = launch("dv4");
    let result = verdict("dv4", start, Duration::from_secs(300));
    stop();
    let validation = Validation::read();
    drop(layers);
    let (pass, detail, _) = result.expect("a dv4 verdict line in logcat");
    validation.assert_clean();
    assert!(pass, "dv4 FAIL {detail}");

    let tmp = std::env::temp_dir().join(format!("ftgsxr-dv4-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("a scratch directory");
    let mut failures = Vec::new();
    for p in PROBES {
        let name = format!("{p}.renders.safetensors");
        let local = tmp.join(&name);
        pull(&name, &local);
        let bytes = std::fs::read(&local).expect("the pulled renders");
        let (_, request) = probe(p);
        let ours = read_renders(&bytes, Some(request.sha256))
            .unwrap_or_else(|e| panic!("{p}: the device's renders read: {e}"));
        let references = reference_render(p);
        assert_eq!(ours.len(), references.len(), "{p}: one render per item");
        let mut min_psnr = f64::INFINITY;
        for (id, reference) in &references {
            let (_, image) = ours
                .iter()
                .find(|(oid, _)| oid == id)
                .unwrap_or_else(|| panic!("{p}: no device render for {id}"));
            let m = metrics(reference, image);
            min_psnr = min_psnr.min(m.psnr);
            if m.psnr < MIN_PSNR || m.share > MAX_SHARE || m.alpha > MAX_SHARE {
                failures.push(format!(
                    "{p} {id}: PSNR {:.2} dB, s {:.4} %, s_α {:.4} %",
                    m.psnr,
                    100.0 * m.share,
                    100.0 * m.alpha
                ));
            }
        }
        eprintln!("{p}: minimum PSNR {min_psnr:.2} dB");
    }
    let _ = std::fs::remove_dir_all(&tmp);
    assert!(
        failures.is_empty(),
        "items outside E15's tier:\n{}",
        failures.join("\n")
    );
}
