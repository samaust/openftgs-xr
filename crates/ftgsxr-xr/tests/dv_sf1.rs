// SPDX-License-Identifier: Apache-2.0

//! Spike S-F1 (PRD #804, #805, `dv` lane; §1.12 Headset display, D-58, E16's floor,
//! Appendix E), on Q-420's pattern extended by Q-454: the driver writes Q-420's
//! fill-heavy model, pushes it as `dv/spike.safetensors`, launches the runner's `sf1`
//! and reads the lines of `display::sf1`: the density-map features and properties, then
//! at render scale 1.0 and E16's floor of 0.6 each level's window summary (the first
//! eye's `GpuTimes`, Q-125) and its counters window. The spike judges nothing: the test
//! fails only on an incomplete sequence or a missing line, never on a figure or a
//! validation message, which it reports with the layer's announcement (Q-123). Run it
//! with `--nocapture` for the report. Prerequisites: `dv_support`'s.
// Q-454, Q-420

use crate::dv_spikes::every_row_persistent;
use crate::dv_support;

use std::time::Duration;

use dv_support::{
    DebugLayers, Validation, apk, clear, device, install, launch, log, push, stop, verdict,
};
use ftgsxr_core::write_exchange;

/// Q-420's rows; `FTGSXR_SPIKES_ROWS` sets another count, as for the spikes.
const ROWS: usize = 10_000;
const ROWS_VAR: &str = "FTGSXR_SPIKES_ROWS";
/// The main log buffer for the run (the spikes' size).
const LOG_BUFFER: &str = "16M";
const SCALES: [&str; 2] = ["1.0", "0.6"];
const LEVELS: [&str; 4] = ["Off", "Low", "Medium", "High"];

/// The lines the spike must have logged that are missing from `lines`, one entry each:
/// the feature line, each window's summary and counters summary at each level and scale,
/// each level applied to the swapchain, and the layer at each scale.
fn missing(lines: &[String]) -> Vec<String> {
    let has = |f: &dyn Fn(&str) -> bool| lines.iter().any(|l| f(l));
    let mut out = Vec::new();
    if !has(&|l| l.contains("sf1: density map: VK_EXT_fragment_density_map ")) {
        out.push("the density-map feature line".to_owned());
    }
    for s in SCALES {
        for l in LEVELS {
            let summary = format!("sf1: scale {s} {l} draw median ");
            if !has(&|x| x.contains(&summary)) {
                out.push(format!("the summary at {s} {l}"));
            }
            let counters = format!("perf metrics sf1 {s} {l}: ");
            if !has(&|x| x.contains(&counters) && x.contains(" frames over ")) {
                out.push(format!("the counters window at {s} {l}"));
            }
        }
        let layer = format!(" at render scale {}", s.trim_end_matches(".0"));
        if !has(&|x| x.contains("splat layer ") && x.contains(&layer)) {
            out.push(format!("the layer at render scale {s}"));
        }
    }
    for l in LEVELS {
        let applied = format!("foveation {l} applied to the swapchain");
        if !has(&|x| x.contains(&applied)) {
            out.push(format!("foveation {l} applied"));
        }
    }
    out
}

#[test]
#[ignore = "dv"]
fn dv_sf1_density_map_features_and_foveation_timings_at_two_scales() {
    let apk = apk();
    device();
    install(&apk);
    let dir = std::env::temp_dir().join(format!("ftgsxr-sf1-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a temporary directory");
    let file = dir.join("spike.safetensors");
    let rows = std::env::var(ROWS_VAR)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(ROWS);
    write_exchange(&file, &every_row_persistent(1, rows)).expect("the model is written");
    clear();
    push(&file, "spike.safetensors");
    std::fs::remove_dir_all(&dir).ok();
    dv_support::adb(&["logcat", "-G", LOG_BUFFER]);
    // The debug APK's validation layer (Q-123), reported, not judged: whether the layer's
    // pipelines need a pass with the map is what the spike finds (#805).
    let layers = DebugLayers::enable();
    let start = launch("sf1");
    let result = verdict("sf1", start, Duration::from_secs(420));
    let lines = log();
    let validation = Validation::read();
    stop();
    drop(layers);
    println!(
        "--- validation: layer announced {}, {} VALIDATION lines ---",
        validation.announced,
        validation.messages.len()
    );
    for m in validation.messages.iter().take(20) {
        println!("{m}");
    }
    println!("--- sf1 report ({rows} rows, degree 0, every row persistent) ---");
    for l in lines.iter().filter(|l| {
        l.contains("sf1: ")
            || l.contains("perf metrics sf1")
            || l.contains("splat layer ")
            || l.contains("foveation ")
            || l.contains("swapchain ")
            || l.contains("Validation")
            || l.contains("VUID")
            || l.contains("E-XR-")
    }) {
        println!("{l}");
    }
    let (pass, detail, _) = result.expect("an sf1 verdict line in logcat");
    assert!(pass, "sf1 FAIL {detail}");
    let gone = missing(&lines);
    assert!(gone.is_empty(), "missing: {gone:?}");
}

/// The line check (a `cpu` test of the driver's own logic): a complete log passes, and
/// each kind of line, taken out, is reported.
#[test]
fn missing_names_each_absent_line() {
    let mut log = vec![
        "sf1: density map: VK_EXT_fragment_density_map true (enabled false), …".to_owned(),
        "splat layer 1680x1760 for a 1680x1760 view at render scale 1".to_owned(),
        "splat layer 1008x1056 for a 1680x1760 view at render scale 0.6".to_owned(),
    ];
    for s in SCALES {
        for l in LEVELS {
            log.push(format!("sf1: scale {s} {l} draw median 5 ms, …"));
            log.push(format!(
                "perf metrics sf1 {s} {l}: 300 frames over 2.5 s, …"
            ));
        }
    }
    for l in LEVELS {
        log.push(format!("foveation {l} applied to the swapchain"));
    }
    assert!(missing(&log).is_empty(), "{:?}", missing(&log));
    let without = |needle: &str| {
        let l: Vec<String> = log
            .iter()
            .filter(|x| !x.contains(needle))
            .cloned()
            .collect();
        missing(&l)
    };
    assert_eq!(without("density map"), ["the density-map feature line"]);
    assert_eq!(without("scale 0.6 High draw"), ["the summary at 0.6 High"]);
    assert_eq!(without("sf1 1.0 Low:"), ["the counters window at 1.0 Low"]);
    assert_eq!(
        without("render scale 0.6"),
        ["the layer at render scale 0.6"]
    );
    assert_eq!(
        without("foveation Medium applied"),
        ["foveation Medium applied"]
    );
}
