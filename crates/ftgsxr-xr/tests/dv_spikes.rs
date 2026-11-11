// SPDX-License-Identifier: Apache-2.0

//! PRD #718's session spikes S1 and S2 (#725, `dv` lane; §1.13 HUD, §1.12 Headset
//! display, D-27, E16's f_gpu, Appendix E.1), on Dv-12's pattern: the driver writes the
//! fill-heavy model (Q-T725.1), pushes it as `dv/spike.safetensors`, launches the
//! runner's `spikes` and reads its lines, `ftgsxr_xr`'s `perf metrics` lines and
//! `display`'s swapchain size. The spikes report and judge nothing: the test fails only
//! when the sequence does not complete or a line is missing, never on a figure (no
//! timestamp queries, no counters). It prints the lines for the PRD's comment; run it with
//! `--nocapture`. Prerequisites: `dv_support`'s.

use crate::dv_support;

use std::time::Duration;

use dv_support::{apk, clear, device, install, launch, log, push, stop, verdict};
use ftgsxr_core::{Model, write_exchange};

/// The model's rows by default (Q-T725.1). The draw cost was linear in N on the device,
/// about 0.77 µs a row (760 ms a frame at the Quest's capacity C = 10⁶, 80 ms at 10⁵,
/// where the statistics blocks came about once a second); 10⁴ puts the first eye's GPU
/// time near P-1's 12.4 ms and gives 120 blocks a level in seconds. `FTGSXR_SPIKES_ROWS`
/// sets another count.
const ROWS: usize = 10_000;
/// The variable naming another row count.
const ROWS_VAR: &str = "FTGSXR_SPIKES_ROWS";
/// The main log buffer for the run: the runtime's own lines overran the default 256 KiB
/// before the host read the runner's.
const LOG_BUFFER: &str = "16M";

/// B-1's every-row-persistent kind (Q-102, Q-T725.1): `random_model(1, ROWS, 0)` with each
/// transient row made persistent as E3's persistent rows are drawn, opacity in
/// \[0.6, 0.99\] and duration 2–4 T_τ, each mapped from the row's own uniform opacity draw
/// in \[0.002, 0.99\], so the model stays the same for the same arguments.
fn every_row_persistent(seed: u64, n: usize) -> Model {
    let mut m = ftgsxr_testkit::random_model(seed, n, 0);
    let span = m.time.tau1 - m.time.tau0;
    for i in 0..n {
        if m.duration[i] >= 2.0 * span {
            continue;
        }
        let u = ((f64::from(m.opacity[i]) - 0.002) / (0.99 - 0.002)).clamp(0.0, 1.0);
        m.opacity[i] = (0.6 + 0.39 * u) as f32;
        m.duration[i] = (2.0 + 2.0 * u) * span;
    }
    m
}

/// The lines of `lines` containing `needle`, after it.
fn after<'a>(lines: &'a [String], needle: &str) -> Vec<&'a str> {
    lines
        .iter()
        .filter_map(|l| l.split_once(needle).map(|(_, r)| r))
        .collect()
}

#[test]
#[ignore = "dv"]
fn dv_spikes_s1_timestamps_and_counters_s2_foveation_gpu_times() {
    let apk = apk();
    device();
    install(&apk);
    let dir = std::env::temp_dir().join(format!("ftgsxr-spikes-{}", std::process::id()));
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
    let start = launch("spikes");
    let result = verdict("spikes", start, Duration::from_secs(300));
    let lines = log();
    stop();

    let report: Vec<&String> = lines
        .iter()
        .filter(|l| {
            l.contains("spikes: ")
                || l.contains("perf metrics")
                || l.contains("swapchain ")
                || l.contains("recommended layer resolution")
                || l.contains("foveation ")
                || l.contains("E-XR-")
        })
        .collect();
    println!("--- spikes report ({rows} rows, degree 0, every row persistent) ---");
    for l in &report {
        println!("{l}");
    }
    let (pass, detail, _) = result.expect("a spikes verdict line in logcat");
    assert!(pass, "spikes FAIL {detail}");

    assert_eq!(
        after(&lines, "spikes: timestamp queries on the XR device: ").len(),
        1,
        "S1's timestamp-query line"
    );
    assert_eq!(after(&lines, "spikes: s1 ").len(), 1, "S1's blocks line");
    // Each window's summary header (logged after its marker goes), not its start line.
    let summary = |label: &str| {
        after(&lines, &format!("perf metrics {label}: "))
            .iter()
            .any(|r| r.contains(" frames over "))
    };
    assert!(summary("s1"), "S1's counters window summary");
    for level in ["Off", "Low", "Medium", "High"] {
        let s2 = after(&lines, &format!("spikes: s2 {level} median total "));
        assert_eq!(s2.len(), 1, "S2's line at {level}");
        let ms = s2[0].split_whitespace().next().unwrap_or("");
        assert!(ms.parse::<f32>().is_ok(), "S2 at {level}: a median, {ms:?}");
        assert!(
            summary(&format!("s2 {level}")),
            "S2's counters window summary at {level}"
        );
        // As Dv-12: the level reached the swapchain, so the medians are of four levels.
        assert!(
            lines
                .iter()
                .any(|l| l.contains(&format!("foveation {level} applied to the swapchain"))),
            "foveation {level} applied"
        );
    }
    let swapchain = lines.iter().find_map(|l| size_after(l, "swapchain "));
    println!("swapchain {swapchain:?}");
    assert!(swapchain.is_some(), "display's swapchain size");
}

/// `(w, h)` from the `WxH` after `prefix` in `line` (Dv-12's reading).
fn size_after(line: &str, prefix: &str) -> Option<(u32, u32)> {
    let rest = line.split_once(prefix)?.1;
    let word = rest.split_whitespace().next()?;
    let (w, h) = word.split_once('x')?;
    Some((w.parse().ok()?, h.parse().ok()?))
}

/// The model generator (private to the driver, so a `cpu` test of its own): every row
/// is persistent by E3's draw, and the model is valid under §1.2.
#[test]
fn every_row_persistent_makes_every_row_persistent() {
    let m = every_row_persistent(1, 2000);
    let span = m.time.tau1 - m.time.tau0;
    assert!(m.validate().is_ok());
    assert!(
        m.duration
            .iter()
            .all(|&d| (2.0 * span..=4.0 * span).contains(&d))
    );
    assert!(m.opacity.iter().all(|&o| (0.6..=0.99).contains(&o)));
    assert_eq!(m.sh_degree, 0);
    assert_eq!(
        every_row_persistent(1, 2000),
        m,
        "the same for the same arguments"
    );
}
