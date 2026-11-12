// SPDX-License-Identifier: Apache-2.0

//! The session benchmarks on the headset (§3.5 B-2, B-7, B-8, B-9; `dv` lane; Q-406,
//! Q-407, Q-T731.1 to Q-T731.4): the driver pushes the model, grants the permissions
//! the setups need (occlusion's USE\_SCENE, the lighting modes' cameras) so no dialog
//! waits for a person, launches the app with the benchmark in `ftgsxr-bench`, the
//! fingerprint in `ftgsxr-bench-headset` and the model's name in `ftgsxr-bench-model`,
//! waits for `ftgsxr-bench: <NAME> DONE|FAIL`, pulls `dv/<NAME>.json` and checks it well
//! formed (Q-67, Q-103) with `machine.headset`. The figures are printed for the PRD
//! (run with `--nocapture`); a budget the Quest misses is printed, not failed, and no
//! baseline is written (Q-406: after the owner's review). Android's GPU debug layers stay
//! off, since the runs time the GPU (Q-418). Prerequisites: `dv_support`'s, and
//! `$OPENFTGS_DATASETS/xr-fixtures/kitchen_1m.safetensors` (the budget model, Q-410).

use crate::dv_support;

use std::path::PathBuf;
use std::time::Duration;

use dv_support::{
    apk, bench_line, clear, device, getprop, install, launch_session_bench, pm, pull, push, stop,
};
use ftgsxr_core::write_exchange;
use serde_json::Value;

/// The permissions of §2.5's manifest the setups use: depth occlusion (Appendix E.1) and
/// light estimation's passthrough camera (§1.9).
const PERMISSIONS: [&str; 3] = [
    "com.oculus.permission.USE_SCENE",
    "android.permission.CAMERA",
    "horizonos.permission.HEADSET_CAMERA",
];

/// The budget model's file name, pushed under it.
const BUDGET_MODEL: &str = "kitchen_1m.safetensors";

/// `$OPENFTGS_DATASETS/xr-fixtures/kitchen_1m.safetensors` (Q-410).
fn budget_model() -> PathBuf {
    let root = std::env::var_os("OPENFTGS_DATASETS")
        .expect("$OPENFTGS_DATASETS names the datasets (the budget model, Q-410)");
    let path = PathBuf::from(root).join("xr-fixtures").join(BUDGET_MODEL);
    assert!(path.is_file(), "the budget model at {}", path.display());
    path
}

/// Runs the session benchmark `name` on `model` (pushed as `pushed`) and returns its
/// pulled file, checked well formed.
fn run(name: &str, model: &std::path::Path, pushed: &str, timeout: Duration) -> Value {
    let apk = apk();
    device();
    install(&apk);
    stop();
    clear();
    push(model, pushed);
    pm(name, "grant", &PERMISSIONS);
    let fingerprint = getprop("ro.build.fingerprint");
    assert!(!fingerprint.is_empty(), "a build fingerprint");
    let start = launch_session_bench(name, &fingerprint, pushed);
    let line = bench_line(name, start, timeout);
    stop();
    let (done, detail) =
        line.unwrap_or_else(|| panic!("an ftgsxr-bench line for {name} (none: killed?)"));
    assert!(done, "{name} FAIL {detail}");
    println!(
        "{name} DONE {detail} in {:.0} s",
        start.elapsed().as_secs_f64()
    );
    let local = std::env::temp_dir().join(format!("ftgsxr-dv-{name}-{}.json", std::process::id()));
    pull(&format!("{name}.json"), &local);
    let text = std::fs::read_to_string(&local).expect("the pulled file");
    let _ = std::fs::remove_file(&local);
    let file: Value = serde_json::from_str(&text).expect("well-formed JSON");
    // Q-67, Q-103's top level.
    assert_eq!(file["format"], "ftgsxr-bench");
    assert_eq!(file["version"], 1);
    assert_eq!(file["bench"], name);
    for field in ["gpu", "driver", "os", "cpu"] {
        assert!(file["machine"][field].is_string(), "machine.{field}");
    }
    assert_eq!(file["machine"]["headset"], fingerprint.as_str());
    for field in ["ftgsxr", "bevy", "wgpu"] {
        assert!(file["versions"][field].is_string(), "versions.{field}");
    }
    assert!(file["profile"].is_string());
    for r in file["results"].as_array().expect("results") {
        assert!(
            r["value"].is_number() || (r["value"].is_null() && r["skipped"].is_string()),
            "{r}"
        );
        assert!(r["measure"].is_string() && r["unit"].is_string(), "{r}");
    }
    assert!(file["budgets"].is_array());
    file
}

/// The results of `measure` without a `camera` (the setups' figures).
fn per_setup<'a>(file: &'a Value, measure: &str) -> Vec<&'a Value> {
    file["results"]
        .as_array()
        .expect("results")
        .iter()
        .filter(|r| r["measure"] == measure && r["setup"]["camera"].is_null())
        .collect()
}

/// Prints each setup's figures, the derived ones, the counters and the budgets.
fn print(file: &Value) {
    let profile = file["profile"].as_str().unwrap_or("?");
    println!("--- {} ({profile} build) ---", file["bench"]);
    for r in file["results"].as_array().expect("results") {
        if !r["setup"]["camera"].is_null() {
            continue;
        }
        let s = &r["setup"];
        let what = match s["counter"].as_str() {
            Some(c) => format!("{} {c}", r["measure"].as_str().unwrap_or("?")),
            None => r["measure"].as_str().unwrap_or("?").to_owned(),
        };
        let value = match r["skipped"].as_str() {
            Some(why) => format!("skipped: {why}"),
            None => format!("{} {}", r["value"], r["unit"].as_str().unwrap_or("")),
        };
        println!(
            "{} {} sort {} rate {} occl {} light {}: {what} = {value}",
            s["tier"].as_str().unwrap_or("-"),
            s["view"].as_str().unwrap_or("-"),
            s["sort"].as_str().unwrap_or("-"),
            s["refresh_rate"],
            s["occlusion"],
            s["lighting"].as_str().unwrap_or("-"),
        );
    }
    for b in file["budgets"].as_array().expect("budgets") {
        println!(
            "budget {} {} {}: {} {} {} -> {}",
            b["id"].as_str().unwrap_or("?"),
            b["measure"].as_str().unwrap_or("?"),
            b["setup"],
            b["value"],
            b["rule"].as_str().unwrap_or("?"),
            b["limit"],
            if b["pass"] == true { "met" } else { "missed" }
        );
    }
    for n in file["notes"].as_array().expect("notes") {
        println!("note: {n}");
    }
}

#[test]
#[ignore = "dv"]
fn dv_b2_writes_the_frame_time_of_each_quest_tier() {
    // §3.5 B-2, §1.12 P-1: each Quest tier, orbit and center, then quest-balanced at
    // 90 Hz; P-1's budgets on quest-balanced's orbit (12.4 ms, 9.6 ms at 90 Hz), printed,
    // not failed (#731's acceptance).
    let file = run(
        "B-2",
        &budget_model(),
        BUDGET_MODEL,
        Duration::from_secs(6 * 3600),
    );
    print(&file);
    let maxima = per_setup(&file, "gpu_ms_max");
    let setups: Vec<(String, String, String)> = maxima
        .iter()
        .map(|r| {
            let s = &r["setup"];
            (
                s["tier"].as_str().unwrap_or("?").to_owned(),
                s["view"].as_str().unwrap_or("?").to_owned(),
                s["refresh_rate"].to_string(),
            )
        })
        .collect();
    let mut expected = Vec::new();
    for (tier, rate) in [
        ("quest-balanced", "\"runtime\""),
        ("quest-high", "\"runtime\""),
        ("quest-low", "\"runtime\""),
        ("quest-balanced", "90.0"),
    ] {
        for view in ["orbit", "center"] {
            expected.push((tier.to_owned(), view.to_owned(), rate.to_owned()));
        }
    }
    assert_eq!(setups, expected, "one result for each tier and view setup");
    for b in file["budgets"].as_array().expect("budgets") {
        assert_eq!(b["id"], "P-1", "{b}");
        assert_eq!(b["setup"]["tier"], "quest-balanced", "{b}");
        assert_eq!(b["setup"]["view"], "orbit", "{b}");
        let limit = if b["setup"]["refresh_rate"] == 90.0 {
            9.6
        } else {
            12.4
        };
        assert_eq!(b["limit"], limit, "{b}");
    }
}

#[test]
#[ignore = "dv"]
fn dv_b7_writes_the_sort_and_frame_times_in_both_sort_modes() {
    // §3.5 B-7: quest-balanced's orbit shared center and per view, the sort and frame
    // times and their ratio; the 1.5× budget printed.
    let file = run(
        "B-7",
        &budget_model(),
        BUDGET_MODEL,
        Duration::from_secs(3 * 3600),
    );
    print(&file);
    let sorts: Vec<&str> = per_setup(&file, "sort_ms_median")
        .iter()
        .map(|r| r["setup"]["sort"].as_str().unwrap_or("?"))
        .collect();
    assert_eq!(sorts, ["SharedCenter", "PerView"]);
    assert_eq!(per_setup(&file, "gpu_ms_median").len(), 2);
    assert_eq!(per_setup(&file, "sort_ratio").len(), 1, "the ratio");
    assert_eq!(
        per_setup(&file, "gpu_ratio").len(),
        1,
        "the frame times' ratio"
    );
}

#[test]
#[ignore = "dv"]
fn dv_b8_writes_the_cost_of_occlusion_and_each_lighting_mode() {
    // §3.5 B-8: occlusion off and on, then each lighting mode, on quest-balanced's orbit.
    let file = run(
        "B-8",
        &budget_model(),
        BUDGET_MODEL,
        Duration::from_secs(4 * 3600),
    );
    print(&file);
    let envs: Vec<(bool, String)> = per_setup(&file, "gpu_ms_median")
        .iter()
        .map(|r| {
            let s = &r["setup"];
            (
                s["occlusion"].as_bool().expect("occlusion"),
                s["lighting"].as_str().unwrap_or("?").to_owned(),
            )
        })
        .collect();
    let expected: Vec<(bool, String)> = [
        (false, "off"),
        (true, "off"),
        (true, "manual"),
        (true, "auto-global"),
        (true, "auto-local"),
    ]
    .iter()
    .map(|(o, l)| (*o, (*l).to_owned()))
    .collect();
    assert_eq!(
        envs, expected,
        "occlusion off and on and each lighting mode"
    );
}

#[test]
#[ignore = "dv"]
fn dv_b9_writes_the_quests_time_to_the_first_frame() {
    // §3.5 B-9, Q-222, Q-T731.3: Q-222's probe, written here and pushed.
    let dir = std::env::temp_dir().join(format!("ftgsxr-dv-b9-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a temporary directory");
    let probe = dir.join("b9-probe.safetensors");
    write_exchange(&probe, &dv_support::every_row_persistent(1, 100_000, 2))
        .expect("the probe is written");
    let file = run(
        "B-9",
        &probe,
        "b9-probe.safetensors",
        Duration::from_secs(10 * 60),
    );
    std::fs::remove_dir_all(&dir).ok();
    print(&file);
    for measure in ["uploaded_s", "first_frame_s"] {
        let r = per_setup(&file, measure);
        assert_eq!(r.len(), 1, "{measure}");
        assert!(r[0]["value"].as_f64().is_some_and(|v| v > 0.0), "{measure}");
        assert_eq!(r[0]["setup"]["run"], "first");
    }
}
