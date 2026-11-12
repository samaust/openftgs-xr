// SPDX-License-Identifier: Apache-2.0

//! The session benchmarks on the headset (§3.5 B-2, B-7, B-8, B-9; `dv` lane; Q-406,
//! Q-407, Q-432 to Q-435): the driver pushes the model, grants the permissions
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
    let (file, text) = run_unjudged(name, model, pushed, timeout);
    // Q-T734.1: the host compares the pulled file with its stored `.quest.json`.
    dv_support::compare_with_baseline(&text);
    file
}

/// `run` without the baseline comparison: the pulled file, checked well formed, and its
/// text (kept under `$FTGSXR_DV_OUT` as `<NAME>.json`).
fn run_unjudged(
    name: &str,
    model: &std::path::Path,
    pushed: &str,
    timeout: Duration,
) -> (Value, String) {
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
    // As Dv-11's driver: the camera permissions go back; USE_SCENE stays, as Dv-7 leaves it.
    pm(name, "revoke", &PERMISSIONS[1..]);
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
    dv_support::keep(&format!("{name}.json"), &text);
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
    (file, text)
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
        ("quest-balanced", "90"),
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
    // §3.5 B-9, Q-222, Q-434: Q-222's probe, written here and pushed.
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

#[test]
#[ignore = "dv"]
fn dv_dv5_budgets_p1_p3_p4_on_the_budget_model() {
    // §3.4 Dv-5, §1.12 P-1, P-3, P-4 (Q-407, Q-438, Q-439): the budget model in
    // quest-balanced, passthrough and occlusion on, held where orbit camera 0 sees it and
    // played at speed 1 over its span; every block's GPU time ≤ 12.4 ms, the load ≤ 1 s
    // per 100 MB, every block's sort ≤ 2.5 ms. Unlike the benchmarks, a miss fails.
    let file = run(
        "Dv-5",
        &budget_model(),
        BUDGET_MODEL,
        Duration::from_secs(20 * 60),
    );
    print(&file);
    let value = |measure: &str| {
        let r = per_setup(&file, measure);
        assert_eq!(r.len(), 1, "one {measure}");
        r[0]["value"]
            .as_f64()
            .unwrap_or_else(|| panic!("{measure}: a number"))
    };
    let setup = &per_setup(&file, "frame_gpu_ms_max")[0]["setup"];
    assert_eq!(setup["tier"], "quest-balanced");
    assert_eq!(setup["passthrough"], true);
    assert_eq!(setup["occlusion"], true);
    assert_eq!(setup["orbit_camera"], 0);
    assert_eq!(setup["speed"], 1);
    let span = setup["span_s"].as_f64().expect("span_s");
    assert!(value("play_s") >= span, "the play covers the span");
    assert!(value("blocks") > 0.0, "blocks arrived");
    println!(
        "Dv-5 worst: f_gpu {:.3} ms (median {:.3}, {} of {} blocks over 12.4), sort {:.3} ms ({} over 2.5), \
         load {:.3} s for {} bytes = {:.3} s per 100 MB, {} frames in {:.2} s, visible max {}, overflow blocks {}",
        value("frame_gpu_ms_max"),
        value("frame_gpu_ms_median"),
        value("blocks_over_p1"),
        value("blocks"),
        value("frame_sort_ms_max"),
        value("blocks_over_p4"),
        value("load_s"),
        value("file_bytes"),
        value("s_per_100mb"),
        value("frames"),
        value("play_s"),
        value("visible_max"),
        value("overflow_blocks"),
    );
    let budgets = file["budgets"].as_array().expect("budgets");
    let ids: Vec<&str> = budgets.iter().filter_map(|b| b["id"].as_str()).collect();
    let overflowed = value("overflow_blocks") > 0.0;
    let expected: &[&str] = if overflowed {
        &["P-3", "P-4"]
    } else {
        &["P-1", "P-3", "P-4"]
    };
    assert_eq!(ids, expected, "Dv-5's budgets");
    let missed: Vec<String> = budgets
        .iter()
        .filter(|b| b["pass"] != true)
        .map(|b| {
            format!(
                "{} {} = {} > {}",
                b["id"], b["measure"], b["value"], b["limit"]
            )
        })
        .collect();
    assert!(missed.is_empty(), "Dv-5 missed: {}", missed.join("; "));
}

/// #622's 3,000,000-row `kitchen` model, the sweep's source (Q-T734.2).
const SWEEP_SOURCE: &str = "kitchen.safetensors";

/// The counts the sweep reduces the source to (Q-409, Q-T734.2).
const SWEEP_COUNTS: [usize; 6] = [250_000, 500_000, 750_000, 1_000_000, 1_250_000, 1_500_000];

/// The Quest tiers, in §1.12's order.
const QUEST_TIERS: [&str; 3] = ["quest-high", "quest-balanced", "quest-low"];

/// P-1's limit at 72 Hz (§1.12), the limit every count is held to (Q-409, Q-T734.2).
const SWEEP_LIMIT_MS: f64 = 12.4;

/// One tier's orbit at the runtime's rate on the model reduced to `n` rows: its
/// slowest camera's f_gpu and its overflowing cameras, `None` where skipped.
#[derive(Clone, Debug, PartialEq)]
struct SweepPoint {
    n: usize,
    tier: String,
    gpu_ms_max: Option<f64>,
    overflow_cameras: Option<f64>,
}

/// The sweep's points of one B-2 file, run on the model reduced to `n` rows.
fn sweep_points(file: &Value, n: usize) -> Vec<SweepPoint> {
    let figure = |tier: &str, measure: &str| {
        per_setup(file, measure)
            .into_iter()
            .find(|r| {
                r["setup"]["tier"] == tier
                    && r["setup"]["view"] == "orbit"
                    && r["setup"]["refresh_rate"] == "runtime"
            })
            .and_then(|r| r["value"].as_f64())
    };
    QUEST_TIERS
        .iter()
        .map(|tier| SweepPoint {
            n,
            tier: (*tier).to_owned(),
            gpu_ms_max: figure(tier, "gpu_ms_max"),
            overflow_cameras: figure(tier, "overflow_cameras"),
        })
        .collect()
}

/// Each Quest tier's derived figure (Q-T734.2): the largest count whose orbit at the
/// runtime's rate holds 12.4 ms with no overflowing camera, `None` when no count does.
// Q-T734.2
fn derive_capacities(points: &[SweepPoint]) -> Vec<(String, Option<usize>)> {
    QUEST_TIERS
        .iter()
        .map(|tier| {
            let held = points
                .iter()
                .filter(|p| p.tier == *tier)
                .filter(|p| {
                    p.gpu_ms_max.is_some_and(|ms| ms <= SWEEP_LIMIT_MS)
                        && p.overflow_cameras == Some(0.0)
                })
                .map(|p| p.n)
                .max();
            ((*tier).to_owned(), held)
        })
        .collect()
}

/// `model`'s rows `rows` (file order), its SH truncated to `degree` (≤ the model's).
fn reduced(model: &ftgsxr_core::Model, rows: &[u32], degree: u8) -> ftgsxr_core::Model {
    let k_in = (usize::from(model.sh_degree) + 1).pow(2) * 3;
    let k_out = (usize::from(degree) + 1).pow(2) * 3;
    let pick = |i: &u32| *i as usize;
    let mut m = model.clone();
    m.position = rows.iter().map(|i| model.position[pick(i)]).collect();
    m.time_center = rows.iter().map(|i| model.time_center[pick(i)]).collect();
    m.duration = rows.iter().map(|i| model.duration[pick(i)]).collect();
    m.velocity = rows.iter().map(|i| model.velocity[pick(i)]).collect();
    m.scale = rows.iter().map(|i| model.scale[pick(i)]).collect();
    m.rotation = rows.iter().map(|i| model.rotation[pick(i)]).collect();
    m.opacity = rows.iter().map(|i| model.opacity[pick(i)]).collect();
    m.sh = rows
        .iter()
        .flat_map(|i| {
            model.sh[pick(i) * k_in..pick(i) * k_in + k_out]
                .iter()
                .copied()
        })
        .collect();
    m.sh_degree = degree;
    m
}

#[test]
#[ignore = "dv"]
fn dv_b2_capacity_sweep_on_the_3m_kitchen() {
    // Q-409's alternative (owner, 2026-11-12, #734), Q-T734.2: B-2 on #622's 3 M kitchen
    // reduced by D-13's rule to each count at degree 2; each tier's derived figure is
    // printed for the owner, not judged: the tier table and §1.12 are unchanged.
    let root = std::env::var_os("OPENFTGS_DATASETS")
        .expect("$OPENFTGS_DATASETS names the datasets (#622's kitchen, Q-T734.2)");
    let source = PathBuf::from(root).join("xr-fixtures").join(SWEEP_SOURCE);
    let model = ftgsxr_core::open_model(&source).expect("#622's kitchen model");
    assert!(model.sh_degree >= 2, "a source of degree 2 or more");
    let dir = std::env::temp_dir().join(format!("ftgsxr-dv-sweep-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a temporary directory");
    let mut points = Vec::new();
    for n in SWEEP_COUNTS {
        assert!(n <= model.position.len(), "{n} rows of the source");
        let rows = ftgsxr_core::reduce_to_budget(&model, n);
        let path = dir.join("kitchen_sweep.safetensors");
        write_exchange(&path, &reduced(&model, &rows, 2)).expect("the reduced model is written");
        let (file, text) = run_unjudged(
            "B-2",
            &path,
            "kitchen_sweep.safetensors",
            Duration::from_secs(2 * 3600),
        );
        dv_support::keep(&format!("B-2.sweep-{n}.json"), &text);
        print(&file);
        let p = sweep_points(&file, n);
        for q in &p {
            println!(
                "sweep N = {n}: {} orbit f_gpu max {:?} ms, overflowing cameras {:?}",
                q.tier, q.gpu_ms_max, q.overflow_cameras
            );
        }
        points.extend(p);
    }
    std::fs::remove_dir_all(&dir).ok();
    for (tier, held) in derive_capacities(&points) {
        match held {
            Some(n) => println!("derived: {tier} holds {SWEEP_LIMIT_MS} ms up to N = {n}"),
            None => println!(
                "derived: {tier} holds {SWEEP_LIMIT_MS} ms at no swept count (below {})",
                SWEEP_COUNTS[0]
            ),
        }
    }
}

#[test]
fn derive_capacities_takes_the_largest_count_that_holds() {
    // Q-T734.2: a count holds for a tier when its orbit at the runtime's rate is at most
    // 12.4 ms with no overflowing camera; the derived figure is the largest that holds,
    // None ("below the smallest count") when none does.
    let point = |n: usize, tier: &str, ms: f64, overflow: f64| SweepPoint {
        n,
        tier: tier.to_owned(),
        gpu_ms_max: Some(ms),
        overflow_cameras: Some(overflow),
    };
    let points = [
        point(250_000, "quest-low", 9.0, 0.0),
        point(500_000, "quest-low", 12.4, 0.0),
        point(750_000, "quest-low", 13.0, 0.0),
        point(250_000, "quest-balanced", 11.0, 0.0),
        point(500_000, "quest-balanced", 12.0, 1.0),
        point(250_000, "quest-high", 20.0, 0.0),
        SweepPoint {
            n: 500_000,
            tier: "quest-high".into(),
            gpu_ms_max: None,
            overflow_cameras: None,
        },
    ];
    assert_eq!(
        derive_capacities(&points),
        vec![
            ("quest-high".to_owned(), None),
            ("quest-balanced".to_owned(), Some(250_000)),
            ("quest-low".to_owned(), Some(500_000)),
        ]
    );
}

#[test]
fn reduced_keeps_the_rows_in_file_order_and_the_low_sh_coefficients() {
    // Q-T734.2: rows 2 and 0 of a degree-1 model at degree 0 keep each row's fields and
    // its first coefficient's RGB (E4's coefficient-major order, RGB innermost).
    let mut m = ftgsxr_testkit::random_model(1, 3, 1);
    m.sh = (0..36).map(|v| v as f32).collect();
    let r = reduced(&m, &[0, 2], 0);
    assert_eq!(r.sh_degree, 0);
    assert_eq!(r.position, vec![m.position[0], m.position[2]]);
    assert_eq!(r.opacity, vec![m.opacity[0], m.opacity[2]]);
    assert_eq!(r.duration, vec![m.duration[0], m.duration[2]]);
    assert_eq!(r.sh, vec![0.0, 1.0, 2.0, 24.0, 25.0, 26.0]);
}
