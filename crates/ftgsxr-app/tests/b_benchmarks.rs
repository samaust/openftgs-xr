// SPDX-License-Identifier: Apache-2.0

//! The desktop benchmarks of §3.5 through `cmd_bench` (§1.7 `bench`, Q-67, Q-102,
//! Q-103): the NAME check (`cpu` lane) and, on a tiny model given with `--model`, the
//! file each benchmark writes (`gpu` lane). These runs are not recorded timings: the
//! baselines come from the full sweeps run alone on the machine.

use std::path::{Path, PathBuf};

use ftgsxr_app::{HeadlessError, cmd_bench, exit_code};
use serde_json::Value;

#[test]
fn bench_refuses_a_benchmark_the_desktop_does_not_run() {
    // §3.5: B-2, B-7 and B-8 are the Quest's or reported elsewhere (Q-103); B-9's
    // desktop half runs here (Q-155, #289), and B-4 on both machines (Q-406, #733).
    for name in ["B-2", "B-7", "B-8", "B-10", "b-1", ""] {
        let e = cmd_bench(name, None, None).expect_err(name);
        assert!(matches!(e, HeadlessError::Input(_)), "{name}: {e:?}");
        assert_eq!(exit_code(&e), 2);
    }
}

#[test]
fn bench_refuses_an_invalid_model_before_the_gpu() {
    let dir = ftgsxr_testkit::scratch_dir(
        env!("CARGO_TARGET_TMPDIR"),
        crate::test_file_stem(),
        "invalid",
    );
    let path = dir.join("broken.ftgs-if");
    std::fs::write(&path, b"not a safetensors file").unwrap();
    let e = cmd_bench("B-6", None, Some(&path)).expect_err("an invalid model");
    assert!(matches!(e, HeadlessError::Input(_)), "{e:?}");
    assert!(e.code().starts_with("E-FILE"), "{e}");
}

/// A tiny model of `n` rows at `degree`, written as a `ftgs-if` file in its own scratch
/// directory (`--model`'s input).
fn tiny_model(name: &str, n: usize, degree: u8) -> (PathBuf, ftgsxr_core::Model) {
    let dir =
        ftgsxr_testkit::scratch_dir(env!("CARGO_TARGET_TMPDIR"), crate::test_file_stem(), name);
    let model = ftgsxr_testkit::random_model(17, n, degree);
    let path = dir.join("tiny.ftgs-if");
    std::fs::write(&path, ftgsxr_testkit::exchange_bytes(&model)).unwrap();
    (path, model)
}

/// Runs `name` on `model` with `--out`, checks the report is the file and Q-103's
/// top level, and returns the parsed file.
fn run(name: &str, model: Option<&Path>, dir_name: &str) -> Value {
    let dir = ftgsxr_testkit::scratch_dir(
        env!("CARGO_TARGET_TMPDIR"),
        crate::test_file_stem(),
        dir_name,
    );
    let out = dir.join("bench.json");
    let report = cmd_bench(name, Some(&out), model).expect("the benchmark runs");
    let text = std::fs::read_to_string(&out).expect("the file is written");
    assert_eq!(report.json, text);
    let file: Value = serde_json::from_str(&text).expect("well-formed JSON");
    // §1.13, Q-116: `cmd_bench` turns wgpu's validation errors into a note, not an
    // error, and R-16's guard sees only the Khronos layer's records, so a note fails here.
    assert!(
        !text.contains("GPU validation reported"),
        "the run reported GPU validation messages: {}",
        file["notes"]
    );
    assert_eq!(file["format"], "ftgsxr-bench");
    assert_eq!(file["version"], 1);
    assert_eq!(file["bench"], name);
    // Q-406 (#724): `cmd_bench`'s file has no `machine.headset`.
    assert!(
        file["machine"].get("headset").is_none(),
        "{}",
        file["machine"]
    );
    for key in ["gpu", "driver", "os", "cpu"] {
        assert!(
            file["machine"][key].as_str().is_some_and(|s| !s.is_empty()),
            "{key}"
        );
    }
    assert!(file["machine"]["threads"].as_u64().is_some_and(|t| t >= 1));
    assert_eq!(file["versions"]["ftgsxr"], env!("CARGO_PKG_VERSION"));
    assert_eq!(file["versions"]["bevy"], "0.19.1");
    assert!(file["versions"]["wgpu"].as_str().is_some());
    for budget in file["budgets"].as_array().unwrap() {
        for key in ["id", "setup", "measure", "rule", "limit", "value", "pass"] {
            assert!(!budget[key].is_null(), "{key} in {budget}");
        }
    }
    file
}

/// The value of `measure` among `file`'s results.
fn result(file: &Value, measure: &str) -> f64 {
    file["results"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["measure"] == measure)
        .unwrap_or_else(|| panic!("no {measure}"))["value"]
        .as_f64()
        .unwrap()
}

#[test]
#[ignore = "gpu"]
fn gpu_b1_with_a_model_writes_a_well_formed_file() {
    let _layer = ftgsxr_testkit::layer_messages(); // R-16, Q-105
    // Q-67's first detecting test: B-1 writes a well-formed file; with `--model` its one
    // setup carries P-2's budget (Q-102).
    let (path, model) = tiny_model("b1", 2000, 2);
    let file = run("B-1", Some(&path), "b1-out");
    let identity: String = model.identity.iter().map(|b| format!("{b:02x}")).collect();
    let first = &file["results"][0];
    assert_eq!(first["setup"]["model"], identity.as_str());
    assert_eq!(first["setup"]["n"], 2000);
    assert_eq!(first["setup"]["width"], 1920);
    assert_eq!(first["setup"]["height"], 1080);
    // Q-102 (#199): `fps_min` = 1000 / the slowest camera's ms, which the file gives as
    // `frame_ms_max`; both fps figures are finite and positive.
    let (fps_min, fps_mean) = (result(&file, "fps_min"), result(&file, "fps_mean"));
    assert!(fps_min.is_finite() && fps_min > 0.0, "{fps_min}");
    assert!(fps_mean.is_finite() && fps_mean > 0.0, "{fps_mean}");
    let slowest = result(&file, "frame_ms_max");
    assert!(
        (fps_min - 1e3 / slowest).abs() <= 1e-9 * fps_min,
        "fps_min {fps_min} against 1000 / {slowest}"
    );
    assert!(result(&file, "visible_splats") <= 2000.0);
    let budgets = file["budgets"].as_array().unwrap();
    assert_eq!(budgets.len(), 1);
    assert_eq!(budgets[0]["id"], "P-2");
    assert_eq!(budgets[0]["limit"], 90);
}

#[test]
#[ignore = "gpu"]
fn gpu_b1_reports_the_draw_time_and_the_synthetic_load_note() {
    let _layer = ftgsxr_testkit::layer_messages(); // R-16, Q-105
    // Q-114's detecting test: B-1 times its splat draw (`GpuTimes::draw`, Q-81, §1.13),
    // and `total`, from the cull's begin to the last timestamp, now spans the draw too.
    let (path, _) = tiny_model("b1-draw", 2000, 2);
    let file = run("B-1", Some(&path), "b1-draw-out");
    let (cull, sort, draw, total) = (
        result(&file, "gpu_cull_ms"),
        result(&file, "gpu_sort_ms"),
        result(&file, "gpu_draw_ms"),
        result(&file, "gpu_total_ms"),
    );
    assert!(cull > 0.0, "{cull}");
    assert!(draw > 0.0, "{draw}");
    assert!(
        total >= sort + draw,
        "total {total}, sort {sort}, draw {draw}"
    );
    // The figures are reported, not budgeted: P-2 stays B-1's one budget.
    let budgets = file["budgets"].as_array().unwrap();
    assert_eq!(budgets.len(), 1);
    assert_eq!(budgets[0]["measure"], "fps_min");
    // §1.12 P-2 ("the same model") against the synthetic sweep's load, until `kitchen` (Q-108).
    let notes = file["notes"].as_array().unwrap();
    assert!(
        notes.iter().any(|n| {
            let n = n.as_str().unwrap();
            n.contains("synthetic") && n.contains("P-2") && n.contains("Q-108")
        }),
        "{notes:?}"
    );
}

#[test]
#[ignore = "gpu"]
fn gpu_b3_with_a_model_reports_sort_times() {
    let _layer = ftgsxr_testkit::layer_messages(); // R-16, Q-105
    let (path, _) = tiny_model("b3", 5000, 0);
    let file = run("B-3", Some(&path), "b3-out");
    assert!(result(&file, "sort_ms") > 0.0);
    let keys = result(&file, "keys");
    assert!(keys > 0.0 && keys <= 5000.0, "{keys}");
    // P-4 is defined at C = 10⁶ only (Q-102).
    assert!(file["budgets"].as_array().unwrap().is_empty());
}

/// The value of `measure` in the setup whose `index` is `index` (B-4, Q-T733.1).
fn b4_result(file: &Value, index: &str, measure: &str) -> f64 {
    file["results"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["measure"] == measure && r["setup"]["index"] == index)
        .unwrap_or_else(|| panic!("no {measure} with index {index}"))["value"]
        .as_f64()
        .unwrap()
}

#[test]
#[ignore = "gpu"]
fn gpu_b4_with_a_model_reports_the_cull_with_and_without_the_index() {
    // §3.5 B-4 (Q-408, Q-T733.1): for the size, the cull time and the rows visited with the
    // temporal index and with every row; the index visits no more rows than N and is not
    // slower than the every-row cull by more than the run's spread. `random_model`'s rows
    // are mostly transient, so at mid-span the index visits fewer than N (E3).
    let _layer = ftgsxr_testkit::layer_messages(); // R-16, Q-105
    let n = 100_000;
    let (path, _) = tiny_model("b4", n, 0);
    let file = run("B-4", Some(&path), "b4-out");
    let (indexed, every) = ("temporal", "every row");
    assert_eq!(b4_result(&file, every, "rows_visited"), n as f64);
    let rows = b4_result(&file, indexed, "rows_visited");
    assert!(rows > 0.0 && rows < n as f64, "{rows}");
    for index in [indexed, every] {
        assert!(b4_result(&file, index, "cull_ms") > 0.0, "{index}");
        assert!(b4_result(&file, index, "cull_ms_spread") >= 0.0, "{index}");
        assert_eq!(b4_result(&file, index, "overflow"), 0.0, "{index}");
    }
    // D-10: the index restricts the rows visited, never the image.
    assert_eq!(
        b4_result(&file, indexed, "visible_splats"),
        b4_result(&file, every, "visible_splats")
    );
    let spread =
        b4_result(&file, indexed, "cull_ms_spread").max(b4_result(&file, every, "cull_ms_spread"));
    assert!(
        b4_result(&file, indexed, "cull_ms") <= b4_result(&file, every, "cull_ms") + spread,
        "{}",
        file["results"]
    );
    let budgets = file["budgets"].as_array().unwrap();
    assert_eq!(budgets.len(), 2, "{budgets:?}");
    for b in budgets {
        assert_eq!(b["id"], "B-4");
        assert_eq!(b["setup"]["index"], indexed);
        assert_eq!(b["rule"], "at_most");
        assert_eq!(b["pass"], true, "{b}");
    }
}

#[test]
#[ignore = "gpu"]
fn gpu_b4_sweeps_every_size_with_both_checks_in_the_headsets_tier() {
    // §3.5 B-4 (Q-408, Q-T733.1, Q-406, Q-413): the synthetic sweep, run as the headset
    // runs it (`cmd_bench_on` with a fingerprint, `quest-balanced`), writes for each N
    // from 10⁵ to 3 × 10⁶ both setups and the two B-4 checks, all met, with the same
    // splats drawn by both (D-10).
    let _layer = ftgsxr_testkit::layer_messages(); // R-16, Q-105
    let dir = ftgsxr_testkit::scratch_dir(
        env!("CARGO_TARGET_TMPDIR"),
        crate::test_file_stem(),
        "b4-sweep",
    );
    let out = dir.join("B-4.json");
    ftgsxr_app::cmd_bench_on("B-4", Some(&out), None, Some("host")).expect("the sweep runs");
    let file: Value = serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    assert_eq!(file["machine"]["headset"], "host");
    let sizes = [1e5, 2.5e5, 5e5, 1e6, 2e6, 3e6];
    let results = file["results"].as_array().unwrap();
    let value = |n: f64, index: &str, measure: &str| -> Value {
        results
            .iter()
            .find(|r| {
                r["setup"]["n"] == n && r["setup"]["index"] == index && r["measure"] == measure
            })
            .unwrap_or_else(|| panic!("{n} {index} {measure}"))["value"]
            .clone()
    };
    for n in sizes {
        for index in ["temporal", "every row"] {
            assert!(
                value(n, index, "cull_ms").as_f64().unwrap() > 0.0,
                "{n} {index}"
            );
        }
        assert_eq!(value(n, "every row", "rows_visited").as_f64(), Some(n));
        assert_eq!(
            value(n, "temporal", "visible_splats"),
            value(n, "every row", "visible_splats"),
            "{n}"
        );
    }
    for r in results {
        assert_eq!(r["setup"]["tier"], "quest-balanced", "{r}");
    }
    let budgets = file["budgets"].as_array().unwrap();
    assert_eq!(budgets.len(), 2 * sizes.len(), "{budgets:?}");
    for b in budgets {
        assert_eq!(b["id"], "B-4");
        assert_eq!(b["pass"], true, "{b}");
    }
}

#[test]
#[ignore = "gpu"]
fn gpu_b5_with_a_model_reports_its_load_time() {
    let _layer = ftgsxr_testkit::layer_messages(); // R-16, Q-105
    let (path, _) = tiny_model("b5", 3000, 2);
    let bytes = std::fs::metadata(&path).unwrap().len();
    let file = run("B-5", Some(&path), "b5-out");
    let load_s = result(&file, "load_s");
    assert!(load_s > 0.0);
    let per_100mb = result(&file, "s_per_100mb");
    assert!((per_100mb - load_s * 1e8 / bytes as f64).abs() <= 1e-9 * per_100mb);
    assert_eq!(file["results"][0]["setup"]["file_bytes"], bytes);
    let budgets = file["budgets"].as_array().unwrap();
    assert_eq!(budgets.len(), 1);
    assert_eq!(budgets[0]["id"], "P-3");
    assert_eq!(budgets[0]["limit"], 1.0);
}

#[test]
#[ignore = "gpu"]
fn gpu_b6_matches_the_memory_table_within_1_percent() {
    let _layer = ftgsxr_testkit::layer_messages(); // R-16, Q-105
    // §1.12 Memory: float32 storage 76, 112, 172, 256 bytes for L = 0 … 3; half precision
    // 48, 66, 96, 138; each tier's SH cap (§1.12's table: desktop 3, quest-high 3,
    // quest-balanced 2, quest-low 1, conformance the model's L) gives the loaded L′.
    let float32 = [76.0, 112.0, 172.0, 256.0];
    let half = [48.0, 66.0, 96.0, 138.0];
    let expected = |tier: &str, l: usize| match tier {
        "conformance" | "desktop" => float32[l],
        "quest-high" => half[l],
        "quest-balanced" => half[l.min(2)],
        "quest-low" => half[l.min(1)],
        other => panic!("tier {other}"),
    };
    let file = run("B-6", None, "b6-out");
    let budgets = file["budgets"].as_array().unwrap();
    assert_eq!(budgets.len(), 20, "five tiers × four degrees");
    for b in budgets {
        let tier = b["setup"]["tier"].as_str().unwrap();
        let l = b["setup"]["sh_degree"].as_u64().unwrap() as usize;
        let want = expected(tier, l);
        let got = b["value"].as_f64().unwrap();
        assert!(
            (got / want - 1.0).abs() <= 0.01,
            "{tier} L={l}: {got} against {want}"
        );
        assert_eq!(b["limit"].as_f64().unwrap(), want, "{tier} L={l}");
        assert_eq!(b["pass"], true);
    }
}
