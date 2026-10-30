// SPDX-License-Identifier: Apache-2.0

//! The desktop benchmarks of §3.5 through `cmd_bench` (§1.7 `bench`, Q-67, Q-T122.1,
//! Q-T122.2): the NAME check (`cpu` lane) and, on a tiny model given with `--model`, the
//! file each benchmark writes (`gpu` lane). These runs are not recorded timings: the
//! baselines come from the full sweeps run alone on the machine.

use std::path::{Path, PathBuf};

use ftgsxr_app::{HeadlessError, cmd_bench, exit_code};
use serde_json::Value;

#[test]
fn bench_refuses_a_benchmark_the_desktop_does_not_run() {
    // §3.5: B-2, B-4, B-7, B-8 and B-9 are the Quest's or reported elsewhere (Q-T122.2).
    for name in ["B-2", "B-4", "B-7", "B-9", "B-10", "b-1", ""] {
        let e = cmd_bench(name, None, None).expect_err(name);
        assert!(matches!(e, HeadlessError::Input(_)), "{name}: {e:?}");
        assert_eq!(exit_code(&e), 2);
    }
}

#[test]
fn bench_refuses_an_invalid_model_before_the_gpu() {
    let dir = ftgsxr_testkit::scratch_dir(env!("CARGO_TARGET_TMPDIR"), "b_benchmarks", "invalid");
    let path = dir.join("broken.ftgs-if");
    std::fs::write(&path, b"not a safetensors file").unwrap();
    let e = cmd_bench("B-6", None, Some(&path)).expect_err("an invalid model");
    assert!(matches!(e, HeadlessError::Input(_)), "{e:?}");
    assert!(e.code().starts_with("E-FILE"), "{e}");
}

/// A tiny model of `n` rows at `degree`, written as a `ftgs-if` file in its own scratch
/// directory (`--model`'s input).
fn tiny_model(name: &str, n: usize, degree: u8) -> (PathBuf, ftgsxr_core::Model) {
    let dir = ftgsxr_testkit::scratch_dir(env!("CARGO_TARGET_TMPDIR"), "b_benchmarks", name);
    let model = ftgsxr_testkit::random_model(17, n, degree);
    let path = dir.join("tiny.ftgs-if");
    std::fs::write(&path, ftgsxr_testkit::exchange_bytes(&model)).unwrap();
    (path, model)
}

/// Runs `name` on `model` with `--out`, checks the report is the file and Q-T122.2's
/// top level, and returns the parsed file.
fn run(name: &str, model: Option<&Path>, dir_name: &str) -> Value {
    let dir = ftgsxr_testkit::scratch_dir(env!("CARGO_TARGET_TMPDIR"), "b_benchmarks", dir_name);
    let out = dir.join("bench.json");
    let report = cmd_bench(name, Some(&out), model).expect("the benchmark runs");
    let text = std::fs::read_to_string(&out).expect("the file is written");
    assert_eq!(report.json, text);
    let file: Value = serde_json::from_str(&text).expect("well-formed JSON");
    assert_eq!(file["format"], "ftgsxr-bench");
    assert_eq!(file["version"], 1);
    assert_eq!(file["bench"], name);
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
    // Q-67's first detecting test: B-1 writes a well-formed file; with `--model` its one
    // setup carries P-2's budget (Q-T122.1).
    let (path, model) = tiny_model("b1", 2000, 2);
    let file = run("B-1", Some(&path), "b1-out");
    let identity: String = model.identity.iter().map(|b| format!("{b:02x}")).collect();
    let first = &file["results"][0];
    assert_eq!(first["setup"]["model"], identity.as_str());
    assert_eq!(first["setup"]["n"], 2000);
    assert_eq!(first["setup"]["width"], 1920);
    assert_eq!(first["setup"]["height"], 1080);
    assert!(result(&file, "fps_min") > 0.0);
    assert!(result(&file, "fps_mean") >= result(&file, "fps_min"));
    assert!(result(&file, "visible_splats") <= 2000.0);
    let budgets = file["budgets"].as_array().unwrap();
    assert_eq!(budgets.len(), 1);
    assert_eq!(budgets[0]["id"], "P-2");
    assert_eq!(budgets[0]["limit"], 90);
}

#[test]
#[ignore = "gpu"]
fn gpu_b3_with_a_model_reports_sort_times() {
    let (path, _) = tiny_model("b3", 5000, 0);
    let file = run("B-3", Some(&path), "b3-out");
    assert!(result(&file, "sort_ms") > 0.0);
    let keys = result(&file, "keys");
    assert!(keys > 0.0 && keys <= 5000.0, "{keys}");
    // P-4 is defined at C = 10⁶ only (Q-T122.1).
    assert!(file["budgets"].as_array().unwrap().is_empty());
}

#[test]
#[ignore = "gpu"]
fn gpu_b5_with_a_model_reports_its_load_time() {
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
