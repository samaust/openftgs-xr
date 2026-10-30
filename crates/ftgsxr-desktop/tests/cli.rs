// SPDX-License-Identifier: Apache-2.0

//! The binary `ftgsxr`'s headless subcommands without a GPU (§1.7, D-36, `cpu` lane):
//! `info [--json]` prints `cmd_info`; `validate`, `import` and `export` exit 0 on success
//! and 2 on an invalid input, printing §1.7's output (Q-77).

use std::fs;
use std::process::{Command, Output};

use ftgsxr_app::cmd_info;
use ftgsxr_testkit::{fixtures_dir, probe, scratch_dir};

fn ftgsxr(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ftgsxr"))
        .args(args)
        .output()
        .expect("the binary runs")
}

fn stdout(o: &Output) -> String {
    String::from_utf8(o.stdout.clone()).unwrap()
}

#[test]
fn info_prints_the_reply() {
    let version = env!("CARGO_PKG_VERSION");
    for (args, json) in [(&["info"][..], false), (&["info", "--json"][..], true)] {
        let o = ftgsxr(args);
        assert_eq!(o.status.code(), Some(0));
        assert_eq!(stdout(&o), format!("{}\n", cmd_info(version, json)));
    }
}

#[test]
fn validate_prints_one_line_per_model() {
    let p1 = fixtures_dir().join("probes/P1.safetensors");
    let ckpt = fixtures_dir().join("checkpoint/tiny");
    let o = ftgsxr(&["validate", p1.to_str().unwrap(), ckpt.to_str().unwrap()]);
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    let out = stdout(&o);
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines.len(), 2, "{out}");
    assert!(
        lines[0].starts_with(&format!("{}: kind ftgs-if, N ", p1.display())),
        "{out}"
    );
    assert!(
        lines[1].starts_with(&format!("{}: kind ftgs-checkpoint, N ", ckpt.display())),
        "{out}"
    );
    assert!(lines[1].ends_with(" bytes"), "{out}");
}

#[test]
fn an_invalid_input_exits_2() {
    let dir = scratch_dir(
        env!("CARGO_TARGET_TMPDIR"),
        env!("CARGO_CRATE_NAME"),
        "invalid",
    );
    let bad = dir.join("bad.safetensors");
    fs::write(&bad, b"not a safetensors file").unwrap();
    let out = dir.join("out.safetensors");
    let bad = bad.to_str().unwrap();
    for args in [
        vec!["validate", bad],
        vec!["import", bad, out.to_str().unwrap()],
        vec!["export", bad, out.to_str().unwrap()],
    ] {
        let o = ftgsxr(&args);
        assert_eq!(o.status.code(), Some(2), "{args:?}");
        assert!(o.stdout.is_empty(), "{args:?}");
        let err = String::from_utf8(o.stderr).unwrap();
        assert!(err.starts_with("ftgsxr: E-"), "{args:?}: {err}");
    }
    assert!(!out.exists());
}

#[test]
fn export_writes_the_file() {
    let dir = scratch_dir(
        env!("CARGO_TARGET_TMPDIR"),
        env!("CARGO_CRATE_NAME"),
        "export",
    );
    let input = fixtures_dir().join("probes/P1.safetensors");
    let out = dir.join("out.safetensors");
    for command in ["import", "export"] {
        let o = ftgsxr(&[command, input.to_str().unwrap(), out.to_str().unwrap()]);
        assert_eq!(o.status.code(), Some(0), "{command}");
        assert!(o.stdout.is_empty(), "{command}");
        assert_eq!(
            fs::read(&out).unwrap(),
            fs::read(&input).unwrap(),
            "{command}"
        );
    }
}

#[test]
fn a_usage_error_exits_2() {
    assert_eq!(ftgsxr(&["validate"]).status.code(), Some(2));
    assert_eq!(ftgsxr(&["frobnicate"]).status.code(), Some(2));
}

/// P1's request with `field`'s JSON value replaced by `value`, written into `dir`.
fn altered_p1_request(dir: &std::path::Path, field: &str, value: &str) -> std::path::PathBuf {
    let text = fs::read_to_string(fixtures_dir().join("probes/P1.request.json")).unwrap();
    let needle = format!("\"{field}\": ");
    let start = text.find(&needle).expect("the field") + needle.len();
    let end = start + text[start..].find([',', '}']).unwrap();
    let path = dir.join("request.json");
    fs::write(&path, format!("{}{value}{}", &text[..start], &text[end..])).unwrap();
    path
}

/// R-15 through the binary (§1.7): a request in another time unit and a request degree
/// above the model's exit 2 with their codes and write no file (Q-90).
#[test]
fn r15_render_refuses_a_request_with_exit_2() {
    let model = fixtures_dir().join("probes/P1.safetensors");
    for (name, field, value, code) in [
        ("unit", "time_unit", "\"ms\"", "E-REQ-UNIT"),
        ("degree", "sh_degree", "1", "E-REQ-DEGREE"),
    ] {
        let dir = scratch_dir(env!("CARGO_TARGET_TMPDIR"), env!("CARGO_CRATE_NAME"), name);
        let request = altered_p1_request(&dir, field, value);
        let out = dir.join("out.safetensors");
        let o = ftgsxr(&[
            "render",
            model.to_str().unwrap(),
            request.to_str().unwrap(),
            out.to_str().unwrap(),
        ]);
        assert_eq!(o.status.code(), Some(2), "{name}");
        let err = String::from_utf8(o.stderr).unwrap();
        assert!(
            err.starts_with(&format!("ftgsxr: {code}: ")),
            "{name}: {err}"
        );
        assert!(!out.exists(), "{name}");
    }
}

/// `--tier` takes the five names of §1.12 and refuses any other as a usage error (exit 2,
/// Q-89).
#[test]
fn render_tier_names() {
    let model = fixtures_dir().join("probes/P1.safetensors");
    let dir = scratch_dir(
        env!("CARGO_TARGET_TMPDIR"),
        env!("CARGO_CRATE_NAME"),
        "tier",
    );
    // A request the model refuses before the GPU, so each accepted tier exits 2 with
    // E-REQ-UNIT rather than clap's usage error.
    let request = altered_p1_request(&dir, "time_unit", "\"ms\"");
    let out = dir.join("out.safetensors");
    let args = |tier: &str| {
        ftgsxr(&[
            "render",
            model.to_str().unwrap(),
            request.to_str().unwrap(),
            out.to_str().unwrap(),
            "--tier",
            tier,
        ])
    };
    for tier in [
        "conformance",
        "desktop",
        "quest-high",
        "quest-balanced",
        "quest-low",
    ] {
        let err = String::from_utf8(args(tier).stderr).unwrap();
        assert!(err.starts_with("ftgsxr: E-REQ-UNIT: "), "{tier}: {err}");
    }
    let o = args("ultra");
    assert_eq!(o.status.code(), Some(2));
    assert!(!String::from_utf8(o.stderr).unwrap().contains("E-REQ-UNIT"));
}

/// The safetensors header of `bytes`, its JSON text.
fn header(bytes: &[u8]) -> &str {
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    std::str::from_utf8(&bytes[8..8 + n]).unwrap()
}

/// The value of the metadata key `key` in a safetensors header, a JSON string without
/// escapes (the strings `render` writes).
fn metadata<'a>(header: &'a str, key: &str) -> &'a str {
    let needle = format!("\"{key}\":\"");
    let start = header
        .find(&needle)
        .unwrap_or_else(|| panic!("no {key} in {header}"))
        + needle.len();
    &header[start..start + header[start..].find('"').unwrap()]
}

/// R-15 through the binary (§1.7, `gpu` lane): P1's request writes a `ftgs-renders` file
/// with openftgs's keys (`rgb/<id>`, `alpha/<id>` for every item) and metadata, the
/// request's SHA-256, prints nothing and exits 0.
#[test]
#[ignore = "gpu"]
fn gpu_r15_render_writes_the_renders_file() {
    let dir = scratch_dir(
        env!("CARGO_TARGET_TMPDIR"),
        env!("CARGO_CRATE_NAME"),
        "render",
    );
    let model = fixtures_dir().join("probes/P1.safetensors");
    let request_path = fixtures_dir().join("probes/P1.request.json");
    let out = dir.join("P1.renders.safetensors");
    let o = ftgsxr(&[
        "render",
        model.to_str().unwrap(),
        request_path.to_str().unwrap(),
        out.to_str().unwrap(),
    ]);
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    assert!(o.stdout.is_empty());
    // The committed request, read by the testkit (its items and its file's SHA-256).
    let (_, request) = probe("P1");
    let bytes = fs::read(&out).unwrap();
    let header = header(&bytes);
    for item in &request.items {
        for key in [format!("rgb/{}", item.id), format!("alpha/{}", item.id)] {
            assert!(header.contains(&format!("\"{key}\":{{")), "no {key}");
        }
    }
    assert_eq!(metadata(header, "format"), "ftgs-renders");
    assert_eq!(metadata(header, "version"), "1");
    let hex: String = request.sha256.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(metadata(header, "request_sha256"), hex);
    let producer = metadata(header, "producer");
    let version = env!("CARGO_PKG_VERSION");
    assert!(
        producer.starts_with(&format!("openftgs-xr {version} (")) && producer.ends_with(')'),
        "{producer}"
    );
}

#[test]
fn bench_refuses_a_name_the_desktop_does_not_run() {
    // Q-T122.2: clap accepts B-1, B-3, B-5 and B-6 only; a usage error exits 2.
    let o = ftgsxr(&["bench", "B-2"]);
    assert_eq!(o.status.code(), Some(2));
}

/// `ftgsxr bench` with the baselines' directory `baselines` (Q-T122.2).
fn bench(args: &[&str], baselines: &std::path::Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ftgsxr"))
        .arg("bench")
        .args(args)
        .env("FTGSXR_BENCH_BASELINES", baselines)
        .output()
        .expect("the binary runs")
}

#[test]
#[ignore = "gpu"]
fn gpu_bench_reports_a_regression_against_a_baseline_20_percent_better() {
    // Q-67's detecting test: the first run, with no baseline, exits 0; a baseline whose
    // budgeted figures are 20 % better (B-6's bytes per primitive × 0.8: lower is
    // better) makes the second run list each budget as a regression and exit 3.
    let dir = scratch_dir(env!("CARGO_TARGET_TMPDIR"), "cli", "bench-regression");
    let baselines = dir.join("baselines");
    fs::create_dir_all(&baselines).unwrap();
    let model = dir.join("tiny.ftgs-if");
    let tiny = ftgsxr_testkit::random_model(5, 1000, 3);
    fs::write(&model, ftgsxr_testkit::exchange_bytes(&tiny)).unwrap();
    let first = dir.join("first.json");
    let args = ["B-6", "--model", model.to_str().unwrap(), "--out"];

    let o = bench(
        &[&args[..], &[first.to_str().unwrap()]].concat(),
        &baselines,
    );
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    assert!(stdout(&o).is_empty(), "the file goes to --out");
    let mut file: serde_json::Value = serde_json::from_slice(&fs::read(&first).unwrap()).unwrap();
    assert_eq!(file["baseline"], serde_json::Value::Null);
    let budgets = file["budgets"].as_array_mut().unwrap();
    assert_eq!(budgets.len(), 5, "five tiers");
    for b in budgets.iter_mut() {
        let better = b["value"].as_f64().unwrap() * 0.8;
        b["value"] = better.into();
    }
    fs::write(
        baselines.join("B-6.json"),
        serde_json::to_vec_pretty(&file).unwrap(),
    )
    .unwrap();

    let second = dir.join("second.json");
    let o = bench(
        &[&args[..], &[second.to_str().unwrap()]].concat(),
        &baselines,
    );
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert_eq!(o.status.code(), Some(3), "{stderr}");
    let regressions: Vec<&str> = stderr
        .lines()
        .filter(|l| l.starts_with("ftgsxr: regression: B-6 bytes_per_primitive"))
        .collect();
    assert_eq!(regressions.len(), 5, "{stderr}");
    let written: serde_json::Value = serde_json::from_slice(&fs::read(&second).unwrap()).unwrap();
    assert!(
        written["baseline"]
            .as_str()
            .is_some_and(|p| p.ends_with("B-6.json"))
    );
}
