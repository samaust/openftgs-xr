// SPDX-License-Identifier: Apache-2.0

//! The binary `ftgsxr`'s headless subcommands without a GPU (§1.7, D-36, `cpu` lane):
//! `info [--json]` prints `cmd_info`; `validate`, `import` and `export` exit 0 on success
//! and 2 on an invalid input, printing §1.7's output (Q-77).

use std::fs;
use std::path::{Path, PathBuf};
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
fn validate_prints_each_valid_model_before_the_first_invalid_one() {
    // §1.7: each model is read and validated and its line printed in turn; the first
    // invalid file stops the run with exit 2, so the models after it are not printed.
    let dir = scratch_dir(
        env!("CARGO_TARGET_TMPDIR"),
        env!("CARGO_CRATE_NAME"),
        "validate_order",
    );
    let bad = dir.join("bad.safetensors");
    fs::write(&bad, b"not a safetensors file").unwrap();
    let p1 = fixtures_dir().join("probes/P1.safetensors");
    let ckpt = fixtures_dir().join("checkpoint/tiny");
    let o = ftgsxr(&[
        "validate",
        p1.to_str().unwrap(),
        bad.to_str().unwrap(),
        ckpt.to_str().unwrap(),
    ]);
    assert_eq!(o.status.code(), Some(2));
    let out = stdout(&o);
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines.len(), 1, "{out}");
    assert!(
        lines[0].starts_with(&format!("{}: kind ftgs-if, N ", p1.display())),
        "{out}"
    );
    let err = String::from_utf8(o.stderr).unwrap();
    assert!(err.starts_with("ftgsxr: E-"), "{err}");
    assert!(err.contains(bad.to_str().unwrap()), "{err}");
}

#[test]
fn validate_warns_of_a_convention_rendered_as_openftgss() {
    // E6, D-39: a `rasterize_mode` other than `classic` is reported as a warning (on
    // standard error, Q-110) and the model still validates. P1's header with its
    // `"classic"` replaced by a word of the same length, so the header keeps its size.
    let dir = scratch_dir(
        env!("CARGO_TARGET_TMPDIR"),
        env!("CARGO_CRATE_NAME"),
        "validate_warning",
    );
    let mut bytes = fs::read(fixtures_dir().join("probes/P1.safetensors")).unwrap();
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let at = bytes[8..8 + n]
        .windows(7)
        .position(|w| w == b"classic")
        .expect("P1's rasterize_mode")
        + 8;
    bytes[at..at + 7].copy_from_slice(b"stochas");
    let model = dir.join("stochas.safetensors");
    fs::write(&model, bytes).unwrap();
    let o = ftgsxr(&["validate", model.to_str().unwrap()]);
    let err = String::from_utf8(o.stderr.clone()).unwrap();
    assert_eq!(o.status.code(), Some(0), "{err}");
    assert_eq!(stdout(&o).lines().count(), 1);
    assert_eq!(
        err,
        format!(
            "ftgsxr: warning: {}: rasterize_mode \"stochas\" is not supported; rendering as \"classic\"\n",
            model.display()
        )
    );
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

/// A Khronos validation layer settings file in `dir` that sends the layer's errors,
/// warnings (performance ones too) and information to `log`; the directory holding it, the
/// one `VK_LAYER_SETTINGS_PATH` names, is returned. The layer writes its log even beside
/// wgpu's own messenger.
fn layer_settings(dir: &Path, log: &Path) -> PathBuf {
    let settings = dir.join("layer-settings");
    fs::create_dir_all(&settings).unwrap();
    let _ = fs::remove_file(log);
    fs::write(
        settings.join("vk_layer_settings.txt"),
        format!(
            "khronos_validation.debug_action = VK_DBG_LAYER_ACTION_LOG_MSG\n\
             khronos_validation.report_flags = error,warn,perf,info\n\
             khronos_validation.log_filename = {}\n",
            log.display()
        ),
    )
    .unwrap();
    settings
}

/// The layer's log: whether it reported itself active (its status message at instance
/// creation) and the header line of each error or warning, the lines that read
/// `<message id>(ERROR / <type>): …` or `<message id>(WARNING / <type>): …`.
fn layer_log(log: &Path) -> (bool, Vec<String>) {
    let text = fs::read_to_string(log).unwrap_or_default();
    let active = text.contains("Khronos Validation Layer Active");
    let messages = text
        .lines()
        .filter(|line| {
            line.split_once('(').is_some_and(|(id, rest)| {
                !id.contains(' ')
                    && (rest.starts_with("ERROR / ") || rest.starts_with("WARNING / "))
            })
        })
        .map(str::to_owned)
        .collect();
    (active, messages)
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
    // R-16 (Q-105): the binary has no logger, so wgpu-hal drops the layer's messages;
    // the layer's own log, chosen by a settings file, keeps them.
    let log = dir.join("validation.log");
    let settings = layer_settings(&dir, &log);
    let o = Command::new(env!("CARGO_BIN_EXE_ftgsxr"))
        .args([
            "render",
            model.to_str().unwrap(),
            request_path.to_str().unwrap(),
            out.to_str().unwrap(),
        ])
        .env("VK_LAYER_SETTINGS_PATH", &settings)
        .output()
        .expect("the binary runs");
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    assert!(o.stdout.is_empty());
    // §1.13: a debug binary has the layers on (`InstanceFlags::from_build_config`), and
    // they reported nothing.
    let (active, messages) = layer_log(&log);
    if cfg!(debug_assertions) {
        assert!(
            active,
            "the Khronos validation layer did not report itself active"
        );
    }
    assert!(messages.is_empty(), "{messages:#?}");
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
    // Q-103: clap accepts B-1, B-3, B-5 and B-6 only; a usage error exits 2.
    let o = ftgsxr(&["bench", "B-2"]);
    assert_eq!(o.status.code(), Some(2));
}

/// `ftgsxr bench` with the baselines' directory `baselines` (Q-103), under the validation
/// layer: as R-15's run, the layer's own log (a settings file beside `baselines`) has no
/// error or warning and, in a debug build, its status message (R-16, Q-105).
fn bench(args: &[&str], baselines: &std::path::Path) -> Output {
    let dir = baselines
        .parent()
        .expect("the baselines' directory has a parent");
    let log = dir.join("validation.log");
    let settings = layer_settings(dir, &log);
    let o = Command::new(env!("CARGO_BIN_EXE_ftgsxr"))
        .arg("bench")
        .args(args)
        .env("FTGSXR_BENCH_BASELINES", baselines)
        .env("VK_LAYER_SETTINGS_PATH", &settings)
        .output()
        .expect("the binary runs");
    let (active, messages) = layer_log(&log);
    if cfg!(debug_assertions) {
        assert!(
            active,
            "the Khronos validation layer did not report itself active"
        );
    }
    assert!(messages.is_empty(), "{messages:#?}");
    o
}

/// `bench` with `args` ending in `--out` followed by `out`, and the file it wrote, or
/// null when it wrote none.
fn bench_file(
    args: &[&str],
    out: &std::path::Path,
    baselines: &std::path::Path,
) -> (Output, serde_json::Value) {
    let o = bench(&[args, &[out.to_str().unwrap()]].concat(), baselines);
    let file = fs::read(out)
        .map(|bytes| serde_json::from_slice(&bytes).unwrap())
        .unwrap_or(serde_json::Value::Null);
    (o, file)
}

#[test]
#[ignore = "gpu"]
fn gpu_bench_reports_a_regression_against_a_baseline_20_percent_better() {
    // Q-67's detecting test: the first run, with no baseline, exits 0; a baseline whose
    // budgeted figure is 20 % better (B-5's seconds per 100 MB × 0.8: lower is better)
    // makes the second run list the budget as a regression and exit 3. B-5, not B-6,
    // whose `within_1_percent` budgets are not compared (Q-103, #180).
    let dir = scratch_dir(env!("CARGO_TARGET_TMPDIR"), "cli", "bench-regression");
    let baselines = dir.join("baselines");
    fs::create_dir_all(&baselines).unwrap();
    // A load time is noisy: 1000 rows missed the 10 % margin about once in 15 runs,
    // 50 000 rows (a file of about 13 MB) none in 20.
    let model = dir.join("b5.ftgs-if");
    let rows = ftgsxr_testkit::random_model(5, 50_000, 3);
    fs::write(&model, ftgsxr_testkit::exchange_bytes(&rows)).unwrap();
    let args = ["B-5", "--model", model.to_str().unwrap(), "--out"];

    let (o, mut file) = bench_file(&args, &dir.join("first.json"), &baselines);
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    assert!(stdout(&o).is_empty(), "the file goes to --out");
    assert_eq!(file["baseline"], serde_json::Value::Null);
    let budgets = file["budgets"].as_array_mut().unwrap();
    assert_eq!(budgets.len(), 1, "one model");
    for b in budgets.iter_mut() {
        let better = b["value"].as_f64().unwrap() * 0.8;
        b["value"] = better.into();
    }
    fs::write(
        baselines.join("B-5.json"),
        serde_json::to_vec_pretty(&file).unwrap(),
    )
    .unwrap();

    let (o, written) = bench_file(&args, &dir.join("second.json"), &baselines);
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert_eq!(o.status.code(), Some(3), "{stderr}");
    let regressions: Vec<&str> = stderr
        .lines()
        .filter(|l| l.starts_with("ftgsxr: regression: P-3 s_per_100mb"))
        .collect();
    assert_eq!(regressions.len(), 1, "{stderr}");
    assert!(
        written["baseline"]
            .as_str()
            .is_some_and(|p| p.ends_with("B-5.json"))
    );
}

#[test]
#[ignore = "gpu"]
fn gpu_bench_compares_neither_b6_nor_another_profile() {
    // Q-103 (#180): a B-6 baseline 20 % better lists nothing, its `within_1_percent`
    // budgets being their own check; a baseline of the other profile is not compared,
    // and a note in the file and on standard error says which field differs.
    let dir = scratch_dir(env!("CARGO_TARGET_TMPDIR"), "cli", "bench-not-compared");
    let baselines = dir.join("baselines");
    fs::create_dir_all(&baselines).unwrap();
    let model = dir.join("tiny.ftgs-if");
    let tiny = ftgsxr_testkit::random_model(5, 1000, 3);
    fs::write(&model, ftgsxr_testkit::exchange_bytes(&tiny)).unwrap();
    let args = ["B-6", "--model", model.to_str().unwrap(), "--out"];

    let (o, mut file) = bench_file(&args, &dir.join("first.json"), &baselines);
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    for b in file["budgets"].as_array_mut().unwrap() {
        let better = b["value"].as_f64().unwrap() * 0.8;
        b["value"] = better.into();
    }
    let baseline = baselines.join("B-6.json");
    fs::write(&baseline, serde_json::to_vec_pretty(&file).unwrap()).unwrap();
    let (o, written) = bench_file(&args, &dir.join("second.json"), &baselines);
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert_eq!(o.status.code(), Some(0), "{stderr}");
    assert!(!stderr.contains("ftgsxr: regression:"), "{stderr}");
    assert!(written["baseline"].is_string());

    // The same baseline from the other profile.
    let (ours, theirs) = if cfg!(debug_assertions) {
        ("debug", "release")
    } else {
        ("release", "debug")
    };
    file["profile"] = theirs.into();
    fs::write(&baseline, serde_json::to_vec_pretty(&file).unwrap()).unwrap();
    let (o, written) = bench_file(&args, &dir.join("third.json"), &baselines);
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert_eq!(o.status.code(), Some(0), "{stderr}");
    let note = format!("baseline not compared: profile {ours} against {theirs} (Q-103)");
    assert!(stderr.contains(&note), "{stderr}");
    assert!(
        written["notes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|n| n.as_str() == Some(note.as_str())),
        "{}",
        written["notes"]
    );
}
