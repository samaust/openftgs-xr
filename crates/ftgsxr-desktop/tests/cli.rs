// SPDX-License-Identifier: Apache-2.0

//! The binary `ftgsxr`'s headless subcommands without a GPU (§1.7, D-36, `cpu` lane):
//! `info [--json]` prints `cmd_info`; `validate`, `import` and `export` exit 0 on success
//! and 2 on an invalid input, printing §1.7's output (Q-77).

use std::fs;
use std::process::{Command, Output};

use ftgsxr_app::cmd_info;
use ftgsxr_testkit::{fixtures_dir, scratch_dir};

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
