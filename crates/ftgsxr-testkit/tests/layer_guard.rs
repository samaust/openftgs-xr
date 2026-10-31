// SPDX-License-Identifier: Apache-2.0

//! R-16 (§1.13, Q-105, #176): every `gpu_` and `ds_` test of the workspace's
//! `crates/*/tests/` makes [`ftgsxr_testkit::layer_messages`]'s guard as its first
//! statement, bound to a name, so a test that forgets it can't pass whatever the validation
//! layer reports. A plain-text scan of the test files; helpers are not checked.

use std::fs;
use std::path::{Path, PathBuf};

/// Tests that check the layer another way, as `<crate>/tests/<file>::<function>` (Q-113):
/// the binary's runs, R-15's and `bench`'s, read the layer's own log file (Q-105).
const EXEMPT: &[&str] = &[
    "ftgsxr-desktop/tests/cli.rs::gpu_r15_render_writes_the_renders_file",
    "ftgsxr-desktop/tests/cli.rs::gpu_bench_reports_a_regression_against_a_baseline_20_percent_better",
    "ftgsxr-desktop/tests/cli.rs::gpu_bench_compares_neither_b6_nor_another_profile",
];

/// This file, whose sample text declares unguarded tests.
const THIS_FILE: &str = "ftgsxr-testkit/tests/layer_guard.rs";

/// The `.rs` files under `dir`, at any depth.
fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// The name of the function a line declares, if it does.
fn declared_fn(line: &str) -> Option<&str> {
    let rest = line.trim_start().strip_prefix("fn ")?;
    let end = rest
        .find(|c: char| !(c.is_alphanumeric() || c == '_'))
        .unwrap_or(rest.len());
    Some(&rest[..end])
}

/// Whether a code line binds the guard to a name: `let <name> = [path::]layer_messages();`
/// with `<name>` other than `_`, which would drop the guard at once.
fn binds_the_guard(line: &str) -> bool {
    let Some(rest) = line.trim().strip_prefix("let ") else {
        return false;
    };
    let Some((name, call)) = rest.split_once('=') else {
        return false;
    };
    let name = name.trim();
    let call = call.trim();
    let named =
        !name.is_empty() && name != "_" && name.chars().all(|c| c.is_alphanumeric() || c == '_');
    let call = call.strip_prefix("ftgsxr_testkit::").unwrap_or(call);
    named && call.starts_with("layer_messages();")
}

/// The `gpu_` and `ds_` test functions of one file's text whose first statement is not the
/// guard.
fn unguarded(text: &str) -> Vec<String> {
    let lines: Vec<&str> = text.lines().collect();
    let mut found = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let Some(name) = declared_fn(line) else {
            continue;
        };
        if !(name.starts_with("gpu_") || name.starts_with("ds_")) {
            continue;
        }
        // The attributes stacked above the function, `#[test]` among them.
        let is_test = lines[..i]
            .iter()
            .rev()
            .map(|l| l.trim())
            .take_while(|l| l.starts_with("#["))
            .any(|l| l == "#[test]");
        if !is_test {
            continue;
        }
        // The first code line after the body's opening brace, past blank and comment lines.
        let open = (i..lines.len()).find(|&j| lines[j].trim_end().ends_with('{'));
        let first = open.and_then(|o| {
            lines[o + 1..]
                .iter()
                .map(|l| l.trim())
                .find(|l| !l.is_empty() && !l.starts_with("//"))
        });
        if !first.is_some_and(binds_the_guard) {
            found.push(name.to_string());
        }
    }
    found
}

#[test]
fn every_gpu_and_ds_test_holds_the_layer_guard() {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut files = Vec::new();
    for entry in fs::read_dir(&crates).unwrap() {
        let tests = entry.unwrap().path().join("tests");
        if tests.is_dir() {
            rust_files(&tests, &mut files);
        }
    }
    files.sort();
    assert!(
        !files.is_empty(),
        "no test files under {}",
        crates.display()
    );

    let mut offending = Vec::new();
    let mut exempt_seen = Vec::new();
    for file in &files {
        let relative = file
            .strip_prefix(&crates)
            .unwrap()
            .to_string_lossy()
            .into_owned();
        if relative == THIS_FILE {
            continue;
        }
        for name in unguarded(&fs::read_to_string(file).unwrap()) {
            let id = format!("{relative}::{name}");
            if EXEMPT.contains(&id.as_str()) {
                exempt_seen.push(id);
            } else {
                offending.push(format!("{relative}:{name}"));
            }
        }
    }
    assert!(
        offending.is_empty(),
        "gpu_ and ds_ tests without `let _layer = layer_messages();` as their first \
         statement (R-16, Q-105):\n{}",
        offending.join("\n")
    );
    // An exemption whose test is gone or now guarded is stale.
    for id in EXEMPT {
        assert!(exempt_seen.iter().any(|s| s == id), "stale exemption {id}");
    }
}

#[test]
fn the_scan_flags_a_missing_or_discarded_guard_and_skips_helpers() {
    let text = r#"
#[test]
#[ignore = "gpu"]
fn gpu_guarded() {
    // R-16
    let _layer = layer_messages(); // Q-105
}

#[test]
fn ds_named_guard() {
    let layer = ftgsxr_testkit::layer_messages();
}

#[test]
#[ignore = "gpu"]
fn gpu_missing() {
    let device = gpu_context();
    let _layer = layer_messages();
}

#[test]
fn gpu_discarded() {
    let _ = layer_messages();
}

#[test]
fn ds_empty() {
}

fn gpu_helper() {
    let device = gpu_context();
}

#[test]
fn cpu_test() {
    let x = 1;
}
"#;
    assert_eq!(
        unguarded(text),
        ["gpu_missing", "gpu_discarded", "ds_empty"]
    );
}
