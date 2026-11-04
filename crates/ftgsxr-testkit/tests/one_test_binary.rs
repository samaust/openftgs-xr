// SPDX-License-Identifier: Apache-2.0

//! #413 (§2.7 Organisation): each crate's integration tests build as one test binary, the
//! target `integration` at `tests/integration.rs`, whose modules are the crate's other
//! `tests/*.rs` files, so that each new test file adds a module, not another executable
//! that links bevy and wgpu statically. A plain-text scan of every crate's manifest and
//! test root, with no exception list.

use std::fs;
use std::path::Path;

/// The one integration-test target's name, the same in every crate.
const TARGET: &str = "integration";

/// A line without its `//` comment, trimmed.
fn code(line: &str) -> &str {
    line.split("//").next().unwrap_or("").trim()
}

/// What is wrong with one crate's integration tests: `manifest` is its `Cargo.toml`, `root`
/// its `tests/integration.rs` if it has one, and `stems` the stems of its other
/// `tests/*.rs` files.
fn problems(manifest: &str, root: Option<&str>, stems: &[String]) -> Vec<String> {
    let lines: Vec<&str> = manifest.lines().map(code).collect();
    let targets = lines.iter().filter(|l| **l == "[[test]]").count();
    let mut found = Vec::new();
    if targets > 1 {
        found.push(format!("{targets} integration-test targets, not one"));
    }
    if stems.is_empty() && root.is_none() && targets == 0 {
        return found;
    }
    if !lines
        .iter()
        .any(|l| l.replace(' ', "") == "autotests=false")
    {
        found.push("no `autotests = false`, so each tests/*.rs file is a target".into());
    }
    // The `[[test]]` table's own lines, up to the next table.
    let table: Vec<&str> = lines
        .iter()
        .skip_while(|l| **l != "[[test]]")
        .skip(1)
        .take_while(|l| !l.starts_with('['))
        .copied()
        .collect();
    let declared = |key: &str, value: &str| {
        table
            .iter()
            .any(|l| l.replace(' ', "") == format!("{key}=\"{value}\""))
    };
    if targets == 0 || !declared("name", TARGET) || !declared("path", "tests/integration.rs") {
        found.push(format!(
            "no `[[test]]` named `{TARGET}` at `tests/integration.rs`"
        ));
    }
    let Some(root) = root else {
        found.push("no tests/integration.rs".into());
        return found;
    };
    let modules: Vec<&str> = root
        .lines()
        .map(code)
        .filter_map(|l| l.strip_prefix("mod ")?.strip_suffix(';'))
        .collect();
    for stem in stems {
        if !modules.contains(&stem.as_str()) {
            found.push(format!(
                "tests/{stem}.rs is not a module of the test binary (`mod {stem};` in tests/integration.rs)"
            ));
        }
    }
    found
}

#[test]
fn every_crate_has_one_test_binary_holding_all_its_test_files() {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut offending = Vec::new();
    let mut checked = 0;
    for entry in fs::read_dir(&crates).unwrap() {
        let dir = entry.unwrap().path();
        let Ok(manifest) = fs::read_to_string(dir.join("Cargo.toml")) else {
            continue;
        };
        let tests = dir.join("tests");
        let mut stems = Vec::new();
        if tests.is_dir() {
            for file in fs::read_dir(&tests).unwrap() {
                let path = file.unwrap().path();
                if path.is_file() && path.extension().is_some_and(|e| e == "rs") {
                    let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
                    if stem != TARGET {
                        stems.push(stem);
                    }
                }
            }
        }
        stems.sort();
        let root = fs::read_to_string(tests.join("integration.rs")).ok();
        checked += usize::from(!stems.is_empty());
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        for problem in problems(&manifest, root.as_deref(), &stems) {
            offending.push(format!("{name}: {problem}"));
        }
    }
    assert!(checked > 0, "no test files under {}", crates.display());
    assert!(
        offending.is_empty(),
        "crates whose integration tests are not one test binary (#413):\n{}",
        offending.join("\n")
    );
}

#[test]
fn the_scan_flags_an_undeclared_file_and_a_second_target() {
    let manifest = r#"
[package]
name = "sample"
autotests = false

[[test]]
name = "integration"
path = "tests/integration.rs"
"#;
    let root = "mod support;\nmod a_test;\n// mod b_test;\nmod c_test;\n";
    let stems = |s: &[&str]| s.iter().map(|s| s.to_string()).collect::<Vec<_>>();

    // Every file declared: nothing to report.
    assert!(problems(manifest, Some(root), &stems(&["a_test", "c_test"])).is_empty());
    // A file the root doesn't declare, or declares only in a comment.
    let found = problems(
        manifest,
        Some(root),
        &stems(&["a_test", "b_test", "d_test"]),
    );
    assert_eq!(found.len(), 2, "{found:#?}");
    assert!(found[0].contains("tests/b_test.rs") && found[1].contains("tests/d_test.rs"));
    // A second target.
    let two = format!("{manifest}\n[[test]]\nname = \"other\"\npath = \"tests/other.rs\"\n");
    let found = problems(&two, Some(root), &stems(&["a_test"]));
    assert!(
        found
            .iter()
            .any(|p| p.contains("2 integration-test targets")),
        "{found:#?}"
    );
    // Automatic discovery left on, with or without the target.
    let auto = manifest.replace("autotests = false\n", "");
    assert!(
        problems(&auto, Some(root), &stems(&["a_test"]))
            .iter()
            .any(|p| p.contains("autotests"))
    );
    let bare = "[package]\nname = \"sample\"\n";
    let found = problems(bare, None, &stems(&["a_test"]));
    assert_eq!(found.len(), 3, "{found:#?}");
    // The target's name and path count only inside its own table.
    let elsewhere = "[package]\nname = \"integration\"\nautotests = false\n\n[[test]]\nname = \"other\"\n\n[[bin]]\npath = \"tests/integration.rs\"\n";
    assert!(
        problems(elsewhere, Some(root), &stems(&["a_test"]))
            .iter()
            .any(|p| p.contains("no `[[test]]` named"))
    );
    // A crate without integration tests needs nothing.
    assert!(problems(bare, None, &[]).is_empty());
}
