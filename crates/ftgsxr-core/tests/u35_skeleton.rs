// SPDX-License-Identifier: Apache-2.0

//! U-35, the always-run skeleton test (§2.7, `cpu` lane): every source file of the
//! workspace starts with its SPDX header; `cargo metadata` lists exactly §2.1's crates,
//! with the binary `ftgsxr` and the cdylib `ftgsxr_quest` (D-36); the crates depend on
//! one another only as §2.1's band table allows (Q-6); every shader file is referenced
//! by `ftgsxr-render`. Run alone with `cargo test -p ftgsxr-core skeleton`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

/// `cargo metadata --no-deps` for the workspace holding this crate.
fn metadata() -> Value {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| env!("CARGO").into());
    let out = Command::new(cargo)
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("cargo metadata runs");
    assert!(
        out.status.success(),
        "cargo metadata failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).expect("cargo metadata prints JSON")
}

fn workspace_root() -> PathBuf {
    PathBuf::from(
        metadata()["workspace_root"]
            .as_str()
            .expect("metadata has a workspace_root"),
    )
}

fn packages(meta: &Value) -> &Vec<Value> {
    meta["packages"].as_array().expect("metadata has packages")
}

fn str_of<'a>(v: &'a Value, key: &str) -> &'a str {
    v[key].as_str().unwrap_or_default()
}

/// The crates of §2.1's band table, each with its library or binary target: (kind, name).
fn expected_crates() -> BTreeMap<&'static str, (&'static str, &'static str)> {
    BTreeMap::from([
        ("ftgsxr-core", ("lib", "ftgsxr_core")),
        ("ftgsxr-io", ("lib", "ftgsxr_io")),
        ("ftgsxr-render", ("lib", "ftgsxr_render")),
        ("ftgsxr-app", ("lib", "ftgsxr_app")),
        ("ftgsxr-xr", ("lib", "ftgsxr_xr")),
        ("ftgsxr-desktop", ("bin", "ftgsxr")),
        ("ftgsxr-quest", ("cdylib", "ftgsxr_quest")),
        ("ftgsxr-testkit", ("lib", "ftgsxr_testkit")),
    ])
}

#[test]
fn u35_skeleton_crates() {
    let meta = metadata();
    let expected = expected_crates();
    let names: BTreeSet<&str> = packages(&meta).iter().map(|p| str_of(p, "name")).collect();
    let want: BTreeSet<&str> = expected.keys().copied().collect();
    assert_eq!(names, want, "the workspace's crates are exactly §2.1's");

    for p in packages(&meta) {
        let name = str_of(p, "name");
        // The library or binary targets only; tests, benches and examples are not the crate's kind.
        let mains: BTreeSet<(String, String)> = p["targets"]
            .as_array()
            .expect("a package has targets")
            .iter()
            .flat_map(|t| {
                let tname = str_of(t, "name").to_owned();
                t["kind"]
                    .as_array()
                    .expect("a target has kinds")
                    .iter()
                    .filter_map(|k| k.as_str())
                    .filter(|k| {
                        matches!(
                            *k,
                            "lib" | "rlib" | "bin" | "cdylib" | "dylib" | "staticlib"
                        )
                    })
                    .map(move |k| (k.to_owned(), tname.clone()))
                    .collect::<Vec<_>>()
            })
            .collect();
        let (kind, tname) = expected[name];
        let want = BTreeSet::from([(kind.to_owned(), tname.to_owned())]);
        assert_eq!(mains, want, "{name}'s kind and target name (§2.1, D-36)");
    }
}

/// The workspace crates each crate may depend on: §2.1's "Depends on" column. Any crate
/// may take `ftgsxr-testkit` as a dev-dependency, and nothing may take it otherwise.
fn allowed_dependencies() -> BTreeMap<&'static str, BTreeSet<&'static str>> {
    BTreeMap::from([
        ("ftgsxr-core", BTreeSet::from([])),
        ("ftgsxr-io", BTreeSet::from(["ftgsxr-core"])),
        (
            "ftgsxr-render",
            BTreeSet::from(["ftgsxr-core", "ftgsxr-io"]),
        ),
        (
            "ftgsxr-app",
            BTreeSet::from(["ftgsxr-core", "ftgsxr-io", "ftgsxr-render"]),
        ),
        (
            "ftgsxr-xr",
            BTreeSet::from(["ftgsxr-core", "ftgsxr-io", "ftgsxr-render", "ftgsxr-app"]),
        ),
        (
            "ftgsxr-desktop",
            BTreeSet::from(["ftgsxr-app", "ftgsxr-xr"]),
        ),
        ("ftgsxr-quest", BTreeSet::from(["ftgsxr-app", "ftgsxr-xr"])),
        (
            "ftgsxr-testkit",
            BTreeSet::from(["ftgsxr-core", "ftgsxr-render"]),
        ),
    ])
}

#[test]
fn u35_skeleton_dependency_bands() {
    let meta = metadata();
    let allowed = allowed_dependencies();
    let members: BTreeSet<&str> = packages(&meta).iter().map(|p| str_of(p, "name")).collect();
    let mut broken = Vec::new();
    for p in packages(&meta) {
        let name = str_of(p, "name");
        for d in p["dependencies"]
            .as_array()
            .expect("a package has dependencies")
        {
            let dep = str_of(d, "name");
            if !members.contains(dep) {
                continue;
            }
            let dev = str_of(d, "kind") == "dev";
            let ok = if dep == "ftgsxr-testkit" {
                dev && name != "ftgsxr-testkit"
            } else {
                allowed.get(name).is_some_and(|a| a.contains(dep))
            };
            if !ok {
                broken.push(format!(
                    "{name} -> {dep}{}",
                    if dev { " (dev)" } else { "" }
                ));
            }
        }
    }
    assert!(
        broken.is_empty(),
        "dependencies outside §2.1's band table: {broken:?}"
    );
}

/// The comment that starts a source file's SPDX line, by extension; `None` for a file
/// that is not source (data, documentation, licenses). Q-7.
fn spdx_comment(path: &Path) -> Option<&'static str> {
    match path.extension()?.to_str()? {
        "rs" | "wgsl" | "wesl" | "kt" | "kts" | "gradle" => Some("//"),
        "toml" | "yaml" | "yml" | "sh" | "py" => Some("#"),
        _ => None,
    }
}

/// Directories the walk skips: build outputs, caches and version control, anywhere; and,
/// from the workspace root, the agent tooling and scratch notes and the committed data
/// fixtures, whose bytes are the documented formats' (Q-7).
fn skipped(root: &Path, dir: &Path) -> bool {
    let name = dir.file_name().and_then(|n| n.to_str()).unwrap_or_default();
    if matches!(
        name,
        ".git" | "target" | "__pycache__" | ".venv" | ".gradle" | "build"
    ) {
        return true;
    }
    let rel = dir.strip_prefix(root).unwrap_or(dir);
    [".claude", ".scratch", "tests/fixtures"]
        .iter()
        .any(|s| rel == Path::new(s))
}

fn walk(root: &Path, dir: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("the workspace's directories are readable") {
        let entry = entry.expect("a directory entry is readable");
        let kind = entry.file_type().expect("a file type is readable");
        let path = entry.path();
        if kind.is_dir() {
            if !skipped(root, &path) {
                walk(root, &path, files);
            }
        } else if kind.is_file() {
            files.push(path);
        }
    }
}

fn workspace_files() -> (PathBuf, Vec<PathBuf>) {
    let root = workspace_root();
    let mut files = Vec::new();
    walk(&root, &root, &mut files);
    files.sort();
    (root, files)
}

#[test]
fn u35_skeleton_spdx_headers() {
    let (root, files) = workspace_files();
    let mut checked = 0;
    let mut missing = Vec::new();
    for path in &files {
        let Some(comment) = spdx_comment(path) else {
            continue;
        };
        checked += 1;
        let header = format!("{comment} SPDX-License-Identifier: Apache-2.0");
        let text = fs::read_to_string(path).unwrap_or_default();
        let mut lines = text.lines();
        let first = lines.next().unwrap_or_default();
        // A script may start with its interpreter line, the header following it.
        let ok = first == header
            || (comment == "#" && first.starts_with("#!") && lines.next() == Some(&header));
        if !ok {
            missing.push(path.strip_prefix(&root).unwrap_or(path).to_owned());
        }
    }
    assert!(checked > 0, "the walk found the workspace's source files");
    assert!(
        missing.is_empty(),
        "source files without their SPDX header: {missing:?}"
    );
}

#[test]
fn u35_skeleton_shaders_referenced() {
    let (root, files) = workspace_files();
    let render_src = root.join("crates/ftgsxr-render/src");
    // The text of `ftgsxr-render`'s Rust and shader sources.
    let render: Vec<(&PathBuf, String)> = files
        .iter()
        .filter(|p| p.starts_with(&render_src) && spdx_comment(p) == Some("//"))
        .map(|p| (p, fs::read_to_string(p).unwrap_or_default()))
        .collect();
    let unreferenced: Vec<&PathBuf> = files
        .iter()
        .filter(|p| {
            matches!(
                p.extension().and_then(|e| e.to_str()),
                Some("wgsl" | "wesl")
            )
        })
        .filter(|shader| {
            let name = shader
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            // Referenced: named in a source file of `ftgsxr-render` other than itself.
            !render
                .iter()
                .any(|(p, text)| p != shader && text.contains(name))
        })
        .collect();
    assert!(
        unreferenced.is_empty(),
        "shader files that ftgsxr-render does not reference: {unreferenced:?}"
    );
}
