// SPDX-License-Identifier: Apache-2.0

//! I-19's paths clause (§1.11, §2.5, `cpu`): `AppPaths::linux` under a temporary XDG
//! environment and `AppPaths::android` from given directories give §1.11's layout
//! through `PlatformRes`, and the library's default source is the models directory.
//! The test sets the environment, which is process-wide, and the crate's other tests share
//! this test binary and run in parallel with it (#413), so it runs again in a child process
//! of the binary, alone, and sets the environment only there.

use crate::support;

use std::path::Path;
use std::process::Command;

use ftgsxr_app::{AppPaths, Library, PlatformRes};
use support::{StubPlatform, TestApp};

/// Set in the child's environment: run the test's clauses.
const CHILD: &str = "FTGSXR_I19_PATHS_CHILD";

fn set(key: &str, value: &Path) {
    // SAFETY: only the child sets the environment, and it runs this one test alone
    // (`--exact`, one test thread) and sets it before any other thread of it reads it.
    unsafe { std::env::set_var(key, value) }
}

/// Runs the test `name` of this file again, alone, in a child process of this test binary
/// with `CHILD` set, and fails unless the child ran it and it passed.
fn run_in_child(name: &str) {
    let path = format!("{}::{name}", crate::test_file_stem());
    let out = Command::new(std::env::current_exe().expect("the test binary"))
        .args([path.as_str(), "--exact", "--test-threads", "1"])
        .env(CHILD, "1")
        .output()
        .expect("the child run");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success() && stdout.contains("test result: ok. 1 passed"),
        "{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn through_platform(name: &str, paths: AppPaths) -> (AppPaths, String) {
    let t = TestApp::on(name, StubPlatform::with_paths(paths));
    let world = t.app.world();
    let got = world.resource::<PlatformRes>().0.paths().clone();
    let source = world.resource::<Library>().sources[0].1.name().to_owned();
    (got, source)
}

#[test]
fn i19_paths_reach_the_world_through_the_platform() {
    if std::env::var_os(CHILD).is_none() {
        return run_in_child("i19_paths_reach_the_world_through_the_platform");
    }
    let root = support::scratch("i19-xdg");
    set("HOME", &root.join("home"));
    set("XDG_CONFIG_HOME", &root.join("config"));
    set("XDG_CACHE_HOME", &root.join("cache"));
    set("XDG_DATA_HOME", &root.join("data"));
    set("XDG_PICTURES_DIR", &root.join("pictures"));
    let (p, source) = through_platform("i19-linux", AppPaths::linux().unwrap());
    let app = |d: &str| root.join(d).join("openftgs-xr");
    assert_eq!(p.config, app("config"));
    assert_eq!(p.scenes, app("config").join("scenes"));
    assert_eq!(p.cache, app("cache"));
    assert_eq!(p.logs, app("cache").join("logs"));
    assert_eq!(p.models, app("data").join("models"));
    assert_eq!(p.screenshots, app("pictures"));
    assert_eq!(source, p.models.to_str().unwrap());

    let (internal, external, cache) = (root.join("int"), root.join("ext"), root.join("c"));
    let (p, source) = through_platform(
        "i19-android",
        AppPaths::android(&internal, Some(&external), &cache),
    );
    assert_eq!(p.config, internal);
    assert_eq!(p.scenes, internal.join("scenes"));
    assert_eq!(p.models, external.join("models"));
    assert_eq!(p.screenshots, external.join("screenshots"));
    assert_eq!(p.cache, cache);
    assert_eq!(p.logs, cache.join("logs"));
    assert_eq!(source, p.models.to_str().unwrap());
}
