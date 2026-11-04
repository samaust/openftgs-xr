// SPDX-License-Identifier: Apache-2.0

//! PRD #1's added `AppPaths` test, the desktop part (`cpu` lane; source of truth §1.11
//! Locations, exact): with `XDG_CONFIG_HOME`, `XDG_CACHE_HOME`, `XDG_DATA_HOME` and
//! `XDG_PICTURES_DIR` pointed at temporary directories, `AppPaths::linux` yields exactly
//! §1.11's six locations. The screenshots root's other sources are Q-14's.
//!
//! The environment is process-wide, and the crate's other tests share this test binary
//! and run in parallel with this one (#413), so every clause that changes it is in the one
//! test function below, which runs again in a child process of the binary, alone, and sets
//! the environment only there.

use std::path::{Path, PathBuf};
use std::process::Command;

use ftgsxr_io::paths::AppPaths;

/// Set in the child's environment: run the test's clauses.
const CHILD: &str = "FTGSXR_APP_PATHS_LINUX_CHILD";

fn set(key: &str, value: &Path) {
    // SAFETY: only the child sets the environment, and it runs this one test alone
    // (`--exact`, one test thread), so no other thread of its process reads or writes the
    // environment while it does.
    unsafe { std::env::set_var(key, value) }
}

fn unset(key: &str) {
    // SAFETY: as in `set`.
    unsafe { std::env::remove_var(key) }
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

#[test]
fn linux_yields_the_six_xdg_locations_of_section_1_11() {
    if std::env::var_os(CHILD).is_none() {
        return run_in_child("linux_yields_the_six_xdg_locations_of_section_1_11");
    }
    let root = std::env::temp_dir().join(format!("ftgsxr-app-paths-{}", std::process::id()));
    let home = root.join("home");
    let config = root.join("config");
    let cache = root.join("cache");
    let data = root.join("data");
    let pictures = root.join("pictures");
    std::fs::create_dir_all(&config).unwrap();
    set("HOME", &home);
    set("XDG_CONFIG_HOME", &config);
    set("XDG_CACHE_HOME", &cache);
    set("XDG_DATA_HOME", &data);
    set("XDG_PICTURES_DIR", &pictures);

    let p = AppPaths::linux().unwrap();
    // Settings, the pose store and scenes under $XDG_CONFIG_HOME/openftgs-xr/.
    assert_eq!(p.config, config.join("openftgs-xr"));
    assert_eq!(p.scenes, config.join("openftgs-xr").join("scenes"));
    // The HTTP cache, thumbnails and logs under $XDG_CACHE_HOME/openftgs-xr/.
    assert_eq!(p.cache, cache.join("openftgs-xr"));
    assert_eq!(p.logs, cache.join("openftgs-xr").join("logs"));
    // The default model directory $XDG_DATA_HOME/openftgs-xr/models/.
    assert_eq!(p.models, data.join("openftgs-xr").join("models"));
    // Screenshots under $XDG_PICTURES_DIR/openftgs-xr/.
    assert_eq!(p.screenshots, pictures.join("openftgs-xr"));

    // Q-14: without the variable, the user-dirs file's entry.
    let user_dirs_pictures = root.join("user-dirs-pictures");
    std::fs::write(
        config.join("user-dirs.dirs"),
        format!("XDG_PICTURES_DIR=\"{}\"\n", user_dirs_pictures.display()),
    )
    .unwrap();
    unset("XDG_PICTURES_DIR");
    let p = AppPaths::linux().unwrap();
    assert_eq!(p.screenshots, user_dirs_pictures.join("openftgs-xr"));

    // Q-14: a relative variable is ignored, as `directories` ignores relative XDG
    // variables.
    set("XDG_PICTURES_DIR", Path::new("relative/pictures"));
    let p = AppPaths::linux().unwrap();
    assert_eq!(p.screenshots, user_dirs_pictures.join("openftgs-xr"));

    // Q-14: with neither, $HOME/Pictures.
    unset("XDG_PICTURES_DIR");
    std::fs::remove_file(config.join("user-dirs.dirs")).unwrap();
    let p = AppPaths::linux().unwrap();
    assert_eq!(
        p.screenshots,
        PathBuf::from(&home).join("Pictures").join("openftgs-xr")
    );

    // Q-14: `linux` creates nothing.
    assert!(!cache.exists() && !data.exists() && !home.exists());

    std::fs::remove_dir_all(&root).unwrap();
}
