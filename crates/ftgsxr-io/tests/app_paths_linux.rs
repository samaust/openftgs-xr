// SPDX-License-Identifier: Apache-2.0

//! PRD #1's added `AppPaths` test, the desktop part (`cpu` lane; source of truth §1.11
//! Locations, exact): with `XDG_CONFIG_HOME`, `XDG_CACHE_HOME`, `XDG_DATA_HOME` and
//! `XDG_PICTURES_DIR` pointed at temporary directories, `AppPaths::linux` yields exactly
//! §1.11's six locations. The screenshots root's other sources are Q-T16.1's.
//!
//! The environment is process-wide, so every clause that changes it is in the one test
//! function of this test binary, which nothing else shares.

use std::path::{Path, PathBuf};

use ftgsxr_io::paths::AppPaths;

fn set(key: &str, value: &Path) {
    // SAFETY: this binary's only test is the one function below, so no other thread of
    // this process reads or writes the environment while it runs.
    unsafe { std::env::set_var(key, value) }
}

fn unset(key: &str) {
    // SAFETY: as in `set`.
    unsafe { std::env::remove_var(key) }
}

#[test]
fn linux_yields_the_six_xdg_locations_of_section_1_11() {
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

    // Q-T16.1: without the variable, the user-dirs file's entry.
    let user_dirs_pictures = root.join("user-dirs-pictures");
    std::fs::write(
        config.join("user-dirs.dirs"),
        format!("XDG_PICTURES_DIR=\"{}\"\n", user_dirs_pictures.display()),
    )
    .unwrap();
    unset("XDG_PICTURES_DIR");
    let p = AppPaths::linux().unwrap();
    assert_eq!(p.screenshots, user_dirs_pictures.join("openftgs-xr"));

    // Q-T16.1: a relative variable is ignored, as `directories` ignores relative XDG
    // variables.
    set("XDG_PICTURES_DIR", Path::new("relative/pictures"));
    let p = AppPaths::linux().unwrap();
    assert_eq!(p.screenshots, user_dirs_pictures.join("openftgs-xr"));

    // Q-T16.1: with neither, $HOME/Pictures.
    unset("XDG_PICTURES_DIR");
    std::fs::remove_file(config.join("user-dirs.dirs")).unwrap();
    let p = AppPaths::linux().unwrap();
    assert_eq!(
        p.screenshots,
        PathBuf::from(&home).join("Pictures").join("openftgs-xr")
    );

    // Q-T16.1: `linux` creates nothing.
    assert!(!cache.exists() && !data.exists() && !home.exists());

    std::fs::remove_dir_all(&root).unwrap();
}
