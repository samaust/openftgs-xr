// SPDX-License-Identifier: Apache-2.0

//! PRD #1's added `AppPaths` test, the Quest part (`cpu` lane; source of truth §1.11
//! Locations, exact): `AppPaths::android` maps the internal, app-external and cache
//! directories to §1.11's Quest layout.

use std::path::{Path, PathBuf};

use ftgsxr_io::paths::AppPaths;

const INTERNAL: &str = "/data/user/0/org.example.ftgsxr/files";
const EXTERNAL: &str = "/storage/emulated/0/Android/data/org.example.ftgsxr/files";
const CACHE: &str = "/data/user/0/org.example.ftgsxr/cache";

#[test]
fn android_maps_the_three_directories_to_the_quest_layout() {
    let p = AppPaths::android(
        Path::new(INTERNAL),
        Some(Path::new(EXTERNAL)),
        Path::new(CACHE),
    );
    // Settings, the pose store and scenes in the internal files directory.
    assert_eq!(p.config, PathBuf::from(INTERNAL));
    assert_eq!(p.scenes, PathBuf::from(INTERNAL).join("scenes"));
    // Models and screenshots in the app-external files directory.
    assert_eq!(p.models, PathBuf::from(EXTERNAL).join("models"));
    assert_eq!(p.screenshots, PathBuf::from(EXTERNAL).join("screenshots"));
    // The HTTP cache, thumbnails and logs in the cache directory.
    assert_eq!(p.cache, PathBuf::from(CACHE));
    assert_eq!(p.logs, PathBuf::from(CACHE).join("logs"));
}

// Q-15: without an app-external directory, models and screenshots fall back under the
// internal files directory; the rest is unchanged.
#[test]
fn android_without_an_external_directory_keeps_models_internal() {
    let p = AppPaths::android(Path::new(INTERNAL), None, Path::new(CACHE));
    assert_eq!(p.config, PathBuf::from(INTERNAL));
    assert_eq!(p.scenes, PathBuf::from(INTERNAL).join("scenes"));
    assert_eq!(p.models, PathBuf::from(INTERNAL).join("models"));
    assert_eq!(p.screenshots, PathBuf::from(INTERNAL).join("screenshots"));
    assert_eq!(p.cache, PathBuf::from(CACHE));
    assert_eq!(p.logs, PathBuf::from(CACHE).join("logs"));
}
