// SPDX-License-Identifier: Apache-2.0

//! I-19's paths clause (§1.11, §2.5, `cpu`): `AppPaths::linux` under a temporary XDG
//! environment and `AppPaths::android` from given directories give §1.11's layout
//! through `PlatformRes`, and the library's default source is the models directory.
//! One test in this binary, as it sets the environment.

mod support;

use std::path::Path;

use ftgsxr_app::{AppPaths, Library, PlatformRes};
use support::{StubPlatform, TestApp};

fn set(key: &str, value: &Path) {
    // SAFETY: this binary holds this one test and sets the environment before any
    // other thread of it reads it.
    unsafe { std::env::set_var(key, value) }
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
