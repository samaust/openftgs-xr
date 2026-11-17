// SPDX-License-Identifier: Apache-2.0

//! The layer's foveation on Monado (`xr` lane, #809; §1.12 Headset display, D-58): each
//! eye's `SplatFoveation` follows an explicit `[xr] foveation` and a governor step
//! (Q-458, Q-421), and the run records which path the lane's device takes: the info line
//! once while the level is not off where the device lacks the density-map support, with
//! no toast, otherwise a foveated layer and no line (Q-455). Prerequisites:
//! `xr_support`'s.

use crate::xr_support;

use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy::render::renderer::{RenderAdapter, RenderDevice};
use bevy_mod_xr::camera::XrCamera;
use bevy_mod_xr::session::XrState;
use ftgsxr_app::{SettingsRes, Toast, XrDisplay};
use ftgsxr_io::Settings;
use ftgsxr_render::{
    DensityMapSupport, SortMode, SplatFoveation, SplatTierState, density_map_support,
};
use ftgsxr_xr::{Foveation, XrPlugin};
use xr_support::{Monado, capture_log, lines, xr_app};

/// The toasts triggered so far.
#[derive(Resource, Default)]
struct Toasts(usize);

fn count(_: On<Toast>, mut n: ResMut<Toasts>) {
    n.0 += 1;
}

/// The app on Monado over `settings`, run until its session runs and a few frames more.
fn running(settings: Settings) -> App {
    let mut app = xr_app(
        XrPlugin {
            passthrough: false,
            hands: false,
            sort: SortMode::PerView,
        },
        |app| {
            app.insert_resource(SettingsRes(settings))
                .init_resource::<Toasts>()
                .add_observer(count);
        },
    );
    let start = Instant::now();
    while app.world().get_resource::<XrState>() != Some(&XrState::Running) {
        assert!(
            start.elapsed() < Duration::from_secs(20),
            "the session runs within 20 s"
        );
        app.update();
    }
    for _ in 0..5 {
        app.update();
    }
    app
}

/// Each eye's level, in `XrCamera` order.
fn eye_levels(app: &mut App) -> Vec<Option<Foveation>> {
    let mut eyes: Vec<(u32, Option<Foveation>)> = app
        .world_mut()
        .query::<(&XrCamera, Option<&SplatFoveation>)>()
        .iter(app.world())
        .map(|(c, f)| (c.0, f.map(|f| f.0)))
        .collect();
    eyes.sort_by_key(|(i, _)| *i);
    eyes.into_iter().map(|(_, f)| f).collect()
}

fn support(app: &App) -> DensityMapSupport {
    let adapter = app.world().resource::<RenderAdapter>();
    let device = app.world().resource::<RenderDevice>();
    density_map_support(adapter, device.wgpu_device()).expect("Monado's device is Vulkan")
}

fn layer_lines() -> Vec<String> {
    lines()
        .into_iter()
        .filter(|l| {
            l.target == "ftgsxr_xr::display"
                && l.message.starts_with("layer foveation unavailable: ")
        })
        .map(|l| l.message)
        .collect()
}

/// Q-458 (#809): with `[xr] foveation = "medium"` both eyes carry Medium against the
/// tier; unset, a governor step (the governor writes `SplatTierState.foveation`) reaches
/// both eyes. Q-456 on the lane: `XrDisplay.foveation` is the device's signal.
#[test]
#[ignore = "xr"]
fn xr_layer_foveation_the_eyes_follow_an_explicit_xr_foveation_and_a_governor_step() {
    capture_log();
    let _monado = Monado::start();
    let mut settings = Settings::default();
    settings.xr.foveation = Some("medium".to_owned());
    let mut app = running(settings);
    let eyes = eye_levels(&mut app);
    assert_eq!(eyes.len(), 2, "two eyes");
    assert_eq!(eyes, vec![Some(Foveation::Medium); 2]);
    assert_eq!(
        app.world().resource::<XrDisplay>().foveation,
        support(&app).enabled
    );
    app.world_mut().resource_mut::<SettingsRes>().0.xr.foveation = None;
    for f in [Foveation::Low, Foveation::High] {
        app.world_mut().resource_mut::<SplatTierState>().foveation = f;
        for _ in 0..3 {
            app.update();
        }
        assert_eq!(eye_levels(&mut app), vec![Some(f); 2], "the step to {f:?}");
    }
}

/// Q-455 (#809): the tier's level, not off, on the lane's device. Without support, one
/// line naming the missing item and no toast, a later level adding none; with it, no line.
/// The path taken is printed as the lane's record.
#[test]
#[ignore = "xr"]
fn xr_layer_foveation_records_the_lanes_path() {
    capture_log();
    let _monado = Monado::start();
    let mut app = running(Settings::default());
    let s = support(&app);
    for f in [Foveation::Low, Foveation::High] {
        app.world_mut().resource_mut::<SplatTierState>().foveation = f;
        for _ in 0..3 {
            app.update();
        }
    }
    let got = layer_lines();
    if s.enabled {
        eprintln!("xr lane path: foveated layer ({s})");
        assert!(got.is_empty(), "{got:?}");
    } else {
        eprintln!("xr lane path: unfoveated layer, {got:?} ({s})");
        assert_eq!(got.len(), 1, "one line per session: {got:?}");
    }
    assert_eq!(app.world().resource::<Toasts>().0, 0, "no toast");
}
