// SPDX-License-Identifier: Apache-2.0

//! `ftgsxr_quest::main` (§2.1, §2.5): logcat under `ftgsxr` (§1.13), the paths (Q-15),
//! the settings, a Bevy `App` with no window driven by the XR session, `SplatPlugin` at
//! the settings' tier and `XrPlugin` through `add_xr` (Q-120).

use std::path::Path;
use std::time::Duration;

use bevy::android::ANDROID_APP;
use bevy::prelude::*;
use bevy::window::ExitCondition;
use bevy::winit::{UpdateMode, WinitSettings};
use bevy_mod_xr::session::{XrSessionCreatedMessage, XrState};
use bevy_mod_xr::spaces::XrPrimaryReferenceSpace;
use ftgsxr_app::{
    AppPaths, ModelBytes, QualityTier, SettingsRes, SortMode, SplatLoad, SplatPlugin, SplatTime,
    load_settings,
};
use ftgsxr_xr::{XrPlugin, add_xr};

use crate::{device_test, jni_calls};

/// The logcat tag (§1.13).
const TAG: &str = "ftgsxr";

/// The exit code of E-XR-RUNTIME on the Quest (§1.15).
const EXIT_ENVIRONMENT: i32 = 2;

/// Logs to logcat at `level` (§1.13): tracing's events reach android_logger through
/// tracing's `log` records, as no tracing subscriber is installed.
fn init_log(level: &str) {
    let filter = tracing::log::LevelFilter::Trace;
    android_logger::init_once(
        android_logger::Config::default()
            .with_tag(TAG)
            .with_max_level(filter)
            .with_filter(android_logger::FilterBuilder::new().parse(level).build()),
    );
}

/// The Quest's locations (§2.5, Q-15).
fn paths() -> AppPaths {
    let app = ANDROID_APP
        .get()
        .expect("invariant: #[bevy_main] sets the AndroidApp before main");
    let internal = app
        .internal_data_path()
        .expect("invariant: Android gives every app its internal files directory");
    let external = app.external_data_path();
    let cache = jni_calls::cache_dir().unwrap_or_else(|| internal.join("cache"));
    AppPaths::android(&internal, external.as_deref(), &cache)
}

/// A model's bytes to load as they are (no kind hint, no reduction).
pub(crate) fn splat_load(bytes: Vec<u8>) -> SplatLoad {
    SplatLoad {
        bytes: ModelBytes::Owned(bytes),
        kind_hint: None,
        reduce_to: None,
    }
}

/// The provisional harness M-4's `AppPlugin` replaces: every exchange file of the models
/// directory as a model instance at the identity pose (#224; D-11's fit is not claimed).
fn spawn_models(app: &mut App, models: &Path) {
    let Ok(entries) = std::fs::read_dir(models) else {
        info!("no models directory at {}", models.display());
        return;
    };
    for path in entries.flatten().map(|e| e.path()) {
        if path.extension().is_none_or(|e| e != "safetensors") {
            continue;
        }
        match std::fs::read(&path) {
            Ok(bytes) => {
                info!("loading {}", path.display());
                app.world_mut().spawn((
                    splat_load(bytes),
                    SplatTime {
                        tau: 0.0,
                        extrapolating: false,
                    },
                    Transform::IDENTITY,
                ));
            }
            Err(e) => warn!("E-IO: {}: {e}", path.display()),
        }
    }
}

/// E-XR-RUNTIME when the session cannot be created (§1.15): the Quest exits 2. The XR
/// crate logs it and inserts the primary reference space only once a session exists, so
/// a session-created message without that space is the failure (or a session left
/// without its reference space, which is E-XR-RUNTIME too).
fn exit_without_session(
    mut created: MessageReader<XrSessionCreatedMessage>,
    space: Option<Res<XrPrimaryReferenceSpace>>,
) {
    if created.read().count() > 0 && space.is_none() {
        std::process::exit(EXIT_ENVIRONMENT);
    }
}

#[bevy_main]
fn main() {
    let test = jni_calls::intent_extra(device_test::EXTRA);
    let paths = paths();
    let (settings, warnings) = match load_settings(&paths.config.join("settings.toml")) {
        Ok(s) => s,
        Err(e) => (Default::default(), vec![format!("{}: {e}", e.code())]),
    };
    // The verdict lines are info; a device test keeps them whatever the settings' level.
    let level = match (&test, settings.general.log_level.as_str()) {
        (Some(_), "error" | "warn") => "info",
        (_, l) => l,
    };
    init_log(level);
    for w in &warnings {
        warn!("{w}");
    }
    info!(
        "openftgs-xr {} on the Quest: {paths:?}",
        env!("CARGO_PKG_VERSION")
    );

    if let Some(name) = &test {
        // Dv-3, Dv-4 (Q-122): before the XR session, then the process exits.
        if let Some(dv) = device_test::dv_dir()
            && device_test::run_headless(name, &dv)
        {
            std::process::exit(0);
        }
    }

    let tier = QualityTier::by_name(&settings.general.tier).unwrap_or_else(|| {
        warn!("unknown tier {}; quest-balanced", settings.general.tier);
        QualityTier::by_name("quest-balanced").expect("invariant: §1.12's tier exists")
    });
    let xr = XrPlugin {
        passthrough: false,
        hands: true,
        // Q-121
        sort: SortMode::PerView,
    };
    let group = DefaultPlugins
        .build()
        .disable::<bevy::log::LogPlugin>()
        .set(WindowPlugin {
            // §2.5: no Bevy window; the XR session drives the frames.
            primary_window: None,
            exit_condition: ExitCondition::DontExit,
            close_when_requested: false,
            ..default()
        });
    let mut app = App::new();
    app.insert_resource(SettingsRes(settings.clone()))
        // Spike check 1 (PRD #221): winit's runner waits on Android events with no window
        // to redraw; Reactive with no wait keeps it polling, and xrWaitFrame paces it.
        .insert_resource(WinitSettings {
            focused_mode: UpdateMode::reactive(Duration::ZERO),
            unfocused_mode: UpdateMode::reactive(Duration::ZERO),
        })
        .add_plugins(add_xr(group, xr))
        .add_plugins(SplatPlugin { tier });
    if app.world().get_resource::<XrState>() == Some(&XrState::Unavailable) {
        // E-XR-RUNTIME, which `XrPlugin` logged: the Quest exits 2 (§1.15).
        std::process::exit(EXIT_ENVIRONMENT);
    }
    app.add_systems(Update, exit_without_session);
    spawn_models(&mut app, &paths.models);
    if let Some(name) = test {
        info!("device test {name}");
        device_test::add(&mut app, &name);
    }
    app.run();
}
