// SPDX-License-Identifier: Apache-2.0

//! `ftgsxr_quest::main` (§2.1, §2.5): logcat under `ftgsxr` (§1.13), the paths (Q-15),
//! the settings, the headless device tests, then a Bevy `App` with no window driven by the
//! XR session: `XrPlugin` through `add_xr` (Q-120), the `Headset` marker (Q-266) and
//! `AppPlugin` over `QuestPlatform` at the settings' tier (#458; Q-129's harness retired).

use std::sync::Arc;
use std::time::Duration;

use bevy::android::ANDROID_APP;
use bevy::prelude::*;
use bevy::window::ExitCondition;
use bevy::winit::{UpdateMode, WinitSettings};
use bevy_mod_xr::session::XrState;
use ftgsxr_app::{
    AppPaths, AppPlugin, Headset, ModelBytes, QualityTier, SettingsRes, SortMode, SplatLoad, Toast,
    ToastLevel, load_settings,
};
use ftgsxr_xr::{XrHands, XrPlugin, add_xr};

use crate::exit::{EXIT_ENVIRONMENT, exit_code, exit_without_session};
use crate::platform::QuestPlatform;
use crate::{bench_runner, device_test, jni_calls};

/// The logcat tag (§1.13).
const TAG: &str = "ftgsxr";

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

/// The Quest's locations (§2.5, Q-15), and whether the app-external directory exists.
fn paths() -> (AppPaths, bool) {
    let app = ANDROID_APP
        .get()
        .expect("invariant: #[bevy_main] sets the AndroidApp before main");
    let internal = app
        .internal_data_path()
        .expect("invariant: Android gives every app its internal files directory");
    let external = app.external_data_path();
    let cache = jni_calls::cache_dir().unwrap_or_else(|| internal.join("cache"));
    (
        AppPaths::android(&internal, external.as_deref(), &cache),
        external.is_some(),
    )
}

/// Q-15 (owner, #36): without the app-external directory, the models and screenshots stay
/// in internal storage; the platform layer says so in the log and a toast.
// Q-15
const NO_EXTERNAL: &str = "shared storage is unavailable: models and screenshots stay in \
     internal storage, out of adb and MTP reach, and are not moved when it returns";

fn toast_no_external(mut toasts: MessageWriter<Toast>) {
    toasts.write(Toast {
        text: NO_EXTERNAL.into(),
        level: ToastLevel::Warning,
    });
}

/// A model's bytes to load as they are (no kind hint, no reduction).
pub(crate) fn splat_load(bytes: Vec<u8>) -> SplatLoad {
    SplatLoad {
        bytes: ModelBytes::Owned(bytes),
        kind_hint: None,
        reduce_to: None,
    }
}

#[bevy_main]
fn main() {
    let test = jni_calls::intent_extra(device_test::EXTRA);
    // Q-406
    let bench = jni_calls::intent_extra(bench_runner::EXTRA);
    let (paths, external) = paths();
    let (settings, warnings) = match load_settings(&paths.config.join("settings.toml")) {
        Ok(s) => s,
        Err(e) => (Default::default(), vec![format!("{}: {e}", e.code())]),
    };
    // The verdict lines are info; a device test or a benchmark keeps them whatever the
    // settings' level.
    let level = match (
        test.is_some() || bench.is_some(),
        settings.general.log_level.as_str(),
    ) {
        (true, "error" | "warn") => "info",
        (_, l) => l,
    };
    init_log(level);
    for w in &warnings {
        warn!("{w}");
    }
    if !external {
        warn!("{NO_EXTERNAL}");
    }
    info!(
        "openftgs-xr {} on the Quest: {paths:?}",
        env!("CARGO_PKG_VERSION")
    );

    if let Some(name) = &bench {
        // Q-406: B-3, B-5 and B-6 before the XR session, as Dv-3 (Q-122), then the process
        // exits.
        let headset = jni_calls::intent_extra(bench_runner::HEADSET_EXTRA);
        let result = match device_test::dv_dir() {
            Some(dv) => bench_runner::run(name, &dv, headset.as_deref()),
            None => Err("E-IO: no app-external files directory for dv/".to_owned()),
        };
        info!("{}", bench_runner::line(name, &result));
        std::process::exit(0);
    }

    if let Some(name) = &test {
        // Dv-3, Dv-4, R-9 and R-10 (Q-122, Q-T728.2): before the XR session, then the
        // process exits.
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
        // §1.9 (#568): the layer where the runtime offers it.
        passthrough: true,
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
        .add_plugins(add_xr(group, xr));
    if app.world().get_resource::<XrState>() == Some(&XrState::Unavailable) {
        // E-XR-RUNTIME, which `XrPlugin` logged: the Quest exits 2 (§1.15). Before the
        // app's first frame, so no warm-up runs yet (Q-325).
        std::process::exit(i32::from(EXIT_ENVIRONMENT));
    }
    // Q-266: before `AppPlugin`, so panel mode, spatial audio and D-11's auto-fit apply.
    app.insert_resource(Headset);
    // Q-163: `SettingsRes` above is left alone by `AppPlugin`.
    let platform = QuestPlatform::new(paths);
    // Q-387: the head's stage pose for the camera's thumbnails, in `Last`, after
    // `tracking` wrote this frame's `XrHands.head` in PostUpdate; `light` reads it from
    // the next frame's `latest`.
    let head = platform.clone();
    app.add_systems(Last, move |hands: Option<Res<XrHands>>| {
        if let Some(hands) = hands {
            head.set_head(hands.head.translation, hands.head.rotation);
        }
    });
    app.add_plugins(AppPlugin {
        platform: Arc::new(platform),
        tier,
        initial: Vec::new(),
        scene: None,
    });
    app.add_systems(Update, exit_without_session);
    if !external {
        app.add_systems(Startup, toast_no_external);
    }
    if let Some(name) = test {
        info!("device test {name}");
        device_test::add(&mut app, &name);
    }
    // #660: the runner drops the app, and with it the render world (Q-381's join), before
    // `run` returns; an error exit (E-XR-RUNTIME's 2, §1.15) then ends the process with
    // its code, and a success returns as before.
    let code = exit_code(&app.run());
    if code != 0 {
        std::process::exit(code);
    }
}
