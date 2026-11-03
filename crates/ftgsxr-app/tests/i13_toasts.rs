// SPDX-License-Identifier: Apache-2.0

//! I-13's toasts (§1.15, #286): the failures the library reports reach a `Toast` with
//! their code and text: an invalid file's E-FILE code and an unreachable URL's
//! E-NET-FETCH (`cpu`), E-MEM-BUDGET with N (Q-117, `gpu_`).

mod support;

use std::fs;
use std::net::TcpListener;

use ftgsxr_app::{EntryState, Library, LibraryCommand, ModelRef, QualityTier, ToastLevel, Url};
use ftgsxr_core::memory_estimate;
use ftgsxr_testkit::{exchange_bytes, fixtures_dir, layer_messages, probe};
use support::ui::{record_toasts, toasted, toasts};
use support::{Setup, TestApp};

/// A localhost URL whose port was bound and closed again.
fn closed_url(path: &str) -> String {
    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    format!("http://127.0.0.1:{port}/{path}")
}

fn listed(t: &mut TestApp) {
    t.command(LibraryCommand::Refresh(0));
    assert!(
        t.until(300, |app| app.world().resource::<Library>().entries.len()
            == 1)
    );
}

#[test]
fn i13_an_invalid_file_is_a_toast_with_its_e_file_code() {
    let mut t = TestApp::new("i13t-file");
    record_toasts(&mut t.app);
    let (mut model, _) = probe("P1");
    model.position[0][0] = f32::NAN;
    let models = t.dir.join("models");
    fs::create_dir_all(&models).unwrap();
    fs::write(models.join("bad.safetensors"), exchange_bytes(&model)).unwrap();
    listed(&mut t);
    t.command(LibraryCommand::Load {
        entry: 0,
        reduce: false,
    });
    assert!(
        t.until(600, |app| toasted(app, "E-FILE-VALUE: ")),
        "{:?}",
        toasts(&t.app)
    );
    t.frames(5);
    let all = toasts(&t.app);
    assert_eq!(all.len(), 1, "one toast per failure: {all:?}");
    assert_eq!(all[0].level, ToastLevel::Error);
}

#[test]
fn i13_an_unreachable_url_is_an_e_net_fetch_toast() {
    let mut t = TestApp::new("i13t-net");
    record_toasts(&mut t.app);
    let url = Url::parse(&closed_url("m.safetensors")).unwrap();
    t.command(LibraryCommand::LoadRef(ModelRef::Http(url)));
    assert!(
        t.until(3000, |app| toasted(app, "E-NET-FETCH: Could not reach")),
        "{:?}",
        toasts(&t.app)
    );
}

#[test]
#[ignore = "gpu"]
fn gpu_i13_over_budget_is_an_e_mem_budget_toast_with_n() {
    let _layer = layer_messages(); // R-16, Q-105
    let (model, _) = probe("P1");
    let n = model.len() as u64;
    let mut tier = QualityTier::by_name("desktop").unwrap();
    let full = memory_estimate(n, model.sh_degree, &tier.layout, false).static_bytes;
    tier.memory_budget_bytes = full / 2;
    let mut t = TestApp::gpu_with(
        "i13t-budget",
        Setup {
            tier,
            ..Setup::default()
        },
    );
    record_toasts(&mut t.app);
    let models = t.dir.join("models");
    fs::create_dir_all(&models).unwrap();
    fs::copy(
        fixtures_dir().join("probes/P1.safetensors"),
        models.join("P1.safetensors"),
    )
    .unwrap();
    listed(&mut t);
    t.command(LibraryCommand::Load {
        entry: 0,
        reduce: false,
    });
    let failed = |app: &mut bevy::app::App| {
        matches!(
            app.world().resource::<Library>().entries[0].state,
            EntryState::Failed(_)
        )
    };
    assert!(t.until(600, failed), "refused");
    t.frames(2);
    let all = toasts(&t.app);
    let budget: Vec<_> = all
        .iter()
        .filter(|t| t.text.starts_with("E-MEM-BUDGET: Needs "))
        .collect();
    assert_eq!(budget.len(), 1, "{all:?}");
    assert!(
        budget[0]
            .text
            .ends_with(&format!("reduce to budget or choose a tier (N = {n})")),
        "{}",
        budget[0].text
    );
}

// Q-217
#[test]
fn a_refused_clock_command_is_a_warning_toast() {
    use ftgsxr_app::{Intent, PlaybackCommand};
    let mut t = TestApp::new("i13t-clock");
    record_toasts(&mut t.app);
    t.app
        .world_mut()
        .write_message(Intent::Playback(PlaybackCommand::Speed(0.0)));
    t.frames(3);
    let all = toasts(&t.app);
    assert_eq!(all.len(), 1, "{all:?}");
    assert_eq!(all[0].level, ToastLevel::Warning);
    assert!(all[0].text.starts_with("Speed 0 refused"), "{all:?}");
}

// Q-161, Q-221: a `Toast` triggered from outside the app crate (the XR crate's
// E-XR-UNSUPPORTED) reaches the line through the rate limit, once for a repeat.
#[test]
fn a_triggered_toast_is_raised_once_through_the_rate_limit() {
    use ftgsxr_app::Toast;
    let mut t = TestApp::new("i13t-triggered");
    record_toasts(&mut t.app);
    let toast = Toast {
        text: "E-XR-UNSUPPORTED: Not available on this runtime (XR_FB_foveation)".to_owned(),
        level: ToastLevel::Warning,
    };
    t.app.world_mut().trigger(toast.clone());
    t.app.world_mut().trigger(toast.clone());
    t.frames(3);
    assert_eq!(toasts(&t.app), vec![toast]);
}

/// Fixer item 19 (#275; §1.15 E-SETTINGS, §1.11, Q-170): `play` reads the settings
/// itself before the log and inserts `SettingsRes`; the set-aside file's E-SETTINGS still
/// reaches a toast through `SettingsWarnings` (Q-T275.2).
#[test]
fn i13_a_set_aside_settings_file_read_by_the_caller_raises_e_settings() {
    use ftgsxr_app::{SettingsPath, SettingsRes, SettingsWarnings, load_settings};
    let dir = support::scratch("i13-settings-caller");
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("settings.toml");
    fs::write(&path, "this is = = not toml [").unwrap();
    let (settings, warnings) = load_settings(&path).expect("a corrupt file loads the defaults");
    assert!(
        warnings.iter().any(|w| w.starts_with("E-SETTINGS")),
        "{warnings:?}"
    );
    let mut t = TestApp::with("i13-settings-caller", Setup::default(), |app| {
        app.insert_resource(SettingsPath(path.clone()))
            .insert_resource(SettingsRes(settings))
            .insert_resource(SettingsWarnings(warnings));
    });
    record_toasts(&mut t.app);
    t.frames(3);
    assert!(toasted(&t.app, "E-SETTINGS"), "{:?}", toasts(&t.app));
}
