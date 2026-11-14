// SPDX-License-Identifier: Apache-2.0

//! The live tier switch (#373; §1.7 Panels, §1.12, D-37; Q-179, Q-238, Q-239, Q-240): a
//! pick of a tier in the dock switches the running tier and the loaded instances are
//! reloaded with the new tier's layout on their own entities, without a restart; a
//! reduced instance takes the new tier's reduce-to-budget count (#395). The pick is
//! crate-private, so the tests that pick are `governor`'s and `persist::panels`'s own
//! (#402); these switch the running tier as a pick does, by writing `SplatTierState`, and
//! check what only the whole app shows.

use crate::support;

use bevy::prelude::*;
use ftgsxr_app::{QualityTier, SettingsRes};
use ftgsxr_render::{SplatModel, SplatTierState};
use support::{Setup, TestApp};

/// The running tier's name.
fn running(app: &App) -> String {
    app.world().resource::<SplatTierState>().tier.name.clone()
}

/// Switches the running tier to `tier`, as `governor` does for a pick.
fn switch(app: &mut App, tier: QualityTier) {
    *app.world_mut().resource_mut::<SplatTierState>() = SplatTierState::of(&tier);
}

// Q-240: `--tier` at startup holds until the user picks a tier.
#[test]
fn live_tier_switch_leaves_a_startup_override_alone() {
    let setup = Setup {
        tier: QualityTier::quest_high(),
        ..Setup::default()
    };
    let mut t = TestApp::with("live_override", setup, |_| {});
    // The settings say `desktop` (the default); the session runs `quest-high`.
    t.frames(5);
    assert_eq!(running(&t.app), "quest-high");
    // An unrelated settings change does not undo the override either.
    t.app
        .world_mut()
        .resource_mut::<SettingsRes>()
        .0
        .display
        .centers_radius = 3.0;
    t.frames(2);
    assert_eq!(running(&t.app), "quest-high");
    // Nor does a write of `[general] tier` that is not a pick (#402).
    t.app
        .world_mut()
        .resource_mut::<SettingsRes>()
        .0
        .general
        .tier = "quest-low".into();
    t.frames(2);
    assert_eq!(running(&t.app), "quest-high");
}

/// Loads the probe P2 by reference and waits for its `LoadedModel`; the instance.
fn load_p2(t: &mut TestApp) -> Entity {
    let path = ftgsxr_testkit::fixtures_dir().join("probes/P2.safetensors");
    t.command(ftgsxr_app::LibraryCommand::LoadRef(
        ftgsxr_app::ModelRef::File(path),
    ));
    let mut found = None;
    assert!(
        t.until_for(std::time::Duration::from_secs(30), |app| {
            let mut q = app
                .world_mut()
                .query_filtered::<Entity, (With<ftgsxr_app::LoadedModel>, With<SplatModel>)>();
            found = q.iter(app.world()).next();
            found.is_some()
        }),
        "P2 loads"
    );
    found.unwrap()
}

/// The instance's asset, if it has one.
fn asset_of(app: &App, e: Entity) -> Option<AssetId<ftgsxr_render::SplatAsset>> {
    app.world().get::<SplatModel>(e).map(|m| m.0.id())
}

// Q-238: the reload happens on the instance's own entity, which keeps its state.
#[test]
fn live_tier_switch_keeps_the_instance() {
    let mut t = TestApp::new("live_keep");
    let e = load_p2(&mut t);
    let pose = Transform::from_xyz(0.5, -0.25, 1.0).with_scale(Vec3::splat(2.0));
    t.app.world_mut().entity_mut(e).insert(pose);
    let before = asset_of(&t.app, e).unwrap();
    let model = t
        .app
        .world()
        .get::<ftgsxr_app::LoadedModel>(e)
        .unwrap()
        .clone();
    switch(&mut t.app, QualityTier::quest_balanced());
    assert!(
        t.until_for(std::time::Duration::from_secs(30), |app| {
            asset_of(app, e).is_some_and(|a| a != before)
        }),
        "the instance is reloaded with a new asset"
    );
    let w = t.app.world();
    assert_eq!(*w.get::<Transform>(e).unwrap(), pose);
    assert_eq!(*w.get::<ftgsxr_app::LoadedModel>(e).unwrap(), model);
    let mut q = t.app.world_mut().query::<&ftgsxr_app::LoadedModel>();
    assert_eq!(q.iter(t.app.world()).count(), 1, "one instance, not two");
}

/// Every `SplatEvent` the app raises, in order.
fn record_events(app: &mut App) -> std::sync::Arc<std::sync::Mutex<Vec<ftgsxr_app::SplatEvent>>> {
    let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let seen = events.clone();
    app.add_observer(move |e: On<ftgsxr_app::SplatEvent>| {
        seen.lock().unwrap().push(e.event().clone());
    });
    events
}

/// The bytes of each `Uploaded` of `entity`, in order.
fn uploads(events: &std::sync::Mutex<Vec<ftgsxr_app::SplatEvent>>, entity: Entity) -> Vec<u64> {
    events
        .lock()
        .unwrap()
        .iter()
        .filter_map(|e| match e {
            ftgsxr_app::SplatEvent::Uploaded {
                entity: x, bytes, ..
            } if *x == entity => Some(*bytes),
            _ => None,
        })
        .collect()
}

// Q-238, Q-239: the reload is packed with the new tier's layout.
#[test]
#[ignore = "gpu"]
fn gpu_live_tier_switch_reload_is_packed_with_the_new_layout() {
    let _layer = ftgsxr_testkit::layer_messages(); // R-16, Q-105
    let mut t = TestApp::gpu("live_layout");
    let events = record_events(&mut t.app);
    let e = load_p2(&mut t);
    assert!(
        t.until_for(std::time::Duration::from_secs(30), |_| !uploads(&events, e)
            .is_empty()),
        "the first upload"
    );
    let info = t
        .app
        .world()
        .get::<ftgsxr_app::LoadedModel>(e)
        .unwrap()
        .info
        .clone();
    let (n, l) = (info.n, u64::from(info.sh_degree));
    // §1.12 Memory: float32 storage 64 + 12 (L+1)² bytes per primitive (`desktop`, cap 3).
    let l3 = l.min(3);
    assert_eq!(
        uploads(&events, e),
        vec![n * (64 + 12 * (l3 + 1) * (l3 + 1))]
    );
    // A hidden instance stays hidden through the reload (Q-238).
    t.app.world_mut().entity_mut(e).insert(Visibility::Hidden);
    switch(&mut t.app, QualityTier::quest_balanced());
    assert!(
        t.until_for(std::time::Duration::from_secs(30), |_| uploads(&events, e)
            .len()
            == 2),
        "the reload's upload"
    );
    // §1.12: `quest-balanced` stores velocity, scale, rotation, opacity and sh in half
    // precision, 42 + 6 (L+1)² bytes per primitive, with its SH cap 2.
    let l2 = l.min(2);
    assert_eq!(uploads(&events, e)[1], n * (42 + 6 * (l2 + 1) * (l2 + 1)));
    let w = t.app.world();
    assert_eq!(*w.get::<Visibility>(e).unwrap(), Visibility::Hidden);
    let gpu = w.resource::<ftgsxr_render::SplatGpu>().device();
    assert_eq!(gpu.model_bytes_in_use(), uploads(&events, e)[1]);
    // §1.12: the new tier's memory budget, 1.5 GiB on a Quest tier.
    assert_eq!(gpu.memory_budget().bytes, 1_610_612_736);
    assert!(t.settle_pipelines());
}

// Q-238, Q-239: the new tier's budget applies to the reloads; a refused instance is
// unloaded with §1.15's text, the others stay.
#[test]
#[ignore = "gpu"]
fn gpu_live_tier_switch_leaves_a_model_the_new_budget_refuses_unloaded() {
    let _layer = ftgsxr_testkit::layer_messages(); // R-16, Q-105
    let mut t = TestApp::gpu("live_budget");
    let events = record_events(&mut t.app);
    let toasts = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let seen = toasts.clone();
    t.app.add_systems(
        Update,
        move |mut r: MessageReader<ftgsxr_app::LibraryFailure>| {
            seen.lock().unwrap().extend(r.read().cloned());
        },
    );
    for p in ["P1", "P2"] {
        let path = ftgsxr_testkit::fixtures_dir().join(format!("probes/{p}.safetensors"));
        t.command(ftgsxr_app::LibraryCommand::LoadRef(
            ftgsxr_app::ModelRef::File(path),
        ));
    }
    let instances = |app: &mut App| {
        let mut q = app
            .world_mut()
            .query_filtered::<Entity, With<ftgsxr_app::LoadedModel>>();
        q.iter(app.world()).collect::<Vec<_>>()
    };
    assert!(
        t.until_for(std::time::Duration::from_secs(30), |app| {
            let all = instances(app);
            all.len() == 2 && all.iter().all(|e| uploads(&events, *e).len() == 1)
        }),
        "two uploads"
    );
    let all = instances(&mut t.app);
    let rows: Vec<(Entity, u64)> = all
        .iter()
        .map(|e| {
            (
                *e,
                t.app
                    .world()
                    .get::<ftgsxr_app::LoadedModel>(*e)
                    .unwrap()
                    .info
                    .n,
            )
        })
        .collect();
    let larger = all.iter().map(|e| uploads(&events, *e)[0]).max().unwrap();
    // The same layout as `desktop`, with a budget that holds the larger model alone: the
    // reloads arrive in either order, and whichever comes second is refused.
    let mut tiny = QualityTier::desktop();
    tiny.name = "tiny".into();
    tiny.memory_budget_bytes = larger;
    *t.app.world_mut().resource_mut::<SplatTierState>() = SplatTierState::of(&tiny);
    let failed = |events: &std::sync::Mutex<Vec<ftgsxr_app::SplatEvent>>| {
        events
            .lock()
            .unwrap()
            .iter()
            .filter_map(|e| match e {
                ftgsxr_app::SplatEvent::Failed { entity, error } => Some((*entity, error.clone())),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    assert!(
        t.until_for(std::time::Duration::from_secs(30), |app| {
            instances(app).len() == 1 && !failed(&events).is_empty()
        }),
        "one reload refused"
    );
    let refused = failed(&events);
    assert_eq!(refused.len(), 1);
    assert!(refused[0].1.starts_with("E-MEM-BUDGET"), "{}", refused[0].1);
    // §1.15's text, and with no library entry a `LibraryFailure` (Q-173) carrying the
    // reduce-to-budget count of the model refused (Q-117).
    let n = rows.iter().find(|(e, _)| *e == refused[0].0).unwrap().1;
    // The failure is a message, read the frame after it is written.
    t.frames(2);
    let toasts = toasts.lock().unwrap().clone();
    assert_eq!(toasts.len(), 1);
    let text = &toasts[0].text;
    assert!(text.starts_with("E-MEM-BUDGET: Needs "), "{text}");
    assert!(
        text.contains(" MB; reduce to budget or choose a tier") && text.contains(" MB, budget "),
        "{text}"
    );
    assert!(text.ends_with(&format!(" (N = {n})")), "{text}");
    assert_eq!(toasts[0].code, "E-MEM-BUDGET");
    let kept = instances(&mut t.app)[0];
    assert_ne!(kept, refused[0].0);
    assert!(
        t.until_for(std::time::Duration::from_secs(10), |_| uploads(
            &events, kept
        )
        .len()
            == 2),
        "the kept instance's reload"
    );
    let gpu = t.app.world().resource::<ftgsxr_render::SplatGpu>().device();
    assert_eq!(gpu.memory_budget().bytes, larger);
    assert!(t.settle_pipelines());
}

/// `desktop` under another name with a budget of `bytes`.
fn desktop_with_budget(name: &str, bytes: u64) -> QualityTier {
    let mut tier = QualityTier::desktop();
    tier.name = name.into();
    tier.memory_budget_bytes = bytes;
    tier
}

/// Q-158's reduce-to-budget count for P1 alone on a device of `budget` bytes: the budget
/// over the static bytes of one primitive under `desktop`'s layout at its SH cap, floored.
fn p1_count(budget: u64) -> u64 {
    let (model, _) = ftgsxr_testkit::probe("P1");
    let tier = QualityTier::desktop();
    let opts = ftgsxr_core::GpuLayoutOptions {
        sh_cap: Some(SplatTierState::of(&tier).sh_cap),
        ..tier.layout
    };
    budget / ftgsxr_core::memory_estimate(1, model.sh_degree, &opts, false).static_bytes
}

/// P1's rows and the static bytes all of them take under `desktop`'s layout at its SH cap,
/// as Q-158's count reckons them.
fn p1_full_bytes() -> (u64, u64) {
    let (model, _) = ftgsxr_testkit::probe("P1");
    let tier = QualityTier::desktop();
    let opts = ftgsxr_core::GpuLayoutOptions {
        sh_cap: Some(SplatTierState::of(&tier).sh_cap),
        ..tier.layout
    };
    let n = model.len() as u64;
    let bytes = ftgsxr_core::memory_estimate(n, model.sh_degree, &opts, false).static_bytes;
    (n, bytes)
}

/// Lists P1 in the default `dir` source and loads it with the reduce option; its instance.
fn load_p1_reduced(t: &mut TestApp) -> Entity {
    use ftgsxr_app::{EntryState, Library, LibraryCommand};
    let models = t.dir.join("models");
    std::fs::create_dir_all(&models).unwrap();
    std::fs::copy(
        ftgsxr_testkit::fixtures_dir().join("probes/P1.safetensors"),
        models.join("P1.safetensors"),
    )
    .unwrap();
    t.command(LibraryCommand::Refresh(0));
    assert!(
        t.until_for(std::time::Duration::from_secs(30), |app| {
            app.world()
                .resource::<Library>()
                .entries
                .first()
                .is_some_and(|e| e.info.is_some())
        }),
        "P1 listed"
    );
    t.command(LibraryCommand::Load {
        entry: 0,
        reduce: true,
    });
    let mut found = None;
    assert!(
        t.until_for(std::time::Duration::from_secs(30), |app| {
            if let EntryState::Loaded(e) = app.world().resource::<Library>().entries[0].state {
                found = Some(e);
            }
            found.is_some()
        }),
        "P1 loads"
    );
    found.unwrap()
}

/// The instance's rows and whether it is labelled reduced.
fn rows_of(app: &App, e: Entity) -> (u64, bool) {
    let m = app.world().get::<ftgsxr_app::LoadedModel>(e).unwrap();
    (m.info.n, m.reduced)
}

/// Switches the running tier to `tier` and waits for `e`'s reload to upload and its entry
/// to be loaded again.
fn switch_and_reload(
    t: &mut TestApp,
    events: &std::sync::Mutex<Vec<ftgsxr_app::SplatEvent>>,
    e: Entity,
    tier: &QualityTier,
) {
    *t.app.world_mut().resource_mut::<SplatTierState>() = SplatTierState::of(tier);
    assert!(
        t.until_for(std::time::Duration::from_secs(30), |app| {
            uploads(events, e).len() == 2
                && app.world().resource::<ftgsxr_app::Library>().entries[0].state
                    == ftgsxr_app::EntryState::Loaded(e)
        }),
        "the reload's upload"
    );
    let failed = events
        .lock()
        .unwrap()
        .iter()
        .any(|x| matches!(x, ftgsxr_app::SplatEvent::Failed { .. }));
    assert!(!failed, "no reload refused");
}

// Q-238 (#395): a reduced instance reloads with the new tier's reduce-to-budget count; a
// budget that holds every row loads them all, no longer labelled reduced.
#[test]
#[ignore = "gpu"]
fn gpu_live_tier_switch_grows_a_reduced_instance_under_a_larger_budget() {
    let _layer = ftgsxr_testkit::layer_messages(); // R-16, Q-105
    let (n, full) = p1_full_bytes();
    let setup = Setup {
        tier: desktop_with_budget("small", full / 2),
        ..Setup::default()
    };
    let mut t = TestApp::gpu_with("live_reduce_grow", setup);
    let events = record_events(&mut t.app);
    let e = load_p1_reduced(&mut t);
    let (k, reduced) = rows_of(&t.app, e);
    assert!(reduced && k < n, "loaded reduced: {k} of {n}");
    assert_eq!(k, p1_count(full / 2));
    switch_and_reload(&mut t, &events, e, &desktop_with_budget("big", full * 4));
    assert_eq!(rows_of(&t.app, e), (n, false));
    assert!(t.settle_pipelines());
}

// Q-238 (#395): an instance loaded with the reduce option whose rows all fitted is reduced,
// not refused, by a tier whose budget holds fewer of them.
#[test]
#[ignore = "gpu"]
fn gpu_live_tier_switch_reduces_instead_of_refusing() {
    let _layer = ftgsxr_testkit::layer_messages(); // R-16, Q-105
    let (n, full) = p1_full_bytes();
    let setup = Setup {
        tier: desktop_with_budget("big", full * 4),
        ..Setup::default()
    };
    let mut t = TestApp::gpu_with("live_reduce_shrink", setup);
    let events = record_events(&mut t.app);
    let e = load_p1_reduced(&mut t);
    assert_eq!(rows_of(&t.app, e), (n, false), "every row fits");
    switch_and_reload(&mut t, &events, e, &desktop_with_budget("small", full / 2));
    let (k, reduced) = rows_of(&t.app, e);
    assert!(reduced && k < n, "reloaded reduced: {k} of {n}");
    // Q-158 under the new tier, its device holding no other model.
    assert_eq!(k, p1_count(full / 2));
    assert!(t.settle_pipelines());
}

// Q-240: a pick is also saved, so the next start runs it.
// Q-240: a pick is also saved, so the next start runs it. The pick writes `[general]
// tier` (`governor::tests::live_tier_switch_follows_the_settings`); this writes it as the
// pick does and checks the debounce saves it, while the write alone switches nothing.
#[test]
fn live_tier_switch_is_saved_for_the_next_start() {
    let target = support::scratch("live_saved_target").join("s.toml");
    let given = target.clone();
    let mut t = TestApp::with("live_saved", Setup::default(), move |app| {
        app.insert_resource(ftgsxr_app::SettingsPath(given));
    });
    t.frames(2);
    t.app
        .world_mut()
        .resource_mut::<SettingsRes>()
        .0
        .general
        .tier = "quest-high".into();
    // The settings are written within a second (§1.11; 60 frames at 60 Hz).
    let saved = |p: &std::path::Path| {
        p.exists()
            && ftgsxr_app::load_settings(p)
                .expect("the settings read")
                .0
                .general
                .tier
                == "quest-high"
    };
    let mut written = false;
    for _ in 0..61 {
        t.frames(1);
        if saved(&target) {
            written = true;
            break;
        }
    }
    assert!(written, "the written tier is saved");
    assert_eq!(running(&t.app), "desktop");
}
