// SPDX-License-Identifier: Apache-2.0

//! The live tier switch (#373; §1.7 Panels, §1.12, D-37; Q-179, Q-233, Q-234, Q-235): a
//! change of `[general] tier` while the app runs switches the running tier, the governor
//! follows Q-179's rule, and the loaded instances are reloaded with the new tier's layout
//! on their own entities, without a restart.

mod support;

use bevy::prelude::*;
use ftgsxr_app::{Governor, QualityTier, SettingsRes};
use ftgsxr_render::{SplatModel, SplatTierState};
use support::{Setup, TestApp};

/// The running tier's name.
fn running(app: &App) -> String {
    app.world().resource::<SplatTierState>().tier.name.clone()
}

/// Sets `[general] tier` as the Display or Settings tab's choice does.
fn pick(app: &mut App, tier: &str) {
    app.world_mut().resource_mut::<SettingsRes>().0.general.tier = tier.into();
}

// Q-235, Q-179
#[test]
fn live_tier_switch_follows_the_settings() {
    let mut t = TestApp::new("live_follow");
    t.frames(2);
    assert_eq!(running(&t.app), "desktop");
    assert!(t.app.world().resource::<Governor>().enabled);
    pick(&mut t.app, "quest-low");
    t.frames(2);
    // §1.12's table: `quest-low` runs at render scale 0.8, SH cap 1; the knobs start at
    // the new tier's own values.
    let s = t.app.world().resource::<SplatTierState>().clone();
    assert_eq!(s.tier, QualityTier::quest_low());
    assert_eq!((s.render_scale, s.sh_cap), (0.8, 1));
    // Q-179: a tier chosen by hand turns the governor off.
    assert!(!t.app.world().resource::<Governor>().enabled);
    // Back to the build's default with `[general] governor` on: back on.
    pick(&mut t.app, "desktop");
    t.frames(2);
    assert_eq!(running(&t.app), "desktop");
    assert!(t.app.world().resource::<Governor>().enabled);
    // A name no tier has leaves the running tier alone.
    pick(&mut t.app, "no-such-tier");
    t.frames(2);
    assert_eq!(running(&t.app), "desktop");
}

// Q-235: `--tier` at startup holds until the user picks a tier.
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

// Q-233: the reload happens on the instance's own entity, which keeps its state.
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
    pick(&mut t.app, "quest-balanced");
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
            ftgsxr_app::SplatEvent::Uploaded { entity: x, bytes } if *x == entity => Some(*bytes),
            _ => None,
        })
        .collect()
}

// Q-233, Q-234: the reload is packed with the new tier's layout.
#[test]
#[ignore = "gpu"]
fn gpu_live_tier_switch_reloads_with_the_new_layout() {
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
    // A hidden instance stays hidden through the reload (Q-233).
    t.app.world_mut().entity_mut(e).insert(Visibility::Hidden);
    pick(&mut t.app, "quest-balanced");
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

// Q-233, Q-234: the new tier's budget applies to the reloads; a refused instance is
// unloaded with §1.15's text, the others stay.
#[test]
#[ignore = "gpu"]
fn gpu_live_tier_switch_unloads_a_model_the_new_budget_refuses() {
    let _layer = ftgsxr_testkit::layer_messages(); // R-16, Q-105
    let mut t = TestApp::gpu("live_budget");
    let events = record_events(&mut t.app);
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
