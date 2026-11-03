// SPDX-License-Identifier: Apache-2.0

//! The live tier switch (#373; §1.7 Panels, §1.12, D-37; Q-179, Q-229, Q-230, Q-231): a
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

// Q-231, Q-179
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

// Q-231: `--tier` at startup holds until the user picks a tier.
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

// Q-229: the reload happens on the instance's own entity, which keeps its state.
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
