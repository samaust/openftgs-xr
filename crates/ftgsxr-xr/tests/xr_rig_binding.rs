// SPDX-License-Identifier: Apache-2.0

//! PRD #452's added `xr` test, the rig binding (#461; source of truth §1.8 Locomotion and
//! E12: the left stick moves the rig, and the tracked eyes are in the rig's space; §2.1
//! `PlayerRig`, "the XR tracking space"): on Monado, with the player's `AppPlugin`, a
//! `Locomote` intent moves the rig and the eye cameras with it, the vignette shown on
//! each eye's own quad meanwhile (Q-300). Spike check 4 (Q-124):
//! the binding survives the session re-creation a render-scale change makes.
//! Prerequisites: `xr_support`'s.

use crate::xr_support;

use std::time::{Duration, Instant};

use bevy::camera::visibility::RenderLayers;
use bevy::prelude::*;
use bevy_mod_xr::camera::XrCamera;
use bevy_mod_xr::session::{XrRootTransform, XrState, XrTrackingRoot};
use ftgsxr_app::{Intent, PlayerRig};
use ftgsxr_render::SortMode;
use ftgsxr_testkit::scratch_dir;
use ftgsxr_xr::{DisplaySettings, XrPlugin};
use tracing::Level;
use xr_support::{Monado, capture_log, lines, swapchain_sizes, xr_player_app};

const TOL: f32 = 1e-4;

fn player(name: &str) -> App {
    let root = scratch_dir(env!("CARGO_TARGET_TMPDIR"), "xr_rig_binding", name);
    xr_player_app(
        XrPlugin {
            passthrough: false,
            hands: false,
            sort: SortMode::PerView,
        },
        &root,
    )
}

/// Runs until the session runs with two eyes, then a few frames more.
fn run_until_two_eyes(app: &mut App) {
    let start = Instant::now();
    while app.world().get_resource::<XrState>() != Some(&XrState::Running) || eyes(app).len() < 2 {
        assert!(
            start.elapsed() < Duration::from_secs(20),
            "the session runs with two eyes within 20 s"
        );
        app.update();
    }
    for _ in 0..5 {
        app.update();
    }
}

fn eyes(app: &mut App) -> Vec<Entity> {
    app.world_mut()
        .query_filtered::<Entity, With<XrCamera>>()
        .iter(app.world())
        .collect()
}

fn single<F: bevy::ecs::query::QueryFilter>(app: &mut App) -> Entity {
    app.world_mut()
        .query_filtered::<Entity, F>()
        .single(app.world())
        .expect("exactly one")
}

fn close(a: &GlobalTransform, b: &GlobalTransform) -> bool {
    let (sa, ra, ta) = a.to_scale_rotation_translation();
    let (sb, rb, tb) = b.to_scale_rotation_translation();
    (ta - tb).length() <= TOL
        && (sa - sb).length() <= TOL
        && (ra * Vec3::X - rb * Vec3::X).length() <= TOL
        && (ra * Vec3::Z - rb * Vec3::Z).length() <= TOL
}

/// The binding's checks: the root is the rig's child, every eye the root's; then 0.5 s
/// of a forward `Locomote` moves the rig, and each eye's world pose is the rig's composed
/// with the root's and the eye's located pose, as is what the render world receives.
fn assert_locomote_moves_the_eyes_with_the_rig(app: &mut App) {
    let rig = single::<With<PlayerRig>>(app);
    let root = single::<With<XrTrackingRoot>>(app);
    assert_eq!(
        app.world().get::<ChildOf>(root).map(ChildOf::parent),
        Some(rig),
        "the tracking root is the rig's child"
    );
    let eyes = eyes(app);
    assert_eq!(eyes.len(), 2);
    for &eye in &eyes {
        assert_eq!(
            app.world().get::<ChildOf>(eye).map(ChildOf::parent),
            Some(root),
            "each eye is the root's child"
        );
    }
    let rig_before = app
        .world()
        .get::<GlobalTransform>(rig)
        .unwrap()
        .translation();
    let start = Instant::now();
    while start.elapsed() < Duration::from_millis(500) {
        app.world_mut().write_message(Intent::Locomote {
            move_xz: Vec2::new(0.0, 1.0),
            turn: 0.0,
        });
        app.update();
    }
    // The vignette (§1.8 Locomotion, Q-269, Q-300) is up during the smooth move: each
    // eye, on its own layer, has a shown quad among its two mesh children.
    for (i, &eye) in eyes.iter().enumerate() {
        let index = app.world().get::<XrCamera>(eye).unwrap().0 as usize;
        let layers = app
            .world()
            .get::<RenderLayers>(eye)
            .expect("the eye's layers");
        assert!(
            layers.intersects(&RenderLayers::layer(0))
                && layers.intersects(&RenderLayers::layer(1 + index))
        );
        let quads: Vec<Entity> = app
            .world()
            .get::<Children>(eye)
            .map(|c| c.iter().collect())
            .unwrap_or_default();
        let quads: Vec<Entity> = quads
            .into_iter()
            .filter(|&q| app.world().get::<Mesh3d>(q).is_some())
            .collect();
        assert_eq!(quads.len(), 2, "eye {i}: the vignette and the fade quads");
        assert!(
            quads
                .iter()
                .any(|&q| app.world().get::<Visibility>(q) == Some(&Visibility::Inherited)),
            "eye {i}: the vignette is shown during the move"
        );
        for &q in &quads {
            assert_eq!(
                app.world().get::<RenderLayers>(q),
                Some(&RenderLayers::layer(1 + index)),
                "eye {i}: its quads only on its own layer"
            );
        }
    }
    // One frame for the propagation.
    app.update();
    let rig_g = *app.world().get::<GlobalTransform>(rig).unwrap();
    let moved = (rig_g.translation() - rig_before).length();
    assert!(moved > 0.05, "the rig moved: {moved} m");
    let root_local = *app.world().get::<Transform>(root).unwrap();
    let root_g = *app.world().get::<GlobalTransform>(root).unwrap();
    assert!(close(&root_g, &rig_g.mul_transform(root_local)));
    assert!(
        close(&app.world().resource::<XrRootTransform>().0, &root_g),
        "the render world's root is the moved one"
    );
    for &eye in &eyes {
        let local = *app.world().get::<Transform>(eye).unwrap();
        let g = *app.world().get::<GlobalTransform>(eye).unwrap();
        assert!(
            close(&g, &rig_g.mul_transform(root_local).mul_transform(local)),
            "the eye follows the rig: {g:?}"
        );
    }
}

fn assert_no_error() {
    let errors: Vec<_> = lines()
        .into_iter()
        .filter(|l| l.level == Level::ERROR)
        .collect();
    assert!(errors.is_empty(), "no error-level log: {errors:?}");
}

#[test]
#[ignore = "xr"]
fn xr_rig_binding_a_locomote_moves_the_eye_cameras_with_the_rig() {
    capture_log();
    let _monado = Monado::start();
    let mut app = player("locomote");
    run_until_two_eyes(&mut app);
    assert_locomote_moves_the_eyes_with_the_rig(&mut app);
    assert_no_error();
}

// Spike check 4 (PRD #452, Q-124): the binding holds through a session re-creation.
#[test]
#[ignore = "xr"]
fn xr_rig_binding_survives_the_session_re_creation() {
    capture_log();
    let _monado = Monado::start();
    let mut app = player("recreation");
    run_until_two_eyes(&mut app);
    app.world_mut()
        .resource_mut::<DisplaySettings>()
        .render_scale = 0.8;
    let start = Instant::now();
    while swapchain_sizes().len() < 2
        || app.world().get_resource::<XrState>() != Some(&XrState::Running)
    {
        assert!(
            start.elapsed() < Duration::from_secs(30),
            "the session is re-created within 30 s: {:?}",
            swapchain_sizes()
        );
        app.update();
    }
    run_until_two_eyes(&mut app);
    assert_locomote_moves_the_eyes_with_the_rig(&mut app);
    assert_no_error();
}
