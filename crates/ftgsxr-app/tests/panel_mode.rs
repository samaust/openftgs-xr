// SPDX-License-Identifier: Apache-2.0

//! Panel mode (§1.8 User interface, D-21; PRD #452, `gpu_`): with the `Headset` marker
//! and `UiSurface::Panel` set, the tabs draw into the panel's texture through its own
//! egui context, and a ray's `PanelPointer` click on the Playback tab's play button
//! toggles the clock. The texts' places are read from the panel context's own output,
//! so the test finds the buttons as drawn, then clicks them through the ray (I-11's
//! harness).

use crate::support;

use bevy::app::PostUpdate;
use bevy::asset::{Assets, RenderAssetUsages};
use bevy::camera::visibility::Visibility;
use bevy::camera::{Camera, RenderTarget};
use bevy::ecs::prelude::*;
use bevy::image::Image;
use bevy::math::{Quat, UVec2, Vec2, Vec3};
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::transform::components::{GlobalTransform, Transform};
use bevy_egui::{EguiFullOutput, EguiPostUpdateSet, PrimaryEguiContext, egui};
use ftgsxr_app::{Hand, Headset, PanelPointer, Playback, UiSurface, WorldPanel, panel_hit};
use ftgsxr_core::{PlayState, Pose};
use ftgsxr_testkit::layer_messages;
use support::TestApp;

/// The texts the panel context drew last, each with its centre in the texture's pixels,
/// and the context's pixels per point.
#[derive(Resource, Default)]
struct Drawn(Vec<(String, Vec2)>, f32);

fn record_texts(
    contexts: Query<(&EguiFullOutput, &RenderTarget)>,
    panels: Query<&WorldPanel>,
    mut drawn: ResMut<Drawn>,
) {
    let Some(image) = panels.iter().next().map(|p| p.image.clone()) else {
        return;
    };
    for (full, target) in &contexts {
        let RenderTarget::Image(i) = target else {
            continue;
        };
        let Some(full) = full.0.as_ref() else {
            continue;
        };
        if i.handle != image {
            continue;
        }
        let ppp = full.pixels_per_point;
        fn walk(s: &egui::Shape, ppp: f32, texts: &mut Vec<(String, Vec2)>) {
            match s {
                egui::Shape::Text(t) => {
                    let c = t.pos + t.galley.size() / 2.0;
                    texts.push((t.galley.text().to_string(), Vec2::new(c.x, c.y) * ppp));
                }
                egui::Shape::Vec(v) => v.iter().for_each(|s| walk(s, ppp, texts)),
                _ => {}
            }
        }
        let mut texts = Vec::new();
        full.shapes
            .iter()
            .for_each(|c| walk(&c.shape, ppp, &mut texts));
        drawn.0 = texts;
        drawn.1 = ppp;
    }
}

#[test]
#[ignore = "gpu"]
fn gpu_panel_mode_a_ray_click_on_the_playback_tabs_play_button_toggles_the_clock() {
    let _layer = layer_messages(); // R-16, Q-105
    let mut t = TestApp::gpu_before("panel_mode", |app| {
        app.insert_resource(Headset);
    });
    let size_px = UVec2::new(1024, 768);
    let size_m = Vec2::new(0.56, 0.42);
    let mut image = Image::new_fill(
        Extent3d {
            width: size_px.x,
            height: size_px.y,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage |= TextureUsages::RENDER_ATTACHMENT;
    let image = t.app.world_mut().resource_mut::<Assets<Image>>().add(image);
    let pose = Transform::from_xyz(0.2, 1.4, -1.0).with_rotation(Quat::from_rotation_y(0.3));
    let panel = t
        .app
        .world_mut()
        .spawn((WorldPanel { image }, pose, GlobalTransform::from(pose)))
        .id();
    t.app.insert_resource(UiSurface::Panel {
        entity: panel,
        size_px,
        size_m,
    });
    t.app.init_resource::<Drawn>().add_systems(
        PostUpdate,
        record_texts
            .after(EguiPostUpdateSet::EndPass)
            .before(EguiPostUpdateSet::ProcessOutput),
    );
    t.frames(5);
    // No primary context latched onto a camera on the headset.
    let mut primary = t
        .app
        .world_mut()
        .query_filtered::<(), With<PrimaryEguiContext>>();
    assert_eq!(primary.iter(t.app.world()).count(), 0);
    // Q-267: the panel's zoom at the default `ui_scale`, so a pixel is not a point.
    let ppp = t.app.world().resource::<Drawn>().1;
    assert!((ppp - 1.75).abs() < 1e-4, "{ppp}");

    let at = Pose {
        translation: pose.translation,
        rotation: pose.rotation,
        scale: 1.0,
    };
    // A ray from the head to the text's pixel; the pointer where `panel_hit` puts it.
    let click = |t: &mut TestApp, label: &str| {
        let target = t
            .app
            .world()
            .resource::<Drawn>()
            .0
            .iter()
            .find(|(s, _)| s == label)
            .map(|(_, p)| *p)
            .unwrap_or_else(|| {
                panic!(
                    "{label} is drawn: {:?}",
                    t.app.world().resource::<Drawn>().0
                )
            });
        assert!(
            target.x < size_px.x as f32 && target.y < size_px.y as f32,
            "{label} inside the texture: {target}"
        );
        let local = Vec2::new(
            (target.x / size_px.x as f32 - 0.5) * size_m.x,
            (0.5 - target.y / size_px.y as f32) * size_m.y,
        );
        let origin = Vec3::new(0.0, 1.6, 0.0);
        let dir = (pose.translation + pose.rotation * local.extend(0.0) - origin).normalize();
        let pixel = panel_hit(origin, dir, at, size_m, size_px).expect("the ray hits the quad");
        for select in [vec![], vec![true], vec![], vec![false], vec![], vec![]] {
            t.app.world_mut().write_message(PanelPointer {
                hand: Hand::Right,
                panel,
                pixel,
                select,
                grab: vec![],
                scroll_px: 0.0,
            });
            t.frames(1);
        }
    };
    click(&mut t, "Playback");
    let before = t.app.world().resource::<Playback>().clock.state();
    let play = match before {
        PlayState::Paused => "Play",
        PlayState::Playing => "Pause",
    };
    click(&mut t, play);
    let after = t.app.world().resource::<Playback>().clock.state();
    assert_ne!(after, before, "the click toggled the clock");
}

/// #269, Q-324: a hidden quad's panel camera is inactive and its egui pass draws
/// nothing, so the headset spends no frame time on a panel no one sees; the frame the
/// quad shows, its camera is active and the tab row draws. The quad starts hidden, as
/// `xr::panel` spawns it (§1.8: hidden until summoned), so its first summon is checked.
#[test]
#[ignore = "gpu"]
fn gpu_panel_mode_a_hidden_quad_idles_its_camera_until_shown() {
    let _layer = layer_messages(); // R-16, Q-105
    let mut t = TestApp::gpu_before("panel_mode_hidden", |app| {
        app.insert_resource(Headset);
    });
    let size_px = UVec2::new(1024, 768);
    let mut image = Image::new_fill(
        Extent3d {
            width: size_px.x,
            height: size_px.y,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage |= TextureUsages::RENDER_ATTACHMENT;
    let image = t.app.world_mut().resource_mut::<Assets<Image>>().add(image);
    let pose = Transform::from_xyz(0.0, 1.4, -1.0);
    let panel = t
        .app
        .world_mut()
        .spawn((
            WorldPanel {
                image: image.clone(),
            },
            pose,
            GlobalTransform::from(pose),
            Visibility::Hidden,
        ))
        .id();
    t.app.insert_resource(UiSurface::Panel {
        entity: panel,
        size_px,
        size_m: Vec2::new(0.56, 0.42),
    });
    t.app.init_resource::<Drawn>().add_systems(
        PostUpdate,
        record_texts
            .after(EguiPostUpdateSet::EndPass)
            .before(EguiPostUpdateSet::ProcessOutput),
    );
    // The panel camera: the one rendering to the quad's image.
    let active = |t: &mut TestApp| {
        let mut q = t.app.world_mut().query::<(&Camera, &RenderTarget)>();
        let found: Vec<bool> = q
            .iter(t.app.world())
            .filter(|(_, r)| matches!(r, RenderTarget::Image(i) if i.handle == image))
            .map(|(c, _)| c.is_active)
            .collect();
        assert_eq!(found.len(), 1, "one camera renders to the quad's image");
        found[0]
    };
    let drawn = |t: &TestApp, label: &str| {
        t.app
            .world()
            .resource::<Drawn>()
            .0
            .iter()
            .any(|(s, _)| s == label)
    };
    let show = |t: &mut TestApp, v: Visibility| {
        *t.app
            .world_mut()
            .get_mut::<Visibility>(panel)
            .expect("the quad") = v;
        t.frames(1);
    };
    let nothing_drawn = |t: &TestApp| t.app.world().resource::<Drawn>().0.is_empty();

    t.frames(5);
    assert!(!active(&mut t), "a hidden quad's camera is inactive");
    assert!(nothing_drawn(&t), "a hidden quad's pass draws nothing");

    show(&mut t, Visibility::Visible);
    assert!(
        active(&mut t),
        "the camera is active the frame the quad first shows"
    );
    assert!(
        drawn(&t, "Library"),
        "the tab row draws the frame the quad first shows: {:?}",
        t.app.world().resource::<Drawn>().0
    );

    show(&mut t, Visibility::Hidden);
    assert!(!active(&mut t), "a quad hidden again idles its camera");
    assert!(nothing_drawn(&t), "a quad hidden again draws nothing");
}
