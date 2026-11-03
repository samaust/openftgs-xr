// SPDX-License-Identifier: Apache-2.0

//! I-11 through bevy_egui's render-to-image path (§1.8, D-21; PRD #275's spike check 2,
//! `gpu_`): a second egui context renders to `WorldPanel.image`; a ray at known quad
//! coordinates moves its pointer to the expected pixel (1 px) and a select press and
//! release toggle a test button. The `cpu` clause: `panel_hit` maps a ray at known quad
//! coordinates to the expected pixel (1 px), and pointer events at that pixel click a
//! button in a second egui context.

mod support;

use bevy::asset::{Assets, RenderAssetUsages};
use bevy::camera::{Camera, Camera3d, RenderTarget};
use bevy::ecs::prelude::*;
use bevy::ecs::schedule::ScheduleLabel;
use bevy::image::Image;
use bevy::math::{Quat, UVec2, Vec2, Vec3};
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::transform::components::{GlobalTransform, Transform};
use bevy_egui::{EguiContext, EguiGlobalSettings, EguiSchedule, egui};
use ftgsxr_app::{Hand, Intent, UiSurface, WorldPanel, panel_hit};
use ftgsxr_core::Pose;
use ftgsxr_testkit::layer_messages;
use support::TestApp;

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
struct PanelPass;

#[derive(Resource, Default)]
struct Toggled(bool);

const BUTTON: egui::Rect = egui::Rect {
    min: egui::pos2(600.0, 300.0),
    max: egui::pos2(680.0, 340.0),
};

fn panel_ui(mut ctx: Query<&mut EguiContext, With<EguiSchedule>>, mut toggled: ResMut<Toggled>) {
    let Ok(mut ctx) = ctx.single_mut() else {
        return;
    };
    let ctx = ctx.get_mut();
    egui::Area::new("test".into())
        .fixed_pos(egui::Pos2::ZERO)
        .show(ctx, |ui| {
            if ui.allocate_rect(BUTTON, egui::Sense::click()).clicked() {
                toggled.0 = !toggled.0;
            }
        });
}

#[test]
#[ignore = "gpu"]
fn gpu_i11_a_ray_clicks_a_button_in_the_panel_context() {
    let _layer = layer_messages(); // R-16, Q-105
    let mut t = TestApp::gpu("i11");
    t.app
        .world_mut()
        .resource_mut::<EguiGlobalSettings>()
        .auto_create_primary_context = false;
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
    t.app
        .init_resource::<Toggled>()
        .init_schedule(PanelPass)
        .add_systems(PanelPass, panel_ui);
    t.app.world_mut().spawn((
        Camera3d::default(),
        Camera::default(),
        RenderTarget::Image(image.clone().into()),
        EguiSchedule::new(PanelPass),
    ));
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
    t.frames(3);
    // The ray aims at pixel (640.5, 320.25).
    let target = Vec2::new(640.5, 320.25);
    let local = Vec2::new(
        (target.x / size_px.x as f32 - 0.5) * size_m.x,
        (0.5 - target.y / size_px.y as f32) * size_m.y,
    );
    let origin = Vec3::new(0.0, 1.6, 0.0);
    let dir = (pose.translation + pose.rotation * local.extend(0.0) - origin).normalize();
    let aim = Intent::Pointer {
        hand: Hand::Right,
        origin,
        dir,
    };
    for pressed in [None, Some(true), Some(false), None] {
        t.app.world_mut().write_message(aim);
        if let Some(pressed) = pressed {
            t.app.world_mut().write_message(Intent::Select {
                hand: Hand::Right,
                pressed,
            });
        }
        t.frames(2);
    }
    let mut q = t
        .app
        .world_mut()
        .query_filtered::<&mut EguiContext, With<EguiSchedule>>();
    let p = q
        .single_mut(t.app.world_mut())
        .unwrap()
        .get_mut()
        .input(|i| i.pointer.latest_pos())
        .expect("a pointer position");
    assert!(
        (p.x - target.x).abs() <= 1.0 && (p.y - target.y).abs() <= 1.0,
        "{p:?}"
    );
    assert!(
        t.app.world().resource::<Toggled>().0,
        "the click toggled the button"
    );
}

/// I-11 (§1.8, `cpu`): a ray at known quad coordinates hits the expected pixel (1 px), and
/// pointer events there move a second egui context's pointer to it and click a button.
#[test]
fn i11_a_ray_on_the_quad_points_and_clicks_in_a_second_context() {
    let size_px = UVec2::new(1024, 768);
    let size_m = Vec2::new(0.56, 0.42);
    let panel = Pose {
        translation: Vec3::new(0.2, 1.4, -1.0),
        rotation: Quat::from_rotation_y(0.3),
        scale: 1.0,
    };
    let target_px = Vec2::new(640.5, 320.25);
    let local = Vec2::new(
        (target_px.x / size_px.x as f32 - 0.5) * size_m.x,
        (0.5 - target_px.y / size_px.y as f32) * size_m.y,
    );
    let origin = Vec3::new(0.0, 1.6, 0.0);
    let dir = (panel.translation + panel.rotation * local.extend(0.0) - origin).normalize();
    let hit = panel_hit(origin, dir, panel, size_m, size_px).expect("a hit");
    assert!((hit - target_px).abs().max_element() <= 1.0, "{hit}");

    let ctx = egui::Context::default();
    let pos = egui::pos2(hit.x, hit.y);
    let button = |pressed| egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    };
    let mut toggled = false;
    let mut pointer = None;
    for events in [
        vec![egui::Event::PointerMoved(pos)],
        vec![button(true)],
        vec![button(false)],
        vec![],
    ] {
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(size_px.x as f32, size_px.y as f32),
            )),
            events,
            ..Default::default()
        };
        let mut output = ctx.run_ui(input, |ui| {
            if ui.allocate_rect(BUTTON, egui::Sense::click()).clicked() {
                toggled = !toggled;
            }
        });
        output.textures_delta.clear();
        pointer = ctx.input(|i| i.pointer.latest_pos());
    }
    let p = pointer.expect("a pointer position");
    assert!(
        (p.x - target_px.x).abs() <= 1.0 && (p.y - target_px.y).abs() <= 1.0,
        "{p:?}"
    );
    assert!(toggled, "the click toggled the button");
}
