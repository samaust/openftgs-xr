// SPDX-License-Identifier: Apache-2.0

//! R-17 (E16, `gpu` lane): on real GPU times, with `Governor::budget_override_ms` at
//! 0.1 every knob of the desktop ladder (Q-153; owner, 2026-11-02) steps down in E16's
//! counts and order, one step per 10 statistics blocks; at 1,000 they step back up, one
//! per 120.

mod support;

use std::sync::{Arc, Mutex};

use bevy::asset::RenderAssetUsages;
use bevy::camera::{Camera, Camera3d, PerspectiveProjection, Projection, RenderTarget};
use bevy::prelude::*;
use bevy::render::render_resource::{TextureFormat, TextureUsages};
use ftgsxr_app::{Governor, LibraryCommand, ModelRef, SplatCamera, SplatEvent};
use ftgsxr_render::{OutputSpace, SortMode, SplatStats, SplatTierState};
use ftgsxr_testkit::layer_messages;
use support::TestApp;

/// (render scale in tenths, SH cap).
type Knobs = (i32, u8);

/// The blocks seen, each knob change at the block count it came at, and the last knobs.
#[derive(Default)]
struct Record {
    blocks: u32,
    changes: Vec<(u32, Knobs)>,
    last: Option<Knobs>,
}

/// What the `Last` system records.
#[derive(Resource, Clone, Default)]
struct Seen(Arc<Mutex<Record>>);

fn record(stats: Res<SplatStats>, tier: Res<SplatTierState>, seen: Res<Seen>) {
    let mut s = seen.0.lock().unwrap();
    if stats.is_changed() && stats.0.is_some() {
        s.blocks += 1;
    }
    let now = ((tier.render_scale * 10.0).round() as i32, tier.sh_cap);
    if s.last != Some(now) {
        if s.last.is_some() {
            let block = s.blocks;
            s.changes.push((block, now));
        }
        s.last = Some(now);
    }
}

/// The knob changes since the last call, each with the block count it came at.
fn take(seen: &Seen) -> Vec<(u32, Knobs)> {
    std::mem::take(&mut seen.0.lock().unwrap().changes)
}

#[test]
#[ignore = "gpu"]
fn gpu_r17_the_desktop_knobs_step_down_and_back_up_on_gpu_times() {
    let _layer = layer_messages(); // R-16, Q-105
    let mut t = TestApp::gpu("r17");
    // Hold the governor until the model is drawn.
    t.app.world_mut().resource_mut::<Governor>().enabled = false;
    // No egui context on the test's camera: the test draws no UI, and an image target's
    // context has no window to hand its textures to.
    t.app
        .world_mut()
        .resource_mut::<bevy_egui::EguiGlobalSettings>()
        .auto_create_primary_context = false;
    let seen = Seen::default();
    t.app.insert_resource(seen.clone());
    t.app.add_systems(Last, record);
    let uploaded = Arc::new(Mutex::new(false));
    let up = uploaded.clone();
    t.app.add_observer(move |e: On<SplatEvent>| {
        if let SplatEvent::Failed { error, .. } = e.event() {
            panic!("the load failed: {error}");
        }
        if matches!(e.event(), SplatEvent::Uploaded { .. }) {
            *up.lock().unwrap() = true;
        }
    });

    let (w, h) = (640, 480);
    let mut image = Image::new_target_texture(w, h, TextureFormat::Rgba8UnormSrgb, None);
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    image.asset_usage = RenderAssetUsages::RENDER_WORLD;
    let target = t.app.world_mut().resource_mut::<Assets<Image>>().add(image);
    t.app.world_mut().spawn((
        Camera3d::default(),
        Camera::default(),
        RenderTarget::Image(target.into()),
        Projection::Perspective(PerspectiveProjection {
            near: 0.01,
            ..PerspectiveProjection::default()
        }),
        Transform::from_xyz(0.0, 0.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
        SplatCamera {
            sort: SortMode::PerView,
            output: OutputSpace::LinearOver { background: None },
        },
    ));
    // A load heavy enough that its GPU frame exceeds 0.105 ms: the probes take a few
    // microseconds on the reference GPU.
    let model = ftgsxr_testkit::random_model(17, 2_000_000, 1);
    let path = support::scratch("r17_model").join("r17.ftgs-if");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, ftgsxr_testkit::exchange_bytes(&model)).unwrap();
    drop(model);
    t.command(LibraryCommand::LoadRef(ModelRef::File(path)));
    assert!(
        t.until(600, |_| *uploaded.lock().unwrap()),
        "the model uploaded"
    );
    t.frames(10);
    let total = t
        .app
        .world()
        .resource::<SplatStats>()
        .0
        .expect("a block")
        .gpu_times_ms
        .total;
    assert!(
        total > 0.0,
        "the device writes timestamps (GpuTimes.total = {total})"
    );
    take(&seen);
    let start = seen.0.lock().unwrap().blocks;

    {
        let mut g = t.app.world_mut().resource_mut::<Governor>();
        g.enabled = true;
        g.budget_override_ms = Some(0.1);
    }
    let blocks = |seen: &Seen| seen.0.lock().unwrap().blocks;
    assert!(t.until(2000, |_| blocks(&seen) >= start + 80), "80 blocks");
    let down: Vec<_> = take(&seen)
        .into_iter()
        .map(|(b, k)| (b - start, k))
        .collect();
    assert_eq!(
        down,
        vec![
            (10, (9, 3)),
            (20, (8, 3)),
            (30, (7, 3)),
            (40, (6, 3)),
            (50, (6, 2)),
            (60, (6, 1)),
        ]
    );

    let start = blocks(&seen);
    t.app
        .world_mut()
        .resource_mut::<Governor>()
        .budget_override_ms = Some(1000.0);
    assert!(
        t.until(6000, |_| blocks(&seen) >= start + 120 * 6 + 60),
        "780 blocks"
    );
    let up: Vec<_> = take(&seen)
        .into_iter()
        .map(|(b, k)| (b - start, k))
        .collect();
    assert_eq!(
        up,
        vec![
            (120, (6, 2)),
            (240, (6, 3)),
            (360, (7, 3)),
            (480, (8, 3)),
            (600, (9, 3)),
            (720, (10, 3)),
        ]
    );
}
