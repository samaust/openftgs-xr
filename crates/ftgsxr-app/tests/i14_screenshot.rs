// SPDX-License-Identifier: Apache-2.0

//! I-14 (§1.7 Screenshots, `gpu_`): F12 writes a PNG with the metadata chunk (Q-150) at
//! `<screenshots>/<date>-<time>.png`, read back and checked against the loaded models,
//! the clock and the camera. The camera renders to an image: a window has no surface
//! without a display (PRD #275's spike check 4 and its fallback). The `cpu` test: a
//! request Bevy never answers ends with a warning (Q-T287.2).

mod support;

use std::sync::{Arc, Mutex};

use bevy::asset::RenderAssetUsages;
use bevy::camera::{Camera, Camera3d, PerspectiveProjection, Projection, RenderTarget};
use bevy::prelude::*;
use bevy::render::render_resource::{TextureFormat, TextureUsages};
use ftgsxr_app::{
    Intent, LibraryCommand, LoadedModel, ModelRef, Playback, PlaybackCommand, SplatCamera,
    SplatEvent, ToastLevel,
};
use ftgsxr_render::{OutputSpace, SortMode};
use ftgsxr_testkit::layer_messages;
use serde_json::Value;
use support::TestApp;
use support::ui::{record_toasts, toasts};

/// The PNG's chunks after the signature: (type, data).
fn chunks(png: &[u8]) -> Vec<([u8; 4], Vec<u8>)> {
    assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n", "the PNG signature");
    let mut out = Vec::new();
    let mut at = 8;
    while at + 12 <= png.len() {
        let len = u32::from_be_bytes(png[at..at + 4].try_into().unwrap()) as usize;
        let kind: [u8; 4] = png[at + 4..at + 8].try_into().unwrap();
        out.push((kind, png[at + 8..at + 8 + len].to_vec()));
        at += 12 + len;
    }
    out
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array()
        .expect("an array")
        .iter()
        .map(|x| x.as_f64().expect("a number") as f32)
        .collect()
}

/// A pose object against a global transform, rotation (x, y, z, w) (§1.14).
fn check_pose(v: &Value, g: &GlobalTransform, what: &str) {
    let (s, r, t) = g.to_scale_rotation_translation();
    assert_eq!(
        floats(&v["translation"]),
        t.to_array(),
        "{what} translation"
    );
    assert_eq!(floats(&v["rotation"]), r.to_array(), "{what} rotation");
    assert_eq!(v["scale"].as_f64().unwrap() as f32, s.x, "{what} scale");
}

#[test]
#[ignore = "gpu"]
fn gpu_i14_f12_writes_a_png_with_the_metadata_chunk() {
    let _layer = layer_messages(); // R-16, Q-105
    let mut t = TestApp::gpu("i14");
    // No egui context on the image camera (#288's note).
    t.app
        .world_mut()
        .resource_mut::<bevy_egui::EguiGlobalSettings>()
        .auto_create_primary_context = false;
    record_toasts(&mut t.app);
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

    let (w, h) = (320, 240);
    let mut image = Image::new_target_texture(w, h, TextureFormat::Rgba8UnormSrgb, None);
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    image.asset_usage = RenderAssetUsages::RENDER_WORLD;
    let target = t.app.world_mut().resource_mut::<Assets<Image>>().add(image);
    let camera = t
        .app
        .world_mut()
        .spawn((
            Camera3d::default(),
            Camera::default(),
            RenderTarget::Image(target.into()),
            Projection::Perspective(PerspectiveProjection {
                near: 0.01,
                fov: 50f32.to_radians(),
                ..PerspectiveProjection::default()
            }),
            Transform::from_xyz(0.3, 0.2, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
            SplatCamera {
                sort: SortMode::PerView,
                output: OutputSpace::LinearOver { background: None },
            },
        ))
        .id();
    let model = ftgsxr_testkit::random_model(14, 20_000, 1);
    let path = support::scratch("i14_model").join("i14 \"model\".ftgs-if");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, ftgsxr_testkit::exchange_bytes(&model)).unwrap();
    t.command(LibraryCommand::LoadRef(ModelRef::File(path)));
    assert!(
        t.until(600, |_| *uploaded.lock().unwrap()),
        "the model uploaded"
    );
    // A pose of its own, and a clock held at a time of its own.
    let instance = {
        let w = t.app.world_mut();
        let e = w
            .query_filtered::<Entity, With<LoadedModel>>()
            .single(w)
            .unwrap();
        *w.get_mut::<Transform>(e).unwrap() = Transform {
            translation: Vec3::new(0.25, -0.5, 0.125),
            rotation: Quat::from_rotation_y(0.7),
            scale: Vec3::splat(1.5),
        };
        e
    };
    t.app
        .world_mut()
        .write_message(Intent::Playback(PlaybackCommand::Pause));
    t.frames(2);
    t.app
        .world_mut()
        .write_message(Intent::Playback(PlaybackCommand::Seek(0.25)));
    t.frames(3);

    let dir = t.dir.join("pictures");
    t.app.world_mut().write_message(Intent::Screenshot);
    let png_in = |dir: &std::path::Path| {
        std::fs::read_dir(dir).ok().and_then(|d| {
            d.flatten()
                .map(|e| e.path())
                .find(|p| p.extension().is_some_and(|x| x == "png"))
        })
    };
    assert!(t.until(300, |_| png_in(&dir).is_some()), "a PNG is written");
    t.frames(2);
    let file = png_in(&dir).unwrap();
    let name = file.file_name().unwrap().to_string_lossy().into_owned();
    let stem = name.strip_suffix(".png").unwrap();
    assert!(
        stem.len() == 15
            && stem.as_bytes()[8] == b'-'
            && stem.chars().filter(char::is_ascii_digit).count() == 14,
        "<date>-<time>.png: {name}"
    );
    assert_eq!(
        std::fs::read_dir(&dir).unwrap().count(),
        1,
        "no temporary file left"
    );

    let bytes = std::fs::read(&file).unwrap();
    let decoded = image::load_from_memory(&bytes).expect("a valid PNG");
    assert_eq!(
        (decoded.width(), decoded.height()),
        (w, h),
        "the viewport's size"
    );
    let chunks = chunks(&bytes);
    assert_eq!(&chunks[0].0, b"IHDR");
    let itxt: Vec<_> = chunks.iter().filter(|c| &c.0 == b"iTXt").collect();
    assert_eq!(itxt.len(), 1, "one text chunk");
    let data = &itxt[0].1;
    let prefix = b"openftgs-xr\0\0\0\0\0";
    assert_eq!(
        &data[..prefix.len()],
        prefix,
        "keyword, uncompressed, no tags"
    );
    let meta: Value = serde_json::from_slice(&data[prefix.len()..]).expect("the chunk's JSON");

    assert_eq!(meta["version"], env!("CARGO_PKG_VERSION"));
    let world = t.app.world_mut();
    let loaded = world.get::<LoadedModel>(instance).unwrap().clone();
    let models = meta["models"].as_array().unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0]["identity"], loaded.identity.as_str());
    assert_eq!(models[0]["name"], loaded.name.as_str());
    assert_eq!(loaded.name, "i14 \"model\"");
    let g = *world.get::<GlobalTransform>(instance).unwrap();
    check_pose(&models[0]["pose"], &g, "the model's");
    let clock = world.resource::<Playback>().clock.time();
    assert_eq!(clock, 0.25);
    assert_eq!(meta["clock_t"].as_f64().unwrap(), clock);
    let g = *world.get::<GlobalTransform>(camera).unwrap();
    check_pose(&meta["camera"]["pose"], &g, "the camera's");
    assert_eq!(
        meta["camera"]["fov_deg"].as_f64().unwrap() as f32,
        50f32.to_radians().to_degrees()
    );
    assert_eq!(meta["camera"]["mode"], "fly");

    let toasts = toasts(&t.app);
    assert!(
        toasts
            .iter()
            .any(|t| t.level == ToastLevel::Info && t.text.contains(&name)),
        "the confirmation names the file: {toasts:?}"
    );
}

#[test]
fn a_screenshot_never_captured_ends_with_a_warning() {
    let mut t = TestApp::new("i14_expiry");
    record_toasts(&mut t.app);
    t.app
        .world_mut()
        .spawn((Camera3d::default(), Camera::default()));
    t.frames(1);
    t.app.world_mut().write_message(Intent::Screenshot);
    t.frames(130);
    let toasts = toasts(&t.app);
    assert!(
        toasts
            .iter()
            .any(|t| t.level == ToastLevel::Warning && t.text == "The screenshot was not captured"),
        "{toasts:?}"
    );
    assert!(!t.dir.join("pictures").exists(), "nothing written");
}
