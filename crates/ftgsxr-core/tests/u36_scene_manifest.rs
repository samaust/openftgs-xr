// SPDX-License-Identifier: Apache-2.0

//! U-36, the `ftgs-scene` reader (§1.5, §1.7, openftgs §2.2, `cpu` lane), on the committed
//! manifest fixture: cameras with K and world-to-camera, images with timestamps, splits and
//! frame indices, a per-image camera override; the preset hint from its camera up vectors;
//! the request camera's checks applied. The checks beyond the cameras are Q-26's, the
//! refusals' codes Q-25's.

use ftgsxr_core::{CoreError, DMat4, Preset, SceneManifest, Vec3, read_scene_manifest};
use ftgsxr_testkit::fixtures_dir;
use serde_json::{Value, json};

fn scene_bytes() -> Vec<u8> {
    std::fs::read(fixtures_dir().join("checkpoint/scene/scene.json")).unwrap()
}

fn scene_json() -> Value {
    serde_json::from_slice(&scene_bytes()).unwrap()
}

fn read(value: &Value) -> Result<SceneManifest, CoreError> {
    read_scene_manifest(&serde_json::to_vec(value).unwrap())
}

/// Row `r` of a world-to-camera matrix.
fn row(m: &DMat4, r: usize) -> [f64; 4] {
    [m.x_axis[r], m.y_axis[r], m.z_axis[r], m.w_axis[r]]
}

#[test]
fn u36_cameras_with_k_and_world_to_camera() {
    let m = read_scene_manifest(&scene_bytes()).unwrap();
    assert_eq!(m.time_unit, "s");
    let names: Vec<&str> = m.cameras.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["cam0", "cam1"]);
    for (_, cam) in &m.cameras {
        assert_eq!((cam.width, cam.height), (80, 60));
        let i = cam.intrinsics();
        assert_eq!((i.fx, i.fy, i.cx, i.cy), (80.0, 80.0, 40.0, 30.0));
        assert_eq!(cam.k.z_axis.z, 1.0);
        assert_eq!(row(&cam.world_to_camera, 3), [0.0, 0.0, 0.0, 1.0]);
    }
    // The fixture's literals (tests/fixtures/checkpoint/scene/scene.json).
    let cam0 = &m.cameras[0].1.world_to_camera;
    assert_eq!(row(cam0, 1), [0.0, -0.19611613, -0.9805807, 8.9249036e-17]);
    assert_eq!(row(cam0, 2), [0.0, 0.9805807, -0.19611613, 3.0594118]);
    let cam1 = &m.cameras[1].1.world_to_camera;
    assert_eq!(row(cam1, 0), [0.6726728, 0.73994005, -0.0, 1.9137176e-16]);
    assert_eq!(
        row(cam1, 2),
        [-0.70820504, 0.6438228, -0.28972024, 3.1064448]
    );
}

#[test]
fn u36_images_with_timestamps_splits_and_frame_indices() {
    let m = read_scene_manifest(&scene_bytes()).unwrap();
    let got: Vec<(&str, f64, &str, Option<i64>)> = m
        .images
        .iter()
        .map(|i| {
            (
                i.camera.as_str(),
                i.timestamp,
                i.split.as_str(),
                i.frame_index,
            )
        })
        .collect();
    let dt = 0.03333333333333333;
    assert_eq!(
        got,
        [
            ("cam0", 0.0, "train", Some(0)),
            ("cam1", 0.0, "train", Some(0)),
            ("cam0", dt, "train", Some(1)),
            ("cam1", dt, "test", Some(1)),
        ]
    );
}

#[test]
fn u36_a_per_image_camera_override() {
    let m = read_scene_manifest(&scene_bytes()).unwrap();
    assert!(m.images[..3].iter().all(|i| i.camera_override.is_none()));
    let over = m.images[3].camera_override.expect("image 3 overrides cam1");
    // K and the size come from cam1, the world-to-camera from the image.
    let cam1 = m.cameras[1].1;
    assert_eq!(
        (over.width, over.height, over.k),
        (cam1.width, cam1.height, cam1.k)
    );
    let w = &over.world_to_camera;
    assert_eq!(row(w, 0), [0.66436386, 0.74740934, -0.0, 5.551115e-17]);
    assert_eq!(
        row(w, 1),
        [0.2140856, -0.1902983, -0.9580991, -1.3415715e-16]
    );
    assert_eq!(row(w, 2), [-0.7160922, 0.6365264, -0.2864369, 3.1420534]);
    assert_eq!(row(w, 3), [0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn u36_an_overridden_k_keeps_the_cameras_world_to_camera() {
    let mut raw = scene_json();
    raw["images"][0]["K"] = json!([[90.0, 0.0, 41.0], [0.0, 91.0, 29.0], [0.0, 0.0, 1.0]]);
    let m = read(&raw).unwrap();
    let over = m.images[0].camera_override.unwrap();
    let i = over.intrinsics();
    assert_eq!((i.fx, i.fy, i.cx, i.cy), (90.0, 91.0, 41.0, 29.0));
    assert_eq!(over.world_to_camera, m.cameras[0].1.world_to_camera);
}

#[test]
fn u36_the_preset_hint_from_its_camera_up_vectors() {
    let m = read_scene_manifest(&scene_bytes()).unwrap();
    // §1.5: each camera's up is −R_wᵀ (0, 1, 0), minus R_w's second row.
    let ups: Vec<Vec3> = m
        .cameras
        .iter()
        .map(|(_, c)| {
            let r = row(&c.world_to_camera, 1);
            Vec3::new(-r[0] as f32, -r[1] as f32, -r[2] as f32)
        })
        .collect();
    // The fixture's world is z up (tests/fixtures/README.md, Layout).
    assert_eq!(Preset::from_camera_ups(&ups), Preset::ZUp);
}

#[test]
fn u36_the_request_camera_checks_apply() {
    let mut raw = scene_json();
    raw["cameras"]["cam1"]["K"][0][1] = json!(0.25);
    assert!(matches!(read(&raw), Err(CoreError::Value(_))), "a skewed K");

    let mut raw = scene_json();
    raw["cameras"]["cam0"]["world_to_camera"][1][1] = json!(-0.3);
    assert!(
        matches!(read(&raw), Err(CoreError::Value(_))),
        "not orthonormal"
    );

    let mut raw = scene_json();
    raw["images"][3]["world_to_camera"][0][0] = json!(0.7);
    assert!(
        matches!(read(&raw), Err(CoreError::Value(_))),
        "an override that is not orthonormal"
    );

    let mut raw = scene_json();
    raw["images"][2]["K"] = json!([[80.0, 0.0, 40.0], [0.0, 80.0, 30.0], [0.0, 0.1, 1.0]]);
    assert!(
        matches!(read(&raw), Err(CoreError::Value(_))),
        "an override's K"
    );

    let mut raw = scene_json();
    raw["cameras"]["cam0"]["width"] = json!(80.0);
    assert!(
        matches!(read(&raw), Err(CoreError::Shape(_))),
        "a width that is no integer"
    );
}

#[test]
fn u36_the_header_is_checked() {
    let mut raw = scene_json();
    raw["format"] = json!("ftgs-render-request");
    assert!(matches!(read(&raw), Err(CoreError::Format(_))));
    let mut raw = scene_json();
    raw["version"] = json!(2);
    assert!(matches!(read(&raw), Err(CoreError::Version(_))));
    assert!(matches!(
        read_scene_manifest(b"[1, 2"),
        Err(CoreError::Json(_))
    ));
}

#[test]
fn u36_cameras_keep_the_files_order() {
    let raw = scene_json();
    let cam0 = raw["cameras"]["cam0"].clone();
    let cam1 = raw["cameras"]["cam1"].clone();
    // Written by hand so that the order is not the sorted one.
    let text = format!(
        r#"{{"format": "ftgs-scene", "version": 1, "time_unit": "s",
            "cameras": {{"zeta": {cam1}, "alpha": {cam0}, "mid": {cam0}}},
            "images": [{{"camera": "alpha", "timestamp": 0.5, "path": "a.png", "split": "train"}}]}}"#
    );
    let m = read_scene_manifest(text.as_bytes()).unwrap();
    let names: Vec<&str> = m.cameras.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["zeta", "alpha", "mid"]);
    assert_eq!(m.cameras[1].1, read(&raw).unwrap().cameras[0].1);
    assert_eq!(m.images[0].frame_index, None);
}

#[test]
fn u36_image_rules() {
    let mut raw = scene_json();
    raw["images"][1]["camera"] = json!("cam9");
    assert!(
        matches!(read(&raw), Err(CoreError::Value(_))),
        "an unknown camera"
    );

    let mut raw = scene_json();
    raw["images"][1]["split"] = json!("val");
    assert!(matches!(read(&raw), Err(CoreError::Value(_))), "a split");

    let mut raw = scene_json();
    raw["images"][1]["frame_index"] = json!(1.5);
    assert!(
        matches!(read(&raw), Err(CoreError::Shape(_))),
        "a frame index"
    );

    let mut raw = scene_json();
    raw["images"][1]["timestamp"] = json!("0");
    assert!(
        matches!(read(&raw), Err(CoreError::Shape(_))),
        "a timestamp"
    );

    let mut raw = scene_json();
    raw["images"] = json!([]);
    assert!(matches!(read(&raw), Err(CoreError::Value(_))), "no image");

    let mut raw = scene_json();
    raw["cameras"]["cam0"]["model"] = json!("fisheye");
    assert!(
        matches!(read(&raw), Err(CoreError::Value(_))),
        "a camera model"
    );
}
