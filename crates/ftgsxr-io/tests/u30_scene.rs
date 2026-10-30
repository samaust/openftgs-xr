// SPDX-License-Identifier: Apache-2.0

//! U-30, the scene-file part: `ftgsxr_io::scene::read_scene` and `write_scene`
//! (`ftgsxr-scene` v1, §1.5 Scene files, §1.11, D-23): round trips; locations stored and
//! returned as given (Q-3's reading of the relative-path clause); a scene naming a file
//! that does not exist still reads (listing it as missing is the library's); the version
//! rules. The shapes are Q-52's, the refusals Q-53's.

use std::path::{Path, PathBuf};

use ftgsxr_core::{EndRule, LoopMode, Pose, Quat, Vec3};
use ftgsxr_io::{
    CameraState, ClockState, EnvironmentSettings, ListingFlavor, ManualLighting, SceneFile,
    SceneModel, SourceSpec, read_scene, write_scene,
};

/// A fresh, empty directory of this test binary's own under the target directory.
fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("u30_scene")
        .join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create the test directory");
    dir
}

/// A pose away from the identity, its rotation (x, y, z, w) a unit quaternion.
fn pose(dx: f32) -> Pose {
    Pose {
        translation: Vec3::new(0.25 + dx, -1.5, 3.0),
        rotation: Quat::from_xyzw(0.0, 0.6, 0.0, 0.8),
        scale: 1.75,
    }
}

/// An instance with every field away from its default.
fn model(location: &str, identity_digit: char) -> SceneModel {
    SceneModel {
        source: SourceSpec {
            kind: "http".into(),
            location: location.into(),
            name: "my server".into(),
            flavor: Some(ListingFlavor::NginxJson),
        },
        identity: std::iter::repeat_n(identity_digit, 64).collect(),
        name: "dancer take 2".into(),
        pose: pose(0.0),
        visible: false,
        mode: "both".into(),
        time_offset_s: -0.125,
        end_rule: EndRule::Extrapolate,
        lighting: Some(ManualLighting {
            exposure_ev: 1.5,
            temperature: -0.25,
            tint: 0.1,
        }),
        audio: Some("dancer.ogg".into()),
    }
}

/// A scene with every field away from its default.
fn full_scene() -> SceneFile {
    let mut second = model("/srv/models/b.safetensors", 'b');
    second.source = SourceSpec {
        kind: "dir".into(),
        location: "/srv/models".into(),
        name: "/srv/models".into(),
        flavor: None,
    };
    second.lighting = None;
    second.audio = None;
    second.end_rule = EndRule::Hold;
    SceneFile {
        models: vec![
            model("https://example.org/models/a.safetensors", 'a'),
            second,
        ],
        clock: ClockState {
            t: 12.375,
            speed: -0.25,
            loop_mode: LoopMode::PingPong,
            range: Some((1.5, 9.25)),
        },
        environment: EnvironmentSettings {
            mode: "passthrough".into(),
            background: [0.2, 0.3, 0.4],
            passthrough_brightness: 0.1,
            passthrough_contrast: -0.2,
            passthrough_saturation: 0.3,
            occlusion: false,
            occlusion_soft: false,
            occlusion_delta_m: 0.08,
            hand_removal: true,
            lighting_mode: "auto-local".into(),
            lighting_manual: ManualLighting {
                exposure_ev: -1.0,
                temperature: 0.5,
                tint: -0.5,
            },
            lighting_gamma: 0.75,
            lighting_reference: Some([0.45, 0.31, 0.33, 0.36]),
            contact_shadows: false,
        },
        anchor: Some("4f1c2a9e-0b7d-4c3e-9a51-7d2e8f6b1c04".into()),
        rig: Some(pose(1.0)),
        camera: Some(CameraState {
            pose: Pose {
                scale: 1.0,
                ..pose(2.0)
            },
            fov_deg: 75.0,
            orbit: true,
        }),
        bookmarks: vec![
            CameraState {
                pose: Pose::IDENTITY,
                fov_deg: 60.0,
                orbit: false,
            },
            CameraState {
                pose: pose(3.0),
                fov_deg: 90.5,
                orbit: true,
            },
        ],
    }
}

/// Writes `scene` under `dir` and reads it back.
fn round_trip(dir: &Path, scene: &SceneFile) -> SceneFile {
    let path = dir.join("scenes").join("test.ftgsxr-scene.json");
    write_scene(&path, scene).expect("write the scene");
    read_scene(&std::fs::read(&path).expect("read the file")).expect("read the scene")
}

/// U-30 round trip: a scene with every field away from its default reads back equal.
#[test]
fn u30_scene_round_trips() {
    let dir = scratch("round_trip");
    let scene = full_scene();
    assert_eq!(round_trip(&dir, &scene), scene);
}

/// U-30 round trip of the empty choices: no instance, no range, no lighting, no anchor,
/// rig, camera or bookmark, the default environment.
#[test]
fn u30_scene_round_trips_the_nulls() {
    let dir = scratch("nulls");
    let scene = SceneFile {
        models: vec![],
        clock: ClockState {
            t: 0.0,
            speed: 1.0,
            loop_mode: LoopMode::Loop,
            range: None,
        },
        environment: EnvironmentSettings::default(),
        anchor: None,
        rig: None,
        camera: None,
        bookmarks: vec![],
    };
    assert_eq!(round_trip(&dir, &scene), scene);
}

/// U-30 paths (Q-3): a location relative to the model directory, an absolute one and a URL,
/// and a relative audio path, are written and read back exactly as given; the written file
/// holds them verbatim.
#[test]
fn u30_scene_locations_round_trip_as_given() {
    let dir = scratch("locations");
    let mut scene = full_scene();
    let locations = [
        ("dir", "captures/dancer.safetensors"),
        ("dir", "/home/user/models/dancer.safetensors"),
        ("http", "https://example.org/models/dancer.safetensors"),
    ];
    scene.models = locations
        .iter()
        .map(|(kind, location)| {
            let mut m = model(location, 'c');
            m.source.kind = (*kind).into();
            m.audio = Some("sounds/../dancer.ogg".into());
            m
        })
        .collect();
    let back = round_trip(&dir, &scene);
    assert_eq!(back, scene);
    let text = std::fs::read_to_string(dir.join("scenes").join("test.ftgsxr-scene.json"))
        .expect("read the file");
    for (_, location) in locations {
        assert!(text.contains(&format!("\"{location}\"")), "{location}");
    }
    assert!(text.contains("\"sounds/../dancer.ogg\""));
}

/// A scene document written by hand from §1.11's shape: one instance whose source is
/// `source`, everything else minimal, with `extra` spliced into the top level.
fn scene_json(version: &str, source: &str, extra: &str) -> String {
    format!(
        r#"{{
  "format": "ftgsxr-scene", "version": {version}, {extra}
  "models": [{{
    "source": {source},
    "identity": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    "name": "dancer",
    "pose": {{"translation": [1, 2, 3], "rotation": [0, 0.6, 0, 0.8], "scale": 0.5}},
    "visible": true, "mode": "splats", "time_offset_s": 2.5, "end_rule": "hold",
    "lighting": null, "audio": null
  }}],
  "clock": {{"t": 1.25, "speed": 2, "loop": "once", "range": [0.5, 3]}},
  "environment": {{
    "mode": "vr", "background": [0.1, 0.1, 0.1],
    "passthrough_brightness": 0, "passthrough_contrast": 0, "passthrough_saturation": 0,
    "occlusion": true, "occlusion_soft": true, "occlusion_delta_m": 0.05,
    "hand_removal": false, "lighting_mode": "off",
    "lighting_manual": {{"exposure_ev": 0, "temperature": 0, "tint": 0}},
    "lighting_gamma": 0.5, "lighting_reference": null, "contact_shadows": true
  }},
  "anchor": null, "rig": {{"pose": {{"translation": [0, 0, 0], "rotation": [0, 0, 0, 1], "scale": 1}}}},
  "camera": null, "bookmarks": []
}}"#
    )
}

/// U-30 "a scene naming a missing model still reads, its location kept as written": a
/// hand-written scene naming a file that does not exist reads, with its values as written.
/// Listing it as missing is M-4's (I-9).
#[test]
fn u30_scene_naming_a_missing_file_reads() {
    let missing = scratch("missing").join("no such model.safetensors");
    let source = format!(r#"{{"kind": "dir", "location": "{}"}}"#, missing.display());
    let scene = read_scene(scene_json("1", &source, "").as_bytes()).expect("the scene reads");
    let m = &scene.models[0];
    assert_eq!(
        m.source,
        SourceSpec {
            kind: "dir".into(),
            location: missing.display().to_string(),
            name: missing.display().to_string(),
            flavor: None,
        }
    );
    assert_eq!(m.pose.translation, Vec3::new(1.0, 2.0, 3.0));
    // §1.14: (x, y, z, w) in the player's files.
    assert_eq!(m.pose.rotation, Quat::from_xyzw(0.0, 0.6, 0.0, 0.8));
    assert_eq!(m.pose.scale, 0.5);
    assert_eq!(m.end_rule, EndRule::Hold);
    assert_eq!(
        scene.clock,
        ClockState {
            t: 1.25,
            speed: 2.0,
            loop_mode: LoopMode::Once,
            range: Some((0.5, 3.0)),
        }
    );
    assert_eq!(scene.environment, EnvironmentSettings::default());
    assert_eq!(scene.rig, Some(Pose::IDENTITY));
    assert_eq!(scene.camera, None);
}

/// `scene_json`'s minimal scene with its `environment` object replaced by `environment`.
fn with_environment(environment: &str) -> String {
    let good = scene_json("1", SOURCE, "");
    let start = good.find(r#""environment": {"#).expect("an environment");
    let end = good.find(r#""anchor""#).expect("an anchor");
    format!(
        r#"{}"environment": {environment},
  {}"#,
        &good[..start],
        &good[end..]
    )
}

/// Q-52 (#91): every key of the scene's `environment` may be absent; an empty object reads
/// as `EnvironmentSettings::default()`.
#[test]
fn u30_scene_empty_environment_is_the_default() {
    let scene = read_scene(with_environment("{}").as_bytes()).expect("the scene reads");
    assert_eq!(scene.environment, EnvironmentSettings::default());
}

/// Q-52 (#91): a key missing from `environment` (`occlusion_delta_m`) and one missing
/// from its `lighting_manual` (`tint`) take the default's values; every other field is
/// read as written, each away from its default.
#[test]
fn u30_scene_missing_environment_keys_take_the_default() {
    let environment = r#"{
    "mode": "passthrough", "background": [0.25, 0.5, 0.75],
    "passthrough_brightness": 10, "passthrough_contrast": -20, "passthrough_saturation": 30,
    "occlusion": false, "occlusion_soft": false,
    "hand_removal": true, "lighting_mode": "manual",
    "lighting_manual": {"exposure_ev": 1.5, "temperature": -40},
    "lighting_gamma": 0.25, "lighting_reference": [1, 2, 3, 4], "contact_shadows": false
  }"#;
    let scene = read_scene(with_environment(environment).as_bytes()).expect("the scene reads");
    assert_eq!(
        scene.environment,
        EnvironmentSettings {
            mode: "passthrough".into(),
            background: [0.25, 0.5, 0.75],
            passthrough_brightness: 10.0,
            passthrough_contrast: -20.0,
            passthrough_saturation: 30.0,
            occlusion: false,
            occlusion_soft: false,
            // Q-50's default (D-16's δ).
            occlusion_delta_m: 0.05,
            hand_removal: true,
            lighting_mode: "manual".into(),
            lighting_manual: ManualLighting {
                exposure_ev: 1.5,
                temperature: -40.0,
                // Q-50's default, the identity transform's.
                tint: 0.0,
            },
            lighting_gamma: 0.25,
            lighting_reference: Some([1.0, 2.0, 3.0, 4.0]),
            contact_shadows: false,
        }
    );
    let absent = with_environment(r#"{"lighting_mode": "manual"}"#);
    let scene = read_scene(absent.as_bytes()).expect("the scene reads");
    assert_eq!(scene.environment.lighting_reference, None);
    assert_eq!(scene.environment.lighting_manual, ManualLighting::default());
}

/// Q-52, Q-53 (#91): only absence falls back to the default. A key present with the wrong
/// type is still E-SCENE-FORMAT naming its path, in `environment` and in its
/// `lighting_manual`, and a scene without `environment` at all is E-SCENE-FORMAT.
#[test]
fn u30_scene_environment_wrong_type_or_missing_object_is_scene_format() {
    let cases = [
        (r#"{"occlusion": 1}"#, "environment.occlusion"),
        (
            r#"{"lighting_manual": {"tint": "warm"}}"#,
            "environment.lighting_manual.tint",
        ),
        (r#"{"lighting_manual": 0}"#, "environment.lighting_manual"),
        (r#"{"mode": "ar"}"#, "environment.mode"),
        (r#"{"background": null}"#, "environment.background"),
        ("[]", "environment"),
    ];
    for (environment, path) in cases {
        let err = read_scene(with_environment(environment).as_bytes()).expect_err(environment);
        assert_eq!(err.code(), "E-SCENE-FORMAT", "{environment}");
        assert!(err.to_string().contains(path), "{environment}: {err}");
    }
    let good = scene_json("1", SOURCE, "");
    let start = good.find(r#""environment": {"#).expect("an environment");
    let end = good.find(r#""anchor""#).expect("an anchor");
    let without = format!("{}{}", &good[..start], &good[end..]);
    let err = read_scene(without.as_bytes()).expect_err("no environment");
    assert_eq!(err.code(), "E-SCENE-FORMAT");
    assert!(err.to_string().contains("environment: missing"), "{err}");
}

/// The §1.15 code of `read_scene`'s refusal of `json`.
fn refusal(json: &str) -> &'static str {
    read_scene(json.as_bytes())
        .expect_err("the scene is refused")
        .code()
}

/// The source every hand-written scene below uses.
const SOURCE: &str = r#"{"kind": "dir", "location": "a.safetensors"}"#;

/// U-30 version rules (§1.11): a minor version is ignored and so are unknown keys, at
/// the top level and inside an instance; an unknown major version, another or a missing
/// `format` or a missing `version` is E-SCENE-FORMAT (Q-53); bytes that are not JSON
/// are E-JSON.
#[test]
fn u30_scene_version_rules() {
    let plain = read_scene(scene_json("1", SOURCE, "").as_bytes()).expect("version 1");
    for version in ["\"1.3\"", "\"1\""] {
        let read = read_scene(scene_json(version, SOURCE, "").as_bytes());
        assert_eq!(read.expect(version), plain, "{version}");
    }
    let extra = r#""thumbnail": "x.png", "future": {"a": [1, 2]},"#;
    let unknown_source = r#"{"kind": "dir", "location": "a.safetensors", "since": 3}"#;
    let read = read_scene(scene_json("1", unknown_source, extra).as_bytes());
    assert_eq!(read.expect("unknown keys"), plain);

    assert_eq!(refusal(&scene_json("2", SOURCE, "")), "E-SCENE-FORMAT");
    assert_eq!(
        refusal(&scene_json("\"2.0\"", SOURCE, "")),
        "E-SCENE-FORMAT"
    );
    let other_format = scene_json("1", SOURCE, "").replace("ftgsxr-scene", "ftgsxr-poses");
    assert_eq!(refusal(&other_format), "E-SCENE-FORMAT");
    let no_format = scene_json("1", SOURCE, "").replace(r#""format": "ftgsxr-scene","#, "");
    assert_eq!(refusal(&no_format), "E-SCENE-FORMAT");
    let no_version = scene_json("1", SOURCE, "").replace(r#""version": 1,"#, "");
    assert_eq!(refusal(&no_version), "E-SCENE-FORMAT");
    assert_eq!(refusal("{\"format\": \"ftgsxr-scene\""), "E-JSON");
    assert_eq!(refusal("not json"), "E-JSON");
}

/// Q-53: a document that breaks the schema is E-SCENE-FORMAT: a missing required key, a
/// wrong type, an identity that is not 64 lowercase hex digits, a spelling outside the
/// lists, a non-positive scale, a zero rotation, a reversed range, ten bookmarks.
#[test]
fn u30_scene_schema_breaks_are_scene_format() {
    let good = scene_json("1", SOURCE, "");
    let identity = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    let camera = r#"{"pose": {"translation": [0, 0, 0], "rotation": [0, 0, 0, 1], "scale": 1}, "fov_deg": 60, "mode": "fly"}"#;
    let ten = [camera; 10].join(", ");
    let cases = [
        (r#""visible": true, "#, ""),
        (r#""visible": true"#, r#""visible": "yes""#),
        (identity, &identity.to_uppercase()),
        (identity, &identity[1..]),
        (r#""mode": "splats""#, r#""mode": "points""#),
        (r#""end_rule": "hold""#, r#""end_rule": "clamp""#),
        (r#""loop": "once""#, r#""loop": "bounce""#),
        (r#""mode": "vr""#, r#""mode": "ar""#),
        (r#""lighting_mode": "off""#, r#""lighting_mode": "auto""#),
        (r#""kind": "dir""#, r#""kind": "ftp""#),
        (r#""scale": 0.5"#, r#""scale": 0"#),
        (
            r#""rotation": [0, 0.6, 0, 0.8]"#,
            r#""rotation": [0, 0, 0, 0]"#,
        ),
        (
            r#""rotation": [0, 0.6, 0, 0.8]"#,
            r#""rotation": [0, 0.6, 0.8]"#,
        ),
        (r#""range": [0.5, 3]"#, r#""range": [3, 0.5]"#),
        (r#""time_offset_s": 2.5"#, r#""time_offset_s": 1e400"#),
        (r#""bookmarks": []"#, &format!(r#""bookmarks": [{ten}]"#)),
        (
            r#""lighting_reference": null"#,
            r#""lighting_reference": [1, 2, 3]"#,
        ),
        (r#""contact_shadows": true"#, r#""contact_shadows": null"#),
    ];
    for (from, to) in cases {
        assert!(good.contains(from), "{from}");
        let bad = good.replacen(from, to, 1);
        let code = read_scene(bad.as_bytes()).map(|_| ()).map_err(|e| e.code());
        // A number out of f64's range is a JSON syntax error in serde_json's reading: E-JSON.
        let expected = if to.contains("1e400") {
            "E-JSON"
        } else {
            "E-SCENE-FORMAT"
        };
        assert_eq!(code, Err(expected), "{from} → {to}");
    }
    let nine = good.replacen(
        r#""bookmarks": []"#,
        &format!(r#""bookmarks": [{}]"#, [camera; 9].join(", ")),
        1,
    );
    assert_eq!(
        read_scene(nine.as_bytes())
            .expect("nine bookmarks")
            .bookmarks
            .len(),
        9
    );
}

/// Q-53: `write_scene` refuses, writing nothing, a value `read_scene` would refuse (a
/// NaN, ten bookmarks, an identity in capitals, an unknown render mode), so a written file
/// always reads back.
#[test]
fn u30_write_scene_refuses_what_read_scene_would() {
    let dir = scratch("refusals");
    let path = dir.join("bad.ftgsxr-scene.json");
    let mut nan = full_scene();
    nan.clock.t = f64::NAN;
    let mut ten = full_scene();
    ten.bookmarks = vec![ten.bookmarks[0].clone(); 10];
    let mut capitals = full_scene();
    capitals.models[0].identity = capitals.models[0].identity.to_uppercase();
    let mut mode = full_scene();
    mode.models[1].mode = "wireframe".into();
    let mut scale = full_scene();
    scale.camera.as_mut().expect("a camera").pose.scale = -1.0;
    for (what, scene) in [
        ("NaN", nan),
        ("ten bookmarks", ten),
        ("capitals", capitals),
        ("mode", mode),
        ("scale", scale),
    ] {
        let err = write_scene(&path, &scene).expect_err(what);
        assert_eq!(err.code(), "E-SCENE-FORMAT", "{what}");
        assert!(!path.exists(), "{what}: nothing written");
    }
    assert_eq!(std::fs::read_dir(&dir).expect("list").count(), 0);
}

/// §1.11 Locations: `write_scene` replaces an existing file through a temporary sibling,
/// leaving only the new file.
#[test]
fn u30_write_scene_replaces_the_file_atomically() {
    let dir = scratch("atomic");
    let path = dir.join("s.ftgsxr-scene.json");
    std::fs::write(&path, "old").expect("write the old file");
    let scene = full_scene();
    write_scene(&path, &scene).expect("write the scene");
    let names: Vec<_> = std::fs::read_dir(&dir)
        .expect("list")
        .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(names, ["s.ftgsxr-scene.json"]);
    let back = read_scene(&std::fs::read(&path).expect("read")).expect("the new scene");
    assert_eq!(back, scene);
}

/// §2.7 Fixtures: the committed sample `scene/v1.ftgsxr-scene.json` (hand-written, every
/// key away from its default, one relative location) reads as `full_scene` with its second
/// source in the model directory; `fixtures_dir` checks its SHA-256 first.
#[test]
fn u30_the_sample_scene_fixture_reads() {
    let path = ftgsxr_testkit::fixtures_dir()
        .join("scene")
        .join("v1.ftgsxr-scene.json");
    let scene = read_scene(&std::fs::read(path).expect("read the sample")).expect("the sample");
    let mut expected = full_scene();
    expected.models[1].source = SourceSpec {
        kind: "dir".into(),
        location: "captures".into(),
        name: "my captures".into(),
        flavor: None,
    };
    assert_eq!(scene, expected);
}
