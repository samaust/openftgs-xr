// SPDX-License-Identifier: Apache-2.0

//! U-30, the pose-store part: `ftgsxr_io::scene::PoseStore` and `SavedPose`
//! (`ftgsxr-poses` v1, §1.5 Poses, §1.11, D-23): get, set, forget; a save and load round
//! trip; the version rules. The shape is Q-52's, the refusals Q-53's.

use std::path::PathBuf;

use ftgsxr_core::{CoreError, Pose, Preset, Quat, Vec3};
use ftgsxr_io::{IoError, PoseStore, SavedPose};

/// A fresh, empty directory of this test binary's own under the target directory.
fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("u30_pose_store")
        .join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create the test directory");
    dir
}

/// A model identity of 64 lowercase hex digits.
fn identity(digit: char) -> String {
    std::iter::repeat_n(digit, 64).collect()
}

/// A saved pose away from the identity with the preset `preset`.
fn saved(preset: Preset, size: f32) -> SavedPose {
    SavedPose {
        pose: Pose {
            translation: Vec3::new(0.5, 0.0, -1.5),
            rotation: Quat::from_xyzw(0.0, 0.28, 0.0, 0.96),
            scale: 0.125,
        },
        preset,
        fit_size_m: size,
        saved_at: "2026-10-29T18:04:05.123456789Z".into(),
    }
}

/// U-30 `PoseStore` get, set, forget: a missing file loads an empty store; `set` saves
/// under an identity and replaces; `forget` removes, and forgetting twice is harmless.
#[test]
fn u30_pose_store_get_set_forget() {
    let dir = scratch("get_set_forget");
    let mut store = PoseStore::load(&dir.join("poses.json")).expect("a missing file");
    let (a, b) = (identity('a'), identity('b'));
    assert_eq!(store.get(&a), None);
    store.set(&a, saved(Preset::YUp, 1.0));
    store.set(&b, saved(Preset::ZUp, 2.0));
    assert_eq!(store.get(&a), Some(&saved(Preset::YUp, 1.0)));
    store.set(&a, saved(Preset::NegYUp, 0.5));
    assert_eq!(store.get(&a), Some(&saved(Preset::NegYUp, 0.5)));
    store.forget(&a);
    assert_eq!(store.get(&a), None);
    store.forget(&a);
    assert_eq!(store.get(&b), Some(&saved(Preset::ZUp, 2.0)));
}

/// U-30 round trip: every preset, a custom one included (Q-52's `preset_rotation`),
/// saves and loads back equal, into a directory that did not exist; the file holds the
/// rotation (x, y, z, w) (§1.14) and leaves no temporary sibling.
#[test]
fn u30_pose_store_round_trips() {
    let dir = scratch("round_trip");
    let path = dir.join("config").join("poses.json");
    let mut store = PoseStore::default();
    let custom = Preset::Custom(Quat::from_xyzw(0.6, 0.0, 0.0, 0.8));
    for (digit, preset) in [
        ('0', Preset::YUp),
        ('1', Preset::ZUp),
        ('2', Preset::NegYUp),
        ('3', custom),
    ] {
        store.set(
            &identity(digit),
            saved(preset, 0.25 + f32::from(digit as u8 - b'0')),
        );
    }
    store.save(&path).expect("save");
    assert_eq!(PoseStore::load(&path).expect("load"), store);
    let text = std::fs::read_to_string(&path).expect("read the file");
    let doc: serde_json::Value = serde_json::from_str(&text).expect("JSON");
    assert_eq!(doc["format"], "ftgsxr-poses");
    assert_eq!(doc["version"], 1);
    let entry = &doc[identity('3')];
    assert_eq!(entry["rotation"], serde_json::json!([0.0, 0.28, 0.0, 0.96]));
    assert_eq!(entry["preset"], "custom");
    assert_eq!(
        entry["preset_rotation"],
        serde_json::json!([0.6, 0.0, 0.0, 0.8])
    );
    assert_eq!(doc[identity('1')]["preset"], "z-up");
    assert_eq!(
        std::fs::read_dir(path.parent().unwrap()).unwrap().count(),
        1
    );
}

/// A pose file written by hand from §1.11's shape and Q-52's flat object.
fn poses_json(version: &str, extra: &str, entry: &str) -> String {
    format!(
        r#"{{"format": "ftgsxr-poses", "version": {version}, {extra}
  "{}": {entry}}}"#,
        identity('e')
    )
}

/// An entry of the hand-written pose file.
const ENTRY: &str = r#"{"translation": [1, 2, 3], "rotation": [0, 0.6, 0, 0.8], "scale": 2,
  "preset": "z-up", "fit_size_m": 1.5, "saved_at": "2026-10-29T12:00:00Z"}"#;

/// Loads the pose file `text` from a file under `dir`.
fn load_text(dir: &std::path::Path, text: &str) -> Result<PoseStore, ftgsxr_io::IoError> {
    let path = dir.join("poses.json");
    std::fs::write(&path, text).expect("write the pose file");
    PoseStore::load(&path)
}

/// U-30 version rules for the pose store (§1.11): a hand-written file reads with the
/// rotation (x, y, z, w); a minor version, unknown keys and keys that are not identities
/// are ignored; a well-formed `version` of another major is E-FILE-VERSION; another
/// `format`, or a missing or malformed `version`, is E-SCENE-FORMAT (Q-53); a file that
/// is not JSON is E-JSON.
#[test]
fn u30_pose_store_version_rules() {
    let dir = scratch("versions");
    let store = load_text(&dir, &poses_json("1", "", ENTRY)).expect("version 1");
    assert_eq!(
        store.get(&identity('e')),
        Some(&SavedPose {
            pose: Pose {
                translation: Vec3::new(1.0, 2.0, 3.0),
                rotation: Quat::from_xyzw(0.0, 0.6, 0.0, 0.8),
                scale: 2.0,
            },
            preset: Preset::ZUp,
            fit_size_m: 1.5,
            saved_at: "2026-10-29T12:00:00Z".into(),
        })
    );
    let extra = r#""comment": "hand-written", "ABC": {"scale": "not a pose"},"#;
    let unknown_in_entry = ENTRY.replace("\"scale\": 2", "\"scale\": 2, \"note\": [1]");
    for text in [
        poses_json("\"1.4\"", "", ENTRY),
        poses_json("\"1.7\"", "", ENTRY),
        poses_json("1", extra, ENTRY),
        poses_json("1", "", &unknown_in_entry),
    ] {
        assert_eq!(load_text(&dir, &text).expect(&text), store, "{text}");
    }
    let refused = |text: String| load_text(&dir, &text).expect_err(&text).code();
    for version in ["2", "\"2.1\""] {
        let text = poses_json(version, "", ENTRY);
        assert_eq!(refused(text), "E-FILE-VERSION", "{version}");
    }
    let newer = load_text(&dir, &poses_json("\"2.1\"", "", ENTRY)).expect_err("2.1");
    assert!(
        matches!(&newer, IoError::Core(CoreError::Version(v)) if v.starts_with("\"2.1\"")),
        "{newer:?}"
    );
    for version in ["true", "1.5", "\"v1\""] {
        let text = poses_json(version, "", ENTRY);
        assert_eq!(refused(text), "E-SCENE-FORMAT", "{version}");
    }
    let no_version = poses_json("1", "", ENTRY).replace(r#""version": 1,"#, "");
    assert_eq!(refused(no_version), "E-SCENE-FORMAT");
    assert_eq!(
        refused(poses_json("1", "", ENTRY).replace("ftgsxr-poses", "ftgsxr-scene")),
        "E-SCENE-FORMAT"
    );
    assert_eq!(refused("{\"format\": ".into()), "E-JSON");
}

/// Q-53: an entry that breaks the schema refuses the file with E-SCENE-FORMAT: a
/// missing key, a spelling outside §1.11's presets, a custom preset without its rotation,
/// a non-positive fit size or scale, a zero rotation.
#[test]
fn u30_pose_store_schema_breaks_are_scene_format() {
    let dir = scratch("schema");
    let cases = [
        (r#", "saved_at": "2026-10-29T12:00:00Z""#, ""),
        (r#""preset": "z-up""#, r#""preset": "x-up""#),
        (r#""preset": "z-up""#, r#""preset": "custom""#),
        (r#""fit_size_m": 1.5"#, r#""fit_size_m": 0"#),
        (r#""scale": 2"#, r#""scale": -2"#),
        (
            r#""rotation": [0, 0.6, 0, 0.8]"#,
            r#""rotation": [0, 0, 0, 0]"#,
        ),
        (r#""translation": [1, 2, 3]"#, r#""translation": [1, 2]"#),
    ];
    for (from, to) in cases {
        assert!(ENTRY.contains(from), "{from}");
        let text = poses_json("1", "", &ENTRY.replacen(from, to, 1));
        let code = load_text(&dir, &text).map(|_| ()).map_err(|e| e.code());
        assert_eq!(code, Err("E-SCENE-FORMAT"), "{from} → {to}");
    }
    let custom = ENTRY.replace(
        r#""preset": "z-up""#,
        r#""preset": "custom", "preset_rotation": [0, 0, 0.6, 0.8]"#,
    );
    let store = load_text(&dir, &poses_json("1", "", &custom)).expect("a custom preset");
    assert_eq!(
        store.get(&identity('e')).map(|s| s.preset),
        Some(Preset::Custom(Quat::from_xyzw(0.0, 0.0, 0.6, 0.8)))
    );
}

/// Q-53: `save` refuses, writing nothing, a store its loader would refuse or ignore
/// (a key that is not an identity, a NaN, a non-positive fit size), and E-IO when the
/// file cannot be written.
#[test]
fn u30_pose_store_save_refuses_what_load_would() {
    let dir = scratch("save_refusals");
    let path = dir.join("poses.json");
    let mut not_identity = PoseStore::default();
    not_identity.set("dancer", saved(Preset::YUp, 1.0));
    let mut nan = PoseStore::default();
    let mut pose = saved(Preset::YUp, 1.0);
    pose.pose.translation.x = f32::NAN;
    nan.set(&identity('a'), pose);
    let mut size = PoseStore::default();
    size.set(&identity('a'), saved(Preset::YUp, 0.0));
    for (what, store) in [
        ("not an identity", not_identity),
        ("NaN", nan),
        ("size", size),
    ] {
        assert_eq!(
            store.save(&path).expect_err(what).code(),
            "E-SCENE-FORMAT",
            "{what}"
        );
        assert!(!path.exists(), "{what}: nothing written");
    }
    // A directory where the file should be: the rename fails.
    std::fs::create_dir_all(path.join("x")).expect("a directory in the way");
    let mut store = PoseStore::default();
    store.set(&identity('a'), saved(Preset::YUp, 1.0));
    assert_eq!(store.save(&path).expect_err("E-IO").code(), "E-IO");
}

/// The names of the files in `dir`, sorted.
fn names(dir: &std::path::Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .expect("list the test directory")
        .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

/// Today's UTC date as `YYYY-MM-DD`, counted day by day from 1970-01-01 (an oracle
/// independent of the crate's date arithmetic).
fn utc_today() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("after 1970")
        .as_secs();
    let (mut days, mut year) = (secs / 86_400, 1970);
    loop {
        let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
        let len = if leap { 366 } else { 365 };
        if days < len {
            let february = if leap { 29 } else { 28 };
            let months = [31, february, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
            let mut month = 0;
            while days >= months[month] {
                days -= months[month];
                month += 1;
            }
            return format!("{year:04}-{:02}-{:02}", month + 1, days + 1);
        }
        days -= len;
        year += 1;
    }
}

/// Q-53 (#74): a `poses.json` that `load` refuses (not JSON, breaking the schema, version
/// 2) is renamed to `poses.json.unknown-<today UTC>` with its bytes unchanged, and `load`
/// returns the matching error, its text naming the new file; a second refusal the same
/// day takes `-2`, so the next save overwrites neither.
#[test]
fn u30_a_refused_pose_store_is_set_aside() {
    let cases = [
        ("not_json", "{\"format\": ".to_owned(), "E-JSON"),
        (
            "schema",
            poses_json("1", "", &ENTRY.replace("\"z-up\"", "\"x-up\"")),
            "E-SCENE-FORMAT",
        ),
        ("version_2", poses_json("2", "", ENTRY), "E-FILE-VERSION"),
    ];
    for (name, text, code) in cases {
        let dir = scratch(&format!("set_aside_{name}"));
        let path = dir.join("poses.json");
        std::fs::write(&path, &text).expect("write the pose file");
        let before = utc_today();
        let error = PoseStore::load(&path).expect_err(name);
        let after = utc_today();
        assert_eq!(error.code(), code, "{name}");
        let files = names(&dir);
        assert_eq!(files.len(), 1, "{name}: {files:?}");
        let kept = files[0].clone();
        assert!(
            kept == format!("poses.json.unknown-{before}")
                || kept == format!("poses.json.unknown-{after}"),
            "{name}: {kept}"
        );
        assert_eq!(
            std::fs::read(dir.join(&kept)).expect("kept"),
            text.as_bytes()
        );
        let shown = error.user_text();
        assert!(shown.contains(&kept), "{name}: {shown}");

        std::fs::write(&path, &text).expect("write the pose file again");
        let again = PoseStore::load(&path).expect_err(name);
        let files = names(&dir);
        assert_eq!(files.len(), 2, "{name}: {files:?}");
        let second = format!("{kept}-2");
        assert!(files.contains(&second), "{name}: {files:?}");
        assert!(again.user_text().contains(&second), "{name}: {again}");
    }
}

/// Q-53 (#74): when a refused `poses.json` cannot be renamed (its directory is read-only),
/// `load` is E-IO and the file stays as it was.
#[cfg(unix)]
#[test]
fn u30_a_pose_store_that_cannot_be_set_aside_is_io() {
    use std::os::unix::fs::PermissionsExt;
    let dir = scratch("set_aside_fails");
    let path = dir.join("poses.json");
    let text = poses_json("2", "", ENTRY);
    std::fs::write(&path, &text).expect("write the pose file");
    let mode = |m| std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(m));
    mode(0o555).expect("make the directory read-only");
    let result = PoseStore::load(&path);
    mode(0o755).expect("restore the directory");
    if names(&dir).len() == 2 {
        return; // Privileges that ignore the mode (root): nothing to check.
    }
    assert_eq!(result.expect_err("E-IO").code(), "E-IO");
    assert_eq!(names(&dir), ["poses.json"]);
    assert_eq!(std::fs::read(&path).expect("the file"), text.as_bytes());
}

/// U-30: a missing `poses.json` is an empty store, and loading it creates no file.
#[test]
fn u30_a_missing_pose_store_is_empty_and_creates_nothing() {
    let dir = scratch("missing");
    let store = PoseStore::load(&dir.join("poses.json")).expect("a missing file");
    assert_eq!(store, PoseStore::default());
    assert!(names(&dir).is_empty());
}

/// §2.7 Fixtures: the committed sample `poses/v1.json` (hand-written, one entry per
/// preset) loads as the store `u30_pose_store_round_trips` saves; `fixtures_dir` checks
/// its SHA-256 first.
#[test]
fn u30_the_sample_pose_fixture_loads() {
    let path = ftgsxr_testkit::fixtures_dir().join("poses").join("v1.json");
    let store = PoseStore::load(&path).expect("the sample");
    let custom = Preset::Custom(Quat::from_xyzw(0.6, 0.0, 0.0, 0.8));
    let mut expected = PoseStore::default();
    for (digit, preset, size) in [
        ('0', Preset::YUp, 0.25),
        ('1', Preset::ZUp, 1.25),
        ('2', Preset::NegYUp, 2.25),
        ('3', custom, 3.25),
    ] {
        expected.set(&identity(digit), saved(preset, size));
    }
    assert_eq!(store, expected);
}
