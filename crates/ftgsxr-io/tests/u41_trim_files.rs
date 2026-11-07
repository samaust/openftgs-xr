// SPDX-License-Identifier: Apache-2.0

//! U-41 (§3.1, `cpu` lane): E17's trim in `poses.json` and in a scene file (§1.11): round
//! trips of each mode; a file without `trim` reads `None`; a `trim` with min > max, with
//! r ≤ 0, or with a value that is not a finite f32 reads as `None` with a warning, as does
//! one that breaks its schema; the files' versions unchanged. JSON carries no NaN, so a
//! NaN reaches only the writers, which refuse it (Q-53, Q-372).

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use ftgsxr_core::{EndRule, LoopMode, Pose, Preset, Trim, TrimMode, Vec3};
use ftgsxr_io::{
    ClockState, EnvironmentSettings, IoError, PoseStore, SavedPose, SceneFile, SceneModel,
    SourceSpec, read_scene, write_scene,
};
use ftgsxr_testkit::scratch_dir;
use serde_json::Value;

fn scratch(name: &str) -> PathBuf {
    scratch_dir(env!("CARGO_TARGET_TMPDIR"), crate::test_file_stem(), name)
}

fn identity(digit: char) -> String {
    std::iter::repeat_n(digit, 64).collect()
}

/// A trim of mode `mode` with both shapes away from zero.
fn trim(mode: TrimMode) -> Trim {
    Trim {
        mode,
        box_min: Vec3::new(-1.5, 0.25, -3.0),
        box_max: Vec3::new(2.0, 1.75, 0.5),
        center: Vec3::new(0.1, -0.2, 0.3),
        radius: 2.5,
    }
}

const MODES: [TrimMode; 3] = [TrimMode::Off, TrimMode::Box, TrimMode::Sphere];

fn saved(trim: Option<Trim>) -> SavedPose {
    SavedPose {
        pose: Pose::IDENTITY,
        preset: Preset::ZUp,
        fit_size_m: 1.0,
        saved_at: "2026-11-07T12:00:00Z".into(),
        trim,
    }
}

fn scene_model(digit: char, trim: Option<Trim>) -> SceneModel {
    SceneModel {
        source: SourceSpec {
            kind: "dir".into(),
            location: "/srv/models".into(),
            name: "/srv/models".into(),
            flavor: None,
        },
        identity: identity(digit),
        name: "m".into(),
        pose: Pose::IDENTITY,
        visible: true,
        mode: "splats".into(),
        time_offset_s: 0.0,
        end_rule: EndRule::Hold,
        lighting: None,
        audio: None,
        trim,
    }
}

fn scene(models: Vec<SceneModel>) -> SceneFile {
    SceneFile {
        models,
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
    }
}

/// Counts the warnings logged on this thread while `f` runs.
struct WarnCounter(Arc<AtomicUsize>);

impl tracing::Subscriber for WarnCounter {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        if *event.metadata().level() == tracing::Level::WARN {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

fn warnings<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let count = Arc::new(AtomicUsize::new(0));
    let out = tracing::subscriber::with_default(WarnCounter(count.clone()), f);
    (out, count.load(Ordering::SeqCst))
}

/// The JSON file at `path`.
fn json(path: &std::path::Path) -> Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

#[test]
fn u41_pose_store_round_trips_each_mode() {
    let dir = scratch("poses_round_trip");
    let path = dir.join("poses.json");
    let mut store = PoseStore::default();
    for (mode, digit) in MODES.into_iter().zip(['a', 'b', 'c']) {
        store.set(&identity(digit), saved(Some(trim(mode))));
    }
    store.set(&identity('d'), saved(None));
    store.save(&path).unwrap();
    let (back, warned) = warnings(|| PoseStore::load(&path).unwrap());
    assert_eq!(back, store);
    assert_eq!(warned, 0);
    let doc = json(&path);
    assert_eq!(doc["version"], 1);
    assert_eq!(doc[identity('b')]["trim"]["mode"], "box");
    assert_eq!(doc[identity('c')]["trim"]["sphere"]["radius"], 2.5);
    // No trim, no key.
    assert!(doc[identity('d')].get("trim").is_none());
}

#[test]
fn u41_scene_round_trips_each_mode() {
    let dir = scratch("scene_round_trip");
    let path = dir.join("s.ftgsxr-scene.json");
    let s = scene(vec![
        scene_model('a', Some(trim(TrimMode::Off))),
        scene_model('b', Some(trim(TrimMode::Box))),
        scene_model('c', Some(trim(TrimMode::Sphere))),
        scene_model('d', None),
    ]);
    write_scene(&path, &s).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let (back, warned) = warnings(|| read_scene(&bytes).unwrap());
    assert_eq!(back, s);
    assert_eq!(warned, 0);
    let doc = json(&path);
    assert_eq!(doc["version"], 1);
    assert_eq!(doc["models"][2]["trim"]["mode"], "sphere");
    assert!(doc["models"][3].get("trim").is_none());
}

/// A pose file with one entry whose `trim` is the JSON text `trim` (no key when `None`).
fn poses_with(trim: Option<&str>) -> String {
    let trim = trim.map_or(String::new(), |t| format!(r#", "trim": {t}"#));
    format!(
        r#"{{"format": "ftgsxr-poses", "version": 1, "{}": {{"translation": [0, 0, 0],
        "rotation": [0, 0, 0, 1], "scale": 1, "preset": "y-up", "fit_size_m": 1,
        "saved_at": "2026-11-07T12:00:00Z"{trim}}}}}"#,
        identity('e')
    )
}

/// A scene file with one model whose `trim` is the JSON text `trim` (no key when `None`).
fn scene_with(trim: Option<&str>) -> String {
    let trim = trim.map_or(String::new(), |t| format!(r#", "trim": {t}"#));
    format!(
        r#"{{"format": "ftgsxr-scene", "version": 1, "models": [{{
        "source": {{"kind": "dir", "location": "/srv"}}, "identity": "{}", "name": "m",
        "pose": {{"translation": [0, 0, 0], "rotation": [0, 0, 0, 1], "scale": 1}},
        "visible": true, "mode": "splats", "time_offset_s": 0, "end_rule": "hold"{trim}}}],
        "clock": {{"t": 0, "speed": 1, "loop": "loop"}}, "environment": {{}},
        "bookmarks": []}}"#,
        identity('e')
    )
}

/// The trim each file reads from `text`, and the warnings logged.
fn read_both(dir: &std::path::Path, trim: Option<&str>) -> [(Option<Trim>, usize); 2] {
    let path = dir.join("poses.json");
    std::fs::write(&path, poses_with(trim)).unwrap();
    let pose = warnings(|| {
        PoseStore::load(&path)
            .unwrap()
            .get(&identity('e'))
            .unwrap()
            .trim
    });
    let scene = warnings(|| read_scene(scene_with(trim).as_bytes()).unwrap().models[0].trim);
    [pose, scene]
}

const SPHERE: &str = r#"{"center": [0, 0, 0], "radius": 1}"#;
const BOX: &str = r#"{"min": [0, 0, 0], "max": [1, 1, 1]}"#;

#[test]
fn u41_a_file_without_trim_reads_none() {
    let dir = scratch("absent");
    for (t, warned) in read_both(&dir, None) {
        assert_eq!((t, warned), (None, 0));
    }
    for (t, warned) in read_both(&dir, Some("null")) {
        assert_eq!((t, warned), (None, 0));
    }
}

#[test]
fn u41_a_broken_trim_reads_none_with_a_warning() {
    let dir = scratch("broken");
    let broken = [
        // min > max on y.
        format!(
            r#"{{"mode": "box", "box": {{"min": [0, 2, 0], "max": [1, 1, 1]}}, "sphere": {SPHERE}}}"#
        ),
        // r = 0 and r < 0.
        format!(
            r#"{{"mode": "sphere", "box": {BOX}, "sphere": {{"center": [0, 0, 0], "radius": 0}}}}"#
        ),
        format!(
            r#"{{"mode": "sphere", "box": {BOX}, "sphere": {{"center": [0, 0, 0], "radius": -1}}}}"#
        ),
        // Not a finite f32.
        format!(
            r#"{{"mode": "box", "box": {{"min": [0, 0, 0], "max": [1e39, 1, 1]}}, "sphere": {SPHERE}}}"#
        ),
        // Malformed: an unknown mode, a missing shape, a wrong type.
        format!(r#"{{"mode": "cube", "box": {BOX}, "sphere": {SPHERE}}}"#),
        format!(r#"{{"mode": "box", "box": {BOX}}}"#),
        r#""box""#.to_string(),
        // A bad shape counts whatever the mode, off included.
        format!(
            r#"{{"mode": "off", "box": {BOX}, "sphere": {{"center": [0, 0, 0], "radius": 0}}}}"#
        ),
    ];
    for t in &broken {
        for (read, warned) in read_both(&dir, Some(t)) {
            assert_eq!((read, warned), (None, 1), "{t}");
        }
    }
    // The rest of the file still reads: the version and the pose are unchanged.
    let path = dir.join("poses.json");
    std::fs::write(&path, poses_with(Some(&broken[0]))).unwrap();
    let store = warnings(|| PoseStore::load(&path).unwrap()).0;
    assert_eq!(store.get(&identity('e')).unwrap().pose, Pose::IDENTITY);
}

#[test]
fn u41_the_writers_refuse_a_trim_the_readers_would_drop() {
    let dir = scratch("refuse");
    let nan = Trim {
        radius: f32::NAN,
        ..trim(TrimMode::Sphere)
    };
    let crossed = Trim {
        box_min: Vec3::splat(2.0),
        ..trim(TrimMode::Box)
    };
    for bad in [nan, crossed] {
        let mut store = PoseStore::default();
        store.set(&identity('a'), saved(Some(bad)));
        let path = dir.join("poses.json");
        assert!(matches!(store.save(&path), Err(IoError::Scene(_))));
        assert!(!path.exists());
        let path = dir.join("s.ftgsxr-scene.json");
        let s = scene(vec![scene_model('a', Some(bad))]);
        assert!(matches!(write_scene(&path, &s), Err(IoError::Scene(_))));
        assert!(!path.exists());
    }
}
