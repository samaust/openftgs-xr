// SPDX-License-Identifier: Apache-2.0

//! The player's scene files (`ftgsxr-scene` v1) and its pose store `poses.json`
//! (`ftgsxr-poses` v1): §1.5 Scene files and Poses, §1.11, D-23. The JSON shapes are
//! Q-52's, the reader rules and errors Q-53's.

use std::collections::BTreeMap;
use std::path::Path;

use ftgsxr_core::{CoreError, EndRule, LoopMode, Pose, Preset, Quat, Vec3};
use serde_json::{Map, Number, Value};

use crate::settings::{
    ENVIRONMENT_MODES, LIGHTING_MODES, LOOP_MODES, RENDER_MODES, SOURCE_KINDS, flavor_name,
    flavor_of, loop_mode_name, loop_mode_of,
};
use crate::util::{version_is_1, write_atomic};
use crate::{EnvironmentSettings, IoError, ManualLighting, SourceSpec};

/// A scene file's `format` (§1.11).
const SCENE_FORMAT: &str = "ftgsxr-scene";
/// The pose store's `format` (§1.11).
const POSES_FORMAT: &str = "ftgsxr-poses";
/// The most bookmarks a scene holds (§1.7).
const MAX_BOOKMARKS: usize = 9;
/// The smallest rotation norm accepted, as §1.2's for a model's rotations (Q-53).
const MIN_NORM: f64 = 1e-8;
/// The end rules' spellings (§1.6, Q-52).
const END_RULES: &[&str] = &["hold", "extrapolate"];
/// The camera modes' spellings (§1.7, Q-52).
const CAMERA_MODES: &[&str] = &["fly", "orbit"];
/// The presets' spellings (§1.11).
const PRESETS: &[&str] = &["y-up", "z-up", "neg-y-up", "custom"];

/// The playback clock a scene restores (§1.6).
#[derive(Clone, Debug, PartialEq)]
pub struct ClockState {
    /// The clock time T in seconds.
    pub t: f64,
    /// The speed, signed: E9's \[−4, 4\] \\ {0}, which the reader does not check.
    pub speed: f64,
    /// The loop mode, the file's key `loop`.
    pub loop_mode: LoopMode,
    /// The playback range \[T\_a, T\_b\] in seconds, `None` for the whole span.
    pub range: Option<(f64, f64)>,
}

/// A desktop camera: the scene's camera and its bookmarks (§1.7).
#[derive(Clone, Debug, PartialEq)]
pub struct CameraState {
    /// The camera's pose in stage space (scale unused).
    pub pose: Pose,
    /// The vertical field of view in degrees.
    pub fov_deg: f32,
    /// The orbit mode rather than the fly mode, the file's `mode` (Q-52).
    pub orbit: bool,
}

/// A scene file's content (§1.5 Scene files, §1.11).
#[derive(Clone, Debug, PartialEq)]
pub struct SceneFile {
    /// Every instance.
    pub models: Vec<SceneModel>,
    /// The clock.
    pub clock: ClockState,
    /// The environment settings of §1.9.
    pub environment: EnvironmentSettings,
    /// The spatial anchor's id, when one was created (§1.9).
    pub anchor: Option<String>,
    /// The XR rig's pose.
    pub rig: Option<Pose>,
    /// The desktop camera.
    pub camera: Option<CameraState>,
    /// The desktop bookmarks, at most nine (§1.7).
    pub bookmarks: Vec<CameraState>,
}

/// One instance of a scene file (§1.5 Instances, §1.11).
#[derive(Clone, Debug, PartialEq)]
pub struct SceneModel {
    /// The model's source; `location` is stored and returned as given (Q-3).
    pub source: SourceSpec,
    /// The model's identity, 64 lowercase hexadecimal digits (D-40, §1.14).
    pub identity: String,
    /// The instance's name.
    pub name: String,
    /// The pose P\_m.
    pub pose: Pose,
    /// Visibility.
    pub visible: bool,
    /// The render mode: `splats`, `centers`, `both` or `hidden` (§1.5, Q-50).
    pub mode: String,
    /// The time offset o\_m in seconds (E9).
    pub time_offset_s: f64,
    /// The end rule (E9).
    pub end_rule: EndRule,
    /// The color transform's manual sliders (E14), `None` for the identity.
    pub lighting: Option<ManualLighting>,
    /// The audio sidecar's path or URL, as given (§1.10, Q-3).
    pub audio: Option<String>,
}

/// The pose store `poses.json`: saved poses by model identity (§1.5 Poses, §1.11).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct PoseStore {
    poses: BTreeMap<String, SavedPose>,
}

/// One saved pose (§1.5 Poses, §1.11).
#[derive(Clone, Debug, PartialEq)]
pub struct SavedPose {
    /// The instance's pose.
    pub pose: Pose,
    /// The orientation preset.
    pub preset: Preset,
    /// The fit size L\_fit in metres.
    pub fit_size_m: f32,
    /// When it was saved, RFC 3339.
    pub saved_at: String,
}

/// Reads a scene file (`ftgsxr-scene` v1, §1.11) from its bytes. E-JSON when they are not
/// JSON, E-SCENE-FORMAT when the document breaks the schema (Q-53). The files it names
/// are not opened: whether they exist is the library's question (§1.5).
// Q-3, Q-52, Q-53
pub fn read_scene(json: &[u8]) -> Result<SceneFile, IoError> {
    let doc = parse(json, SCENE_FORMAT)?;
    scene_of(&doc)
}

/// Writes `scene` to `path` as `ftgsxr-scene` v1 (§1.11): to the sibling
/// `<file name>.tmp`, then renamed over `path`; the directory is created when missing.
/// E-SCENE-FORMAT, writing nothing, for a value `read_scene` would refuse; E-IO when the
/// file cannot be written (Q-52, Q-53).
// Q-3, Q-52, Q-53
pub fn write_scene(path: &Path, scene: &SceneFile) -> Result<(), IoError> {
    let doc = scene_doc(scene)?;
    scene_of(&doc)?;
    write_json(path, &doc)
}

impl PoseStore {
    /// Reads the pose store at `path` (`ftgsxr-poses` v1, §1.11); a missing file is an
    /// empty store. A key that is not a model identity (64 lowercase hexadecimal digits;
    /// an identity spelled in uppercase hex is not one) is an unknown key: it is ignored,
    /// and the next `save` drops it (Q-52). E-JSON when the file is not JSON, E-SCENE-FORMAT when it
    /// breaks the schema, E-IO when it cannot be read (Q-52, Q-53).
    // Q-52, Q-53
    pub fn load(path: &Path) -> Result<PoseStore, IoError> {
        let bytes = match std::fs::read(path) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(PoseStore::default());
            }
            other => other?,
        };
        let doc = parse(&bytes, POSES_FORMAT)?;
        store_of(&doc)
    }

    /// Writes the store to `path` (§1.11), atomically, identities in sorted order; the
    /// directory is created when missing. E-SCENE-FORMAT, writing nothing, for an entry
    /// `load` would refuse or ignore; E-IO when the file cannot be written (Q-52,
    /// Q-53).
    // Q-52, Q-53
    pub fn save(&self, path: &Path) -> Result<(), IoError> {
        let doc = self.doc()?;
        if store_of(&doc)? != *self {
            let key = self.poses.keys().find(|k| !is_identity(k));
            return Err(schema(
                key.map(String::as_str).unwrap_or(POSES_FORMAT),
                "not a model identity (64 lowercase hexadecimal digits)",
            ));
        }
        write_json(path, &doc)
    }

    /// The pose saved under `identity`, if any.
    pub fn get(&self, identity: &str) -> Option<&SavedPose> {
        self.poses.get(identity)
    }

    /// Saves `pose` under `identity`, replacing any earlier one.
    pub fn set(&mut self, identity: &str, pose: SavedPose) {
        self.poses.insert(identity.to_string(), pose);
    }

    /// Forgets the pose saved under `identity`; nothing happens when there is none.
    pub fn forget(&mut self, identity: &str) {
        self.poses.remove(identity);
    }

    /// The `ftgsxr-poses` v1 document of the store.
    fn doc(&self) -> Result<Value, IoError> {
        let mut map = header(POSES_FORMAT);
        for (identity, saved) in &self.poses {
            let mut entry = pose_doc(&saved.pose, identity)?;
            let (name, custom) = match saved.preset {
                Preset::YUp => ("y-up", None),
                Preset::ZUp => ("z-up", None),
                Preset::NegYUp => ("neg-y-up", None),
                Preset::Custom(q) => ("custom", Some(q)),
            };
            entry.insert("preset".into(), name.into());
            if let Some(q) = custom {
                let path = format!("{identity}.preset_rotation");
                entry.insert("preset_rotation".into(), floats(&q.to_array(), &path)?);
            }
            let path = format!("{identity}.fit_size_m");
            entry.insert("fit_size_m".into(), float(saved.fit_size_m, &path)?);
            entry.insert("saved_at".into(), saved.saved_at.clone().into());
            map.insert(identity.clone(), Value::Object(entry));
        }
        Ok(Value::Object(map))
    }
}

// ---------------------------------------------------------------------------------------
// Reading

/// An E-SCENE-FORMAT error at the key path `at`.
fn schema(at: &str, detail: impl std::fmt::Display) -> IoError {
    IoError::Scene(format!("{at}: {detail}"))
}

/// The JSON document of `bytes`, checked to be an object of `format` and major version 1
/// (§1.11's versioning rule; Q-37's `version` forms).
fn parse(bytes: &[u8], format: &str) -> Result<Value, IoError> {
    let doc: Value = serde_json::from_slice(bytes)
        .map_err(|e| CoreError::Json(format!("the {format} file ({e})")))?;
    let map = object(&doc, format)?;
    match map.get("format") {
        Some(Value::String(f)) if f == format => {}
        Some(other) => return Err(schema("format", format!("{other}, expected {format}"))),
        None => return Err(schema("format", "missing")),
    }
    match map.get("version") {
        Some(v) if version_is_1(v) => Ok(doc),
        Some(v) => Err(schema(
            "version",
            format!("{v}, this player reads version 1"),
        )),
        None => Err(schema("version", "missing")),
    }
}

/// `v` as an object.
fn object<'a>(v: &'a Value, at: &str) -> Result<&'a Map<String, Value>, IoError> {
    v.as_object()
        .ok_or_else(|| schema(at, "expected an object"))
}

/// The required key `key` of `map`.
fn field<'a>(map: &'a Map<String, Value>, key: &str, at: &str) -> Result<&'a Value, IoError> {
    map.get(key)
        .ok_or_else(|| schema(&format!("{at}.{key}"), "missing"))
}

/// The optional key `key` of `map`: `None` when absent or null.
fn optional<'a>(map: &'a Map<String, Value>, key: &str) -> Option<&'a Value> {
    map.get(key).filter(|v| !v.is_null())
}

/// `v` as a finite number.
fn number(v: &Value, at: &str) -> Result<f64, IoError> {
    v.as_f64()
        .filter(|x| x.is_finite())
        .ok_or_else(|| schema(at, format!("{v}, expected a number")))
}

/// `v` as a finite f32.
fn number32(v: &Value, at: &str) -> Result<f32, IoError> {
    let x = number(v, at)? as f32;
    if x.is_finite() {
        Ok(x)
    } else {
        Err(schema(at, format!("{v} does not fit a 32-bit float")))
    }
}

/// `v` as an array of `N` finite f32.
fn numbers32<const N: usize>(v: &Value, at: &str) -> Result<[f32; N], IoError> {
    let items = v
        .as_array()
        .filter(|a| a.len() == N)
        .ok_or_else(|| schema(at, format!("expected an array of {N} numbers")))?;
    let mut out = [0.0; N];
    for (i, (o, item)) in out.iter_mut().zip(items).enumerate() {
        *o = number32(item, &format!("{at}[{i}]"))?;
    }
    Ok(out)
}

/// `v` as a boolean.
fn boolean(v: &Value, at: &str) -> Result<bool, IoError> {
    v.as_bool()
        .ok_or_else(|| schema(at, format!("{v}, expected true or false")))
}

/// `v` as a string.
fn text<'a>(v: &'a Value, at: &str) -> Result<&'a str, IoError> {
    v.as_str()
        .ok_or_else(|| schema(at, format!("{v}, expected a string")))
}

/// `v` as one of the spellings `values`.
fn choice<'a>(v: &'a Value, at: &str, values: &[&str]) -> Result<&'a str, IoError> {
    let s = text(v, at)?;
    if values.contains(&s) {
        Ok(s)
    } else {
        Err(schema(at, format!("{s:?}, expected one of {values:?}")))
    }
}

/// Whether `s` is a model identity: 64 lowercase hexadecimal digits (D-40, §1.14).
fn is_identity(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// A quaternion stored (x, y, z, w) (§1.14), kept as written, of norm above 10⁻⁸.
fn rotation(v: &Value, at: &str) -> Result<Quat, IoError> {
    let q = numbers32::<4>(v, at)?;
    let norm = q.iter().map(|&c| f64::from(c).powi(2)).sum::<f64>().sqrt();
    if norm > MIN_NORM {
        Ok(Quat::from_array(q))
    } else {
        Err(schema(at, "a rotation of norm 0"))
    }
}

/// A pose `{"translation", "rotation", "scale"}` held in `map` (§1.11).
fn pose_in(map: &Map<String, Value>, at: &str) -> Result<Pose, IoError> {
    let translation = numbers32::<3>(field(map, "translation", at)?, &format!("{at}.translation"))?;
    let rotation = rotation(field(map, "rotation", at)?, &format!("{at}.rotation"))?;
    let scale_at = format!("{at}.scale");
    let scale = number32(field(map, "scale", at)?, &scale_at)?;
    if scale <= 0.0 {
        return Err(schema(
            &scale_at,
            format!("{scale}, expected a positive number"),
        ));
    }
    Ok(Pose {
        translation: Vec3::from_array(translation),
        rotation,
        scale,
    })
}

/// A pose object `v`.
fn pose_of(v: &Value, at: &str) -> Result<Pose, IoError> {
    pose_in(object(v, at)?, at)
}

/// The pose under the key `pose` of the object `v` (the rig's and a camera's).
fn posed<'a>(v: &'a Value, at: &str) -> Result<(&'a Map<String, Value>, Pose), IoError> {
    let map = object(v, at)?;
    let pose = pose_of(field(map, "pose", at)?, &format!("{at}.pose"))?;
    Ok((map, pose))
}

/// A camera `{"pose", "fov_deg", "mode"}` (§1.7, Q-52).
fn camera_of(v: &Value, at: &str) -> Result<CameraState, IoError> {
    let (map, pose) = posed(v, at)?;
    let fov_deg = number32(field(map, "fov_deg", at)?, &format!("{at}.fov_deg"))?;
    let mode = choice(field(map, "mode", at)?, &format!("{at}.mode"), CAMERA_MODES)?;
    Ok(CameraState {
        pose,
        fov_deg,
        orbit: mode == "orbit",
    })
}

/// E14's manual sliders `{"exposure_ev", "temperature", "tint"}`.
fn lighting_of(v: &Value, at: &str) -> Result<ManualLighting, IoError> {
    let map = object(v, at)?;
    let get = |key: &str| number32(field(map, key, at)?, &format!("{at}.{key}"));
    Ok(ManualLighting {
        exposure_ev: get("exposure_ev")?,
        temperature: get("temperature")?,
        tint: get("tint")?,
    })
}

/// A model's `source` (Q-52): `kind` and `location` required, `name` and `flavor`
/// optional.
fn source_of(v: &Value, at: &str) -> Result<SourceSpec, IoError> {
    let map = object(v, at)?;
    let kind = choice(field(map, "kind", at)?, &format!("{at}.kind"), SOURCE_KINDS)?;
    let location = text(field(map, "location", at)?, &format!("{at}.location"))?;
    let name = match optional(map, "name") {
        Some(n) => text(n, &format!("{at}.name"))?,
        None => location,
    };
    let flavor = match optional(map, "flavor") {
        Some(f) => {
            let f_at = format!("{at}.flavor");
            let spelled = text(f, &f_at)?;
            Some(flavor_of(spelled).ok_or_else(|| schema(&f_at, format!("{spelled:?}")))?)
        }
        None => None,
    };
    Ok(SourceSpec {
        kind: kind.into(),
        location: location.into(),
        name: name.into(),
        flavor,
    })
}

/// One instance of `models`.
fn model_of(v: &Value, at: &str) -> Result<SceneModel, IoError> {
    let map = object(v, at)?;
    let sub = |key: &str| format!("{at}.{key}");
    let identity = text(field(map, "identity", at)?, &sub("identity"))?;
    if !is_identity(identity) {
        return Err(schema(
            &sub("identity"),
            format!("{identity:?} is not 64 lowercase hexadecimal digits"),
        ));
    }
    let end_rule = match choice(field(map, "end_rule", at)?, &sub("end_rule"), END_RULES)? {
        "hold" => EndRule::Hold,
        _ => EndRule::Extrapolate,
    };
    Ok(SceneModel {
        source: source_of(field(map, "source", at)?, &sub("source"))?,
        identity: identity.into(),
        name: text(field(map, "name", at)?, &sub("name"))?.into(),
        pose: pose_of(field(map, "pose", at)?, &sub("pose"))?,
        visible: boolean(field(map, "visible", at)?, &sub("visible"))?,
        mode: choice(field(map, "mode", at)?, &sub("mode"), RENDER_MODES)?.into(),
        time_offset_s: number(field(map, "time_offset_s", at)?, &sub("time_offset_s"))?,
        end_rule,
        lighting: optional(map, "lighting")
            .map(|l| lighting_of(l, &sub("lighting")))
            .transpose()?,
        audio: optional(map, "audio")
            .map(|a| text(a, &sub("audio")).map(String::from))
            .transpose()?,
    })
}

/// The `clock` (§1.6).
fn clock_of(v: &Value, at: &str) -> Result<ClockState, IoError> {
    let map = object(v, at)?;
    let sub = |key: &str| format!("{at}.{key}");
    let spelled = choice(field(map, "loop", at)?, &sub("loop"), LOOP_MODES)?;
    let loop_mode =
        loop_mode_of(spelled).ok_or_else(|| schema(&sub("loop"), format!("{spelled:?}")))?;
    let range = match optional(map, "range") {
        Some(r) => {
            let r_at = sub("range");
            let pair = r
                .as_array()
                .filter(|a| a.len() == 2)
                .ok_or_else(|| schema(&r_at, "expected [a, b] or null"))?;
            let a = number(&pair[0], &format!("{r_at}[0]"))?;
            let b = number(&pair[1], &format!("{r_at}[1]"))?;
            if a > b {
                return Err(schema(&r_at, format!("[{a}, {b}] has a > b")));
            }
            Some((a, b))
        }
        None => None,
    };
    Ok(ClockState {
        t: number(field(map, "t", at)?, &sub("t"))?,
        speed: number(field(map, "speed", at)?, &sub("speed"))?,
        loop_mode,
        range,
    })
}

/// The `environment` (§1.9, Q-52): the object is required, but each key of
/// `EnvironmentSettings` (and of its `lighting_manual`) may be absent and then takes
/// `EnvironmentSettings::default()`'s value (Q-50); a key present with the wrong type or
/// spelling is E-SCENE-FORMAT (Q-53).
// Q-50, Q-52, Q-53
fn environment_of(v: &Value, at: &str) -> Result<EnvironmentSettings, IoError> {
    let map = object(v, at)?;
    let d = EnvironmentSettings::default();
    let sub = |key: &str| format!("{at}.{key}");
    let f = |key: &str, default: f32| map.get(key).map_or(Ok(default), |v| number32(v, &sub(key)));
    let b = |key: &str, default: bool| map.get(key).map_or(Ok(default), |v| boolean(v, &sub(key)));
    let c = |key: &str, default: String, values: &[&str]| {
        map.get(key).map_or(Ok(default), |v| {
            choice(v, &sub(key), values).map(String::from)
        })
    };
    let lighting_manual = match map.get("lighting_manual") {
        None => d.lighting_manual,
        Some(v) => {
            let at = sub("lighting_manual");
            let lighting = object(v, &at)?;
            let d = d.lighting_manual;
            let f = |key: &str, default: f32| {
                lighting
                    .get(key)
                    .map_or(Ok(default), |v| number32(v, &format!("{at}.{key}")))
            };
            ManualLighting {
                exposure_ev: f("exposure_ev", d.exposure_ev)?,
                temperature: f("temperature", d.temperature)?,
                tint: f("tint", d.tint)?,
            }
        }
    };
    Ok(EnvironmentSettings {
        mode: c("mode", d.mode, ENVIRONMENT_MODES)?,
        background: match map.get("background") {
            None => d.background,
            Some(v) => numbers32::<3>(v, &sub("background"))?,
        },
        passthrough_brightness: f("passthrough_brightness", d.passthrough_brightness)?,
        passthrough_contrast: f("passthrough_contrast", d.passthrough_contrast)?,
        passthrough_saturation: f("passthrough_saturation", d.passthrough_saturation)?,
        occlusion: b("occlusion", d.occlusion)?,
        occlusion_soft: b("occlusion_soft", d.occlusion_soft)?,
        occlusion_delta_m: f("occlusion_delta_m", d.occlusion_delta_m)?,
        hand_removal: b("hand_removal", d.hand_removal)?,
        lighting_mode: c("lighting_mode", d.lighting_mode, LIGHTING_MODES)?,
        lighting_manual,
        lighting_gamma: f("lighting_gamma", d.lighting_gamma)?,
        lighting_reference: optional(map, "lighting_reference")
            .map(|r| numbers32::<4>(r, &sub("lighting_reference")))
            .transpose()?,
        contact_shadows: b("contact_shadows", d.contact_shadows)?,
    })
}

/// The scene of a document `parse` accepted.
fn scene_of(doc: &Value) -> Result<SceneFile, IoError> {
    let at = SCENE_FORMAT;
    let map = object(doc, at)?;
    let list = |key: &str| {
        field(map, key, at)?
            .as_array()
            .ok_or_else(|| schema(key, "expected an array"))
    };
    let models = list("models")?
        .iter()
        .enumerate()
        .map(|(i, m)| model_of(m, &format!("models[{i}]")))
        .collect::<Result<Vec<_>, _>>()?;
    let bookmarks = list("bookmarks")?;
    if bookmarks.len() > MAX_BOOKMARKS {
        return Err(schema(
            "bookmarks",
            format!("{} bookmarks, at most {MAX_BOOKMARKS}", bookmarks.len()),
        ));
    }
    let bookmarks = bookmarks
        .iter()
        .enumerate()
        .map(|(i, c)| camera_of(c, &format!("bookmarks[{i}]")))
        .collect::<Result<Vec<_>, _>>()?;
    let anchor = match optional(map, "anchor") {
        Some(a) => {
            let a_map = object(a, "anchor")?;
            Some(text(field(a_map, "uuid", "anchor")?, "anchor.uuid")?.to_string())
        }
        None => None,
    };
    Ok(SceneFile {
        models,
        clock: clock_of(field(map, "clock", at)?, "clock")?,
        environment: environment_of(field(map, "environment", at)?, "environment")?,
        anchor,
        rig: optional(map, "rig")
            .map(|r| posed(r, "rig").map(|(_, p)| p))
            .transpose()?,
        camera: optional(map, "camera")
            .map(|c| camera_of(c, "camera"))
            .transpose()?,
        bookmarks,
    })
}

/// The pose store of a document `parse` accepted: every key that is a model identity;
/// other keys are unknown keys, ignored (Q-52).
fn store_of(doc: &Value) -> Result<PoseStore, IoError> {
    let mut poses = BTreeMap::new();
    for (key, v) in object(doc, POSES_FORMAT)? {
        if !is_identity(key) {
            continue;
        }
        let map = object(v, key)?;
        let sub = |k: &str| format!("{key}.{k}");
        let pose = pose_in(map, key)?;
        let preset = match choice(field(map, "preset", key)?, &sub("preset"), PRESETS)? {
            "y-up" => Preset::YUp,
            "z-up" => Preset::ZUp,
            "neg-y-up" => Preset::NegYUp,
            _ => Preset::Custom(rotation(
                field(map, "preset_rotation", key)?,
                &sub("preset_rotation"),
            )?),
        };
        let fit_at = sub("fit_size_m");
        let fit_size_m = number32(field(map, "fit_size_m", key)?, &fit_at)?;
        if fit_size_m <= 0.0 {
            return Err(schema(
                &fit_at,
                format!("{fit_size_m}, expected a positive number"),
            ));
        }
        let saved_at = text(field(map, "saved_at", key)?, &sub("saved_at"))?.to_string();
        poses.insert(
            key.clone(),
            SavedPose {
                pose,
                preset,
                fit_size_m,
                saved_at,
            },
        );
    }
    Ok(PoseStore { poses })
}

// ---------------------------------------------------------------------------------------
// Writing

/// A document's `format` and `version` (§1.11).
fn header(format: &str) -> Map<String, Value> {
    let mut map = Map::new();
    map.insert("format".into(), format.into());
    map.insert("version".into(), 1.into());
    map
}

/// An f32 as the JSON number its shortest decimal form names, so the file reads `0.1`
/// rather than `0.10000000149011612` and reads back to the same f32 (Q-52); a
/// non-finite value is refused at the key path `at` (Q-53).
fn float(x: f32, at: &str) -> Result<Value, IoError> {
    let shortest: f64 = x.to_string().parse().unwrap_or(f64::from(x));
    Number::from_f64(shortest)
        .map(Value::Number)
        .ok_or_else(|| schema(at, format!("{x}, expected a finite number")))
}

/// An f64 as a JSON number; a non-finite value is refused (Q-53).
fn float64(x: f64, at: &str) -> Result<Value, IoError> {
    Number::from_f64(x)
        .map(Value::Number)
        .ok_or_else(|| schema(at, format!("{x}, expected a finite number")))
}

/// `values` as a JSON array of numbers.
fn floats(values: &[f32], at: &str) -> Result<Value, IoError> {
    values
        .iter()
        .enumerate()
        .map(|(i, &x)| float(x, &format!("{at}[{i}]")))
        .collect::<Result<Vec<_>, _>>()
        .map(Value::Array)
}

/// The pose object `{"translation", "rotation": [x, y, z, w], "scale"}` (§1.11, §1.14).
fn pose_doc(pose: &Pose, at: &str) -> Result<Map<String, Value>, IoError> {
    let mut map = Map::new();
    let sub = |key: &str| format!("{at}.{key}");
    map.insert(
        "translation".into(),
        floats(&pose.translation.to_array(), &sub("translation"))?,
    );
    map.insert(
        "rotation".into(),
        floats(&pose.rotation.to_array(), &sub("rotation"))?,
    );
    map.insert("scale".into(), float(pose.scale, &sub("scale"))?);
    Ok(map)
}

/// `{"pose": …}`, the rig's object and a camera's start.
fn posed_doc(pose: &Pose, at: &str) -> Result<Map<String, Value>, IoError> {
    let mut map = Map::new();
    map.insert(
        "pose".into(),
        Value::Object(pose_doc(pose, &format!("{at}.pose"))?),
    );
    Ok(map)
}

/// A camera `{"pose", "fov_deg", "mode"}`.
fn camera_doc(camera: &CameraState, at: &str) -> Result<Value, IoError> {
    let mut map = posed_doc(&camera.pose, at)?;
    map.insert(
        "fov_deg".into(),
        float(camera.fov_deg, &format!("{at}.fov_deg"))?,
    );
    let mode = if camera.orbit { "orbit" } else { "fly" };
    map.insert("mode".into(), mode.into());
    Ok(Value::Object(map))
}

/// E14's manual sliders.
fn lighting_doc(l: &ManualLighting, at: &str) -> Result<Value, IoError> {
    let mut map = Map::new();
    let sub = |key: &str| format!("{at}.{key}");
    map.insert(
        "exposure_ev".into(),
        float(l.exposure_ev, &sub("exposure_ev"))?,
    );
    map.insert(
        "temperature".into(),
        float(l.temperature, &sub("temperature"))?,
    );
    map.insert("tint".into(), float(l.tint, &sub("tint"))?);
    Ok(Value::Object(map))
}

/// One instance.
fn model_doc(m: &SceneModel, at: &str) -> Result<Value, IoError> {
    let sub = |key: &str| format!("{at}.{key}");
    let mut source = Map::new();
    source.insert("kind".into(), m.source.kind.clone().into());
    source.insert("location".into(), m.source.location.clone().into());
    source.insert("name".into(), m.source.name.clone().into());
    let flavor = m
        .source
        .flavor
        .map_or(Value::Null, |f| flavor_name(f).into());
    source.insert("flavor".into(), flavor);
    let mut map = Map::new();
    map.insert("source".into(), Value::Object(source));
    map.insert("identity".into(), m.identity.clone().into());
    map.insert("name".into(), m.name.clone().into());
    map.insert(
        "pose".into(),
        Value::Object(pose_doc(&m.pose, &sub("pose"))?),
    );
    map.insert("visible".into(), m.visible.into());
    map.insert("mode".into(), m.mode.clone().into());
    map.insert(
        "time_offset_s".into(),
        float64(m.time_offset_s, &sub("time_offset_s"))?,
    );
    let end_rule = match m.end_rule {
        EndRule::Hold => "hold",
        EndRule::Extrapolate => "extrapolate",
    };
    map.insert("end_rule".into(), end_rule.into());
    let lighting = match &m.lighting {
        Some(l) => lighting_doc(l, &sub("lighting"))?,
        None => Value::Null,
    };
    map.insert("lighting".into(), lighting);
    map.insert(
        "audio".into(),
        m.audio.clone().map_or(Value::Null, Value::from),
    );
    Ok(Value::Object(map))
}

/// The `clock`.
fn clock_doc(c: &ClockState) -> Result<Value, IoError> {
    let mut map = Map::new();
    map.insert("t".into(), float64(c.t, "clock.t")?);
    map.insert("speed".into(), float64(c.speed, "clock.speed")?);
    map.insert("loop".into(), loop_mode_name(c.loop_mode).into());
    let range = match c.range {
        Some((a, b)) => Value::Array(vec![
            float64(a, "clock.range[0]")?,
            float64(b, "clock.range[1]")?,
        ]),
        None => Value::Null,
    };
    map.insert("range".into(), range);
    Ok(Value::Object(map))
}

/// The `environment`: every key of `EnvironmentSettings` written, though the reader
/// lets any of them be absent (Q-52).
fn environment_doc(e: &EnvironmentSettings) -> Result<Value, IoError> {
    let at = |key: &str| format!("environment.{key}");
    let mut map = Map::new();
    map.insert("mode".into(), e.mode.clone().into());
    map.insert(
        "background".into(),
        floats(&e.background, &at("background"))?,
    );
    for (key, x) in [
        ("passthrough_brightness", e.passthrough_brightness),
        ("passthrough_contrast", e.passthrough_contrast),
        ("passthrough_saturation", e.passthrough_saturation),
    ] {
        map.insert(key.into(), float(x, &at(key))?);
    }
    map.insert("occlusion".into(), e.occlusion.into());
    map.insert("occlusion_soft".into(), e.occlusion_soft.into());
    map.insert(
        "occlusion_delta_m".into(),
        float(e.occlusion_delta_m, &at("occlusion_delta_m"))?,
    );
    map.insert("hand_removal".into(), e.hand_removal.into());
    map.insert("lighting_mode".into(), e.lighting_mode.clone().into());
    map.insert(
        "lighting_manual".into(),
        lighting_doc(&e.lighting_manual, &at("lighting_manual"))?,
    );
    map.insert(
        "lighting_gamma".into(),
        float(e.lighting_gamma, &at("lighting_gamma"))?,
    );
    let reference = match &e.lighting_reference {
        Some(r) => floats(r, &at("lighting_reference"))?,
        None => Value::Null,
    };
    map.insert("lighting_reference".into(), reference);
    map.insert("contact_shadows".into(), e.contact_shadows.into());
    Ok(Value::Object(map))
}

/// The `ftgsxr-scene` v1 document of `scene` (§1.11, Q-52).
fn scene_doc(scene: &SceneFile) -> Result<Value, IoError> {
    let mut map = header(SCENE_FORMAT);
    let models = scene
        .models
        .iter()
        .enumerate()
        .map(|(i, m)| model_doc(m, &format!("models[{i}]")))
        .collect::<Result<Vec<_>, _>>()?;
    map.insert("models".into(), Value::Array(models));
    map.insert("clock".into(), clock_doc(&scene.clock)?);
    map.insert("environment".into(), environment_doc(&scene.environment)?);
    let anchor = match &scene.anchor {
        Some(uuid) => {
            let mut a = Map::new();
            a.insert("uuid".into(), uuid.clone().into());
            Value::Object(a)
        }
        None => Value::Null,
    };
    map.insert("anchor".into(), anchor);
    let rig = match &scene.rig {
        Some(p) => Value::Object(posed_doc(p, "rig")?),
        None => Value::Null,
    };
    map.insert("rig".into(), rig);
    let camera = match &scene.camera {
        Some(c) => camera_doc(c, "camera")?,
        None => Value::Null,
    };
    map.insert("camera".into(), camera);
    let bookmarks = scene
        .bookmarks
        .iter()
        .enumerate()
        .map(|(i, c)| camera_doc(c, &format!("bookmarks[{i}]")))
        .collect::<Result<Vec<_>, _>>()?;
    map.insert("bookmarks".into(), Value::Array(bookmarks));
    Ok(Value::Object(map))
}

/// Writes `doc`, pretty-printed, to `path` atomically, creating the directory (§1.11).
fn write_json(path: &Path, doc: &Value) -> Result<(), IoError> {
    let mut text = serde_json::to_string_pretty(doc).map_err(std::io::Error::other)?;
    text.push('\n');
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    write_atomic(path, text.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `float`'s JSON number, printed and parsed back by serde_json then rounded to f32,
    /// is the f32 it came from, over a stride through every finite f32 bit pattern.
    #[test]
    fn float_reads_back_to_the_same_f32() {
        for bits in (0..=u32::MAX).step_by(9_973) {
            let x = f32::from_bits(bits);
            if !x.is_finite() {
                continue;
            }
            let text = float(x, "x").expect("finite").to_string();
            let back: Value = serde_json::from_str(&text).expect("JSON");
            let y = number32(&back, "x").expect("a number");
            assert_eq!(y.to_bits(), x.to_bits(), "{x:e} → {text}");
        }
        assert!(float(f32::NAN, "x").is_err());
        assert!(float(f32::INFINITY, "x").is_err());
    }
}
