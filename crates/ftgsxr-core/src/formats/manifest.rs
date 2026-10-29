// SPDX-License-Identifier: Apache-2.0

//! The `ftgs-scene` manifest reader (§1.5, §1.7, openftgs §2.2, openftgs
//! `io.read_scene_manifest`).

use std::fmt;

use serde::de::{Deserializer, IgnoredAny, MapAccess, Visitor};
use serde_json::{Map, Value};

use super::json::{self, invalid, required, shape};
use crate::types::{Camera, CoreError};

/// The manifest's `format`.
const FORMAT: &str = "ftgs-scene";
/// The splits an image may belong to (openftgs §2.2).
const SPLITS: [&str; 2] = ["train", "test"];
/// The camera keys an image may override, for moving or zooming cameras (openftgs §2.2).
const OVERRIDES: [&str; 2] = ["K", "world_to_camera"];

/// openftgs's `ftgs-scene` manifest (openftgs §2.2): the cameras and images the preset
/// hint and the training-camera frustums use (§1.5, §1.7).
#[derive(Clone, Debug, PartialEq)]
pub struct SceneManifest {
    /// The unit of the images' timestamps.
    pub time_unit: String,
    /// The named cameras, in the order of the file's `cameras` object.
    pub cameras: Vec<(String, Camera)>,
    /// The images, in file order.
    pub images: Vec<ManifestImage>,
}

/// One image entry of a `ftgs-scene` manifest (openftgs §2.2; §1.5, §1.7).
#[derive(Clone, Debug, PartialEq)]
pub struct ManifestImage {
    /// The name of its camera, a key of `SceneManifest::cameras`.
    pub camera: String,
    /// τ in the manifest's time unit.
    pub timestamp: f64,
    /// `train` or `test`.
    pub split: String,
    /// The frame index, when the entry gives one.
    pub frame_index: Option<i64>,
    /// The image's own camera when the entry overrides `K` or `world_to_camera`: the named
    /// camera with the overridden entries replaced, checked again.
    pub camera_override: Option<Camera>,
}

/// Reads and checks a `ftgs-scene` file (§1.5, §1.7): the `format`, a version of major 1,
/// a non-empty `time_unit`; `cameras` an object of pinhole cameras (an absent `model` is
/// pinhole), each checked as a request's camera is (an orthonormal rotation, an unskewed K),
/// kept in the file's key order; `images` a non-empty list whose entries name one of the
/// cameras and give a finite `timestamp`, a `split` of `train` or `test` and an optional
/// integer `frame_index`, an entry overriding `K` or `world_to_camera` getting a camera of
/// its own, checked again. The entries the player does not read (`path`, `mask`,
/// `metadata`) are not checked, and unknown keys are ignored (Q-26). Codes of Q-25.
// Q-25, Q-26
pub fn read_scene_manifest(json: &[u8]) -> Result<SceneManifest, CoreError> {
    let map = json::parse_object(json, FORMAT)?;
    json::check_header(&map, FORMAT)?;
    let time_unit = json::time_unit(&map)?;
    let raw_cameras = required(&map, "cameras")?;
    let raw_cameras = raw_cameras
        .as_object()
        .ok_or_else(|| shape("cameras", raw_cameras, "an object"))?;
    let mut cameras = Vec::with_capacity(raw_cameras.len());
    for name in camera_order(json)? {
        let value = &raw_cameras[&name];
        let what = format!("camera {name:?}");
        if let Some(model) = value.get("model")
            && model.as_str() != Some("pinhole")
        {
            return Err(invalid(&format!("{what} model"), model));
        }
        cameras.push((name.clone(), json::camera(&what, value)?));
    }
    let list = required(&map, "images")?;
    let list = list
        .as_array()
        .ok_or_else(|| shape("images", list, "a list"))?;
    if list.is_empty() {
        return Err(invalid("images", "[] (no image)"));
    }
    let images = list
        .iter()
        .enumerate()
        .map(|(k, v)| image(k, v, raw_cameras))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(SceneManifest {
        time_unit,
        cameras,
        images,
    })
}

/// Image entry `k`.
fn image(
    k: usize,
    value: &Value,
    raw_cameras: &Map<String, Value>,
) -> Result<ManifestImage, CoreError> {
    let what = |key: &str| format!("image {k} {key}");
    let map = value
        .as_object()
        .ok_or_else(|| shape(&format!("image {k}"), value, "an object"))?;
    let get = |key: &str| map.get(key).ok_or_else(|| CoreError::Keys(what(key)));
    let camera = match get("camera")? {
        Value::String(name) if raw_cameras.contains_key(name) => name.clone(),
        Value::String(name) => {
            return Err(invalid(
                &what("camera"),
                format!("{name:?} (no such camera)"),
            ));
        }
        other => return Err(shape(&what("camera"), other, "a string")),
    };
    let camera_override = if OVERRIDES.iter().any(|key| map.contains_key(*key)) {
        let mut merged = raw_cameras[&camera]
            .as_object()
            .expect("invariant: every camera was checked to be an object")
            .clone();
        for key in OVERRIDES {
            if let Some(v) = map.get(key) {
                merged.insert(key.to_owned(), v.clone());
            }
        }
        Some(json::camera(
            &format!("image {k} camera"),
            &Value::Object(merged),
        )?)
    } else {
        None
    };
    let timestamp = json::number(&what("timestamp"), get("timestamp")?)?;
    let split = match get("split")? {
        Value::String(s) if SPLITS.contains(&s.as_str()) => s.clone(),
        Value::String(s) => return Err(invalid(&what("split"), format!("{s:?}"))),
        other => return Err(shape(&what("split"), other, "a string")),
    };
    let frame_index = match map.get("frame_index") {
        None | Some(Value::Null) => None,
        Some(Value::Number(n)) if n.is_i64() => n.as_i64(),
        Some(Value::Number(n)) if n.is_u64() => {
            return Err(invalid(&what("frame_index"), n));
        }
        Some(other) => return Err(shape(&what("frame_index"), other, "an integer")),
    };
    Ok(ManifestImage {
        camera,
        timestamp,
        split,
        frame_index,
        camera_override,
    })
}

/// The keys of the top-level `cameras` object in file order, each once at its first
/// position (as Python's `dict` keeps them); serde_json's `Map` sorts its keys.
fn camera_order(json: &[u8]) -> Result<Vec<String>, CoreError> {
    struct Top;
    struct Keys;
    impl<'de> Visitor<'de> for Top {
        type Value = Vec<String>;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("a JSON object")
        }
        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Vec<String>, A::Error> {
            let mut order = Vec::new();
            while let Some(key) = map.next_key::<String>()? {
                if key == "cameras" {
                    // The last `cameras` entry wins, as in serde_json's `Map`.
                    order = map.next_value_seed(KeysSeed)?;
                } else {
                    map.next_value::<IgnoredAny>()?;
                }
            }
            Ok(order)
        }
    }
    struct KeysSeed;
    impl<'de> serde::de::DeserializeSeed<'de> for KeysSeed {
        type Value = Vec<String>;
        fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<Vec<String>, D::Error> {
            d.deserialize_map(Keys)
        }
    }
    impl<'de> Visitor<'de> for Keys {
        type Value = Vec<String>;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("a JSON object")
        }
        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Vec<String>, A::Error> {
            let mut keys: Vec<String> = Vec::new();
            while let Some(key) = map.next_key::<String>()? {
                map.next_value::<IgnoredAny>()?;
                if !keys.contains(&key) {
                    keys.push(key);
                }
            }
            Ok(keys)
        }
    }
    let mut de = serde_json::Deserializer::from_slice(json);
    de.deserialize_map(Top)
        .map_err(|e| CoreError::Json(format!("the {FORMAT} file ({e})")))
}
