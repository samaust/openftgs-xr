// SPDX-License-Identifier: Apache-2.0

//! The `ftgs-render-request` reader (§1.7, openftgs §2.2, openftgs `io.read_render_request`).

use std::collections::HashSet;

use serde_json::Value;
use sha2::{Digest, Sha256};

use super::json::{self, invalid, required, shape};
use crate::types::{Camera, CoreError};

/// The request's `format`.
const FORMAT: &str = "ftgs-render-request";

/// One item of a render request: render `camera` at `timestamp`.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderItem {
    /// The item's id, the suffix of its `rgb/<id>` and `alpha/<id>` renders.
    pub id: String,
    /// τ in the request's time unit.
    pub timestamp: f64,
    /// The camera, in openftgs's conventions (E6).
    pub camera: Camera,
}

/// A `ftgs-render-request` v1 file (openftgs §2.2).
#[derive(Clone, Debug, PartialEq)]
pub struct RenderRequest {
    /// The unit of the timestamps, which `render` compares with the model's (E-REQ-UNIT).
    pub time_unit: String,
    /// The background RGB in \[0, 1\]; black when the file gives none.
    pub background: [f32; 3],
    /// The SH degree to render at, 0–3; `None` (absent or null) is the model's full degree.
    pub sh_degree: Option<u8>,
    /// The items, in file order.
    pub items: Vec<RenderItem>,
    /// The SHA-256 of the file's bytes, which `ftgs-renders` records as `request_sha256`.
    pub sha256: [u8; 32],
}

/// Reads and checks a `ftgs-render-request` file with openftgs's checks (§1.7): the
/// `format`, a version of major 1, a non-empty `time_unit`, the background (black by
/// default, three numbers in \[0, 1\]), `sh_degree` (null by default, else 0–3), a
/// non-empty item list with non-empty unique ids, finite timestamps and cameras that pass
/// the camera rules (an orthonormal rotation, an unskewed K). Unknown keys are ignored.
/// Each refusal has the §1.15 code of Q-25.
// Q-25
pub fn read_request(json: &[u8]) -> Result<RenderRequest, CoreError> {
    let map = json::parse_object(json, FORMAT)?;
    json::check_header(&map, FORMAT)?;
    let time_unit = json::time_unit(&map)?;
    let background = match map.get("background") {
        None => [0.0; 3],
        Some(v) => {
            let list = v
                .as_array()
                .filter(|a| a.len() == 3)
                .ok_or_else(|| shape("background", v, "a list of three numbers"))?;
            let mut rgb = [0.0f32; 3];
            for (c, x) in rgb.iter_mut().zip(list) {
                let x = json::number("background", x)?;
                if !(0.0..=1.0).contains(&x) {
                    return Err(invalid("background", v));
                }
                *c = x as f32;
            }
            rgb
        }
    };
    let sh_degree = match map.get("sh_degree") {
        None | Some(Value::Null) => None,
        Some(Value::Number(n)) if n.is_i64() || n.is_u64() => Some(
            n.as_u64()
                .filter(|&d| d <= 3)
                .map(|d| d as u8)
                .ok_or_else(|| CoreError::Sh(format!("request sh_degree {n}")))?,
        ),
        Some(other) => return Err(shape("sh_degree", other, "null or an integer")),
    };
    let list = required(&map, "items")?;
    let list = list
        .as_array()
        .ok_or_else(|| shape("items", list, "a list"))?;
    if list.is_empty() {
        return Err(invalid("items", "[] (no item)"));
    }
    let items = list.iter().map(item).collect::<Result<Vec<_>, _>>()?;
    let mut seen = HashSet::new();
    if let Some(dup) = items.iter().find(|i| !seen.insert(i.id.as_str())) {
        return Err(invalid("items", format!("the id {:?} repeated", dup.id)));
    }
    Ok(RenderRequest {
        time_unit,
        background,
        sh_degree,
        items,
        sha256: Sha256::digest(json).into(),
    })
}

/// One item: a non-empty string id, a finite timestamp and a camera.
fn item(value: &Value) -> Result<RenderItem, CoreError> {
    let map = value
        .as_object()
        .ok_or_else(|| shape("item", value, "an object"))?;
    let id = match map.get("id") {
        Some(Value::String(s)) if !s.is_empty() => s.clone(),
        Some(Value::String(_)) => return Err(invalid("item id", "\"\"")),
        Some(other) => return Err(shape("item id", other, "a string")),
        None => return Err(CoreError::Keys("item id".to_owned())),
    };
    let at = |k: &str| format!("item {id:?} {k}");
    let timestamp = json::number(
        &at("timestamp"),
        map.get("timestamp")
            .ok_or_else(|| CoreError::Keys(at("timestamp")))?,
    )?;
    let camera = json::camera(
        &at("camera"),
        map.get("camera")
            .ok_or_else(|| CoreError::Keys(at("camera")))?,
    )?;
    Ok(RenderItem {
        id,
        timestamp,
        camera,
    })
}
