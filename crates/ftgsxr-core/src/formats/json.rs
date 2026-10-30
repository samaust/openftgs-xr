// SPDX-License-Identifier: Apache-2.0

//! The checks the JSON formats share: the `format` and `version` header, the time unit and
//! the camera (openftgs §2.2, openftgs `io._check_json_header`, `io._parse_camera`).
//!
//! Every refusal carries a §1.15 code (Q-25): a missing key is E-FILE-KEYS, a value of
//! the wrong JSON type or matrix size E-FILE-SHAPE, a value of the right type that breaks a
//! rule E-FILE-VALUE.

use glam::{DMat3, DMat4, DVec3, DVec4};
use serde_json::{Map, Value};

use crate::loader::check_version;
use crate::types::{Camera, CoreError};

/// The largest |RᵀR − I| entry a camera rotation may have (openftgs `io`, its Q-56).
const ORTHONORMAL_TOLERANCE: f64 = 1e-5;

/// Parses `bytes` as one JSON object: text that is not JSON is E-JSON, a JSON value that is
/// not an object is not a file of the format `what` (E-FILE-FORMAT). Q-25
pub(crate) fn parse_object(bytes: &[u8], what: &str) -> Result<Map<String, Value>, CoreError> {
    let value: Value = serde_json::from_slice(bytes)
        .map_err(|e| CoreError::Json(format!("the {what} file ({e})")))?;
    object(value, what)
}

/// `value` as a JSON object, else not a file of the format `what` (E-FILE-FORMAT). Q-25
pub(crate) fn object(value: Value, what: &str) -> Result<Map<String, Value>, CoreError> {
    match value {
        Value::Object(map) => Ok(map),
        _ => Err(CoreError::Format(format!(
            "a JSON {} where a {what} object was expected",
            kind(&value)
        ))),
    }
}

/// The name of a JSON value's type, for messages.
pub(crate) fn kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// The entry `key`, or E-FILE-KEYS.
pub(crate) fn required<'a>(map: &'a Map<String, Value>, key: &str) -> Result<&'a Value, CoreError> {
    map.get(key).ok_or_else(|| CoreError::Keys(key.to_owned()))
}

/// E-FILE-SHAPE: `what` holds a value of the wrong JSON type or size.
pub(crate) fn shape(what: &str, value: &Value, expected: &str) -> CoreError {
    CoreError::Shape(format!(
        "{what} is a JSON {}, expected {expected}",
        kind(value)
    ))
}

/// E-FILE-VALUE naming `what` and its value.
pub(crate) fn invalid(what: &str, value: impl std::fmt::Display) -> CoreError {
    CoreError::Value(format!("{what} = {value}"))
}

/// Checks the header of a JSON format (openftgs `io._check_json_header`): `format` must
/// be `expected` (E-FILE-FORMAT) and `version` a JSON integer or a decimal `major[.minor]`
/// string of major 1 (E-FILE-VERSION; absent, E-FILE-KEYS). Q-25
pub(crate) fn check_header(map: &Map<String, Value>, expected: &str) -> Result<(), CoreError> {
    match map.get("format") {
        Some(Value::String(s)) if s == expected => {}
        Some(other) => {
            return Err(CoreError::Format(format!(
                "format {other}, expected {expected:?}"
            )));
        }
        None => return Err(CoreError::Keys("format".to_owned())),
    }
    let version = required(map, "version")?;
    let text = match version {
        Value::Number(n) if n.is_i64() || n.is_u64() => n.to_string(),
        Value::String(s) => s.clone(),
        other => return Err(CoreError::Version(other.to_string())),
    };
    check_version(&text)
}

/// `time_unit`: a non-empty string (openftgs `io._time_unit`).
pub(crate) fn time_unit(map: &Map<String, Value>) -> Result<String, CoreError> {
    match required(map, "time_unit")? {
        Value::String(s) if !s.is_empty() => Ok(s.clone()),
        Value::String(_) => Err(invalid("time_unit", "\"\"")),
        other => Err(shape("time_unit", other, "a string")),
    }
}

/// A JSON number, checked finite (serde_json parses no NaN or infinity, but an integer
/// may lie beyond f64's range).
pub(crate) fn number(what: &str, value: &Value) -> Result<f64, CoreError> {
    match value {
        Value::Number(n) => n
            .as_f64()
            .filter(|x| x.is_finite())
            .ok_or_else(|| invalid(what, n)),
        other => Err(shape(what, other, "a number")),
    }
}

/// A JSON integer ≥ 1 that fits a u32 (openftgs `io._positive_int`).
fn positive_u32(what: &str, value: &Value) -> Result<u32, CoreError> {
    match value {
        Value::Number(n) if n.is_i64() || n.is_u64() => n
            .as_u64()
            .and_then(|x| u32::try_from(x).ok())
            .filter(|&x| x >= 1)
            .ok_or_else(|| invalid(what, n)),
        other => Err(shape(what, other, "an integer")),
    }
}

/// A `rows` × `cols` matrix given as a list of rows of finite numbers.
fn matrix(what: &str, value: &Value, rows: usize, cols: usize) -> Result<Vec<Vec<f64>>, CoreError> {
    let expected = format!("a {rows}×{cols} list of rows");
    let list = value
        .as_array()
        .filter(|a| a.len() == rows)
        .ok_or_else(|| shape(what, value, &expected))?;
    list.iter()
        .map(|row| {
            let row = row
                .as_array()
                .filter(|r| r.len() == cols)
                .ok_or_else(|| shape(what, value, &expected))?;
            row.iter().map(|x| number(what, x)).collect()
        })
        .collect()
}

/// A camera as the JSON formats give it (openftgs §2.2, `io._parse_camera`): integer width
/// and height ≥ 1; K = \[\[f_x, 0, c_x\], \[0, f_y, c_y\], \[0, 0, 1\]\] with f_x, f_y > 0
/// and no skew; a world-to-camera matrix with the last row (0, 0, 0, 1) whose rotation has
/// max |RᵀR − I| ≤ 10⁻⁵ and a positive determinant, both computed in float64. The values
/// are kept as parsed, in float64; unknown keys are ignored.
pub(crate) fn camera(what: &str, value: &Value) -> Result<Camera, CoreError> {
    let map = value
        .as_object()
        .ok_or_else(|| shape(what, value, "an object"))?;
    let key = |k: &str| format!("{what}.{k}");
    let get = |k: &str| map.get(k).ok_or_else(|| CoreError::Keys(key(k)));
    let width = positive_u32(&key("width"), get("width")?)?;
    let height = positive_u32(&key("height"), get("height")?)?;
    let k = matrix(&key("K"), get("K")?, 3, 3)?;
    if !(k[0][0] > 0.0 && k[1][1] > 0.0) {
        return Err(invalid(
            &key("K"),
            format!("f_x {} and f_y {}", k[0][0], k[1][1]),
        ));
    }
    if k[0][1] != 0.0 || k[1][0] != 0.0 || k[2] != [0.0, 0.0, 1.0] {
        return Err(invalid(
            &key("K"),
            format!("{k:?} (skewed, or a last row other than (0, 0, 1))"),
        ));
    }
    let w = matrix(&key("world_to_camera"), get("world_to_camera")?, 4, 4)?;
    if w[3] != [0.0, 0.0, 0.0, 1.0] {
        return Err(invalid(
            &key("world_to_camera"),
            format!("last row {:?}", w[3]),
        ));
    }
    let r = DMat3::from_cols(
        DVec3::new(w[0][0], w[1][0], w[2][0]),
        DVec3::new(w[0][1], w[1][1], w[2][1]),
        DVec3::new(w[0][2], w[1][2], w[2][2]),
    );
    let gram = r.transpose() * r - DMat3::IDENTITY;
    let error = gram
        .to_cols_array()
        .iter()
        .fold(0.0f64, |m, x| m.max(x.abs()));
    if !(error <= ORTHONORMAL_TOLERANCE && r.determinant() > 0.0) {
        return Err(invalid(
            &key("world_to_camera"),
            format!(
                "a rotation with |RᵀR − I| = {error:e} and determinant {}",
                r.determinant()
            ),
        ));
    }
    Ok(Camera {
        width,
        height,
        k: DMat3::from_cols(
            DVec3::new(k[0][0], k[1][0], k[2][0]),
            DVec3::new(k[0][1], k[1][1], k[2][1]),
            DVec3::new(k[0][2], k[1][2], k[2][2]),
        ),
        world_to_camera: DMat4::from_cols(
            DVec4::new(w[0][0], w[1][0], w[2][0], w[3][0]),
            DVec4::new(w[0][1], w[1][1], w[2][1], w[3][1]),
            DVec4::new(w[0][2], w[1][2], w[2][2], w[3][2]),
            DVec4::new(w[0][3], w[1][3], w[2][3], w[3][3]),
        ),
    })
}
