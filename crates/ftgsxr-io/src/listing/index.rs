// SPDX-License-Identifier: Apache-2.0

//! The player's own listing, `index.json` (`ftgsxr-index` v1, §1.11).

use ftgsxr_core::CoreError;
use serde_json::{Map, Value};
use url::Url;

use super::{Entry, EntryKind};
use crate::IoError;
use crate::util::version_major;

/// The manifest's `format`.
const FORMAT: &str = "ftgsxr-index";

/// An E-SCENE-FORMAT error naming the index fetched from `base`.
fn schema(base: &Url, detail: impl std::fmt::Display) -> IoError {
    IoError::Scene(format!("index at {base}: {detail}"))
}

/// The string entry `key` of `map`, `None` when absent or null.
fn optional_string<'a>(
    map: &'a Map<String, Value>,
    key: &str,
    base: &Url,
) -> Result<Option<&'a str>, IoError> {
    match map.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s)),
        Some(other) => Err(schema(base, format!("{key} is {other}, expected a string"))),
    }
}

/// `path` resolved against `base`.
fn resolve(base: &Url, path: &str) -> Result<Url, IoError> {
    base.join(path)
        .map_err(|e| schema(base, format!("path {path:?} does not resolve ({e})")))
}

/// `url` with a `/` ending its path, its query and fragment kept: a checkpoint is a
/// directory, so its two files resolve inside it (Q-36); an HTTP index's base too (Q-48).
// Q-36
pub(crate) fn as_directory(mut url: Url) -> Url {
    if !url.path().ends_with('/') {
        let path = format!("{}/", url.path());
        url.set_path(&path);
    }
    url
}

/// One entry of `entries`.
fn entry(value: &Value, base: &Url) -> Result<Entry, IoError> {
    let Value::Object(map) = value else {
        return Err(schema(base, format!("entry {value} is not an object")));
    };
    let required = |key: &str| {
        optional_string(map, key, base)?.ok_or_else(|| schema(base, format!("missing key {key}")))
    };
    let name = required("name")?;
    let path = required("path")?;
    let kind = match required("kind")? {
        "ftgs-if" => EntryKind::Exchange,
        "ftgs-checkpoint" => EntryKind::Checkpoint,
        "ply" => EntryKind::Ply,
        other => return Err(schema(base, format!("unknown kind {other:?}"))),
    };
    let size = match map.get("size") {
        Some(Value::Number(n)) if n.as_u64().is_some() => n.as_u64(),
        Some(other) => {
            return Err(schema(
                base,
                format!("size is {other}, expected a byte count"),
            ));
        }
        None => return Err(schema(base, "missing key size")),
    };
    let sha256 = match optional_string(map, "sha256", base)? {
        Some(d) if d.len() == 64 && d.bytes().all(|b| b.is_ascii_hexdigit()) => {
            Some(d.to_ascii_lowercase())
        }
        Some(d) => return Err(schema(base, format!("sha256 {d:?} is not 64 hex digits"))),
        None => None,
    };
    let linked = |key: &str| -> Result<Option<String>, IoError> {
        optional_string(map, key, base)?
            .map(|p| resolve(base, p).map(String::from))
            .transpose()
    };
    let location = resolve(base, path)?;
    let location = match kind {
        EntryKind::Checkpoint => as_directory(location),
        _ => location,
    };
    Ok(Entry {
        name: name.to_owned(),
        location: location.into(),
        kind,
        size,
        mtime: None,
        sha256,
        thumbnail: linked("thumbnail")?,
        audio: linked("audio")?,
        scene: linked("scene")?,
    })
}

/// Parses an `ftgsxr-index` v1 manifest (§1.11) fetched from `base` (Q-36, Q-37): every
/// path resolves against `base`, and a checkpoint's location ends in `/`.
// Q-36, Q-37
pub(super) fn parse(body: &str, base: &Url) -> Result<Vec<Entry>, IoError> {
    let value: Value = serde_json::from_str(body)
        .map_err(|e| CoreError::Json(format!("index at {base} is not JSON ({e})")))?;
    let Value::Object(map) = value else {
        return Err(schema(base, "not a JSON object"));
    };
    match map.get("format") {
        Some(Value::String(f)) if f == FORMAT => {}
        Some(other) => return Err(schema(base, format!("format {other}, expected {FORMAT:?}"))),
        None => return Err(schema(base, "missing key format")),
    }
    match map.get("version") {
        Some(v) if version_major(v) == Some(1) => {}
        Some(v) => return Err(schema(base, format!("version {v} is not supported"))),
        None => return Err(schema(base, "missing key version")),
    }
    match map.get("entries") {
        Some(Value::Array(entries)) => entries.iter().map(|e| entry(e, base)).collect(),
        Some(other) => Err(schema(
            base,
            format!("entries is {other}, expected an array"),
        )),
        None => Err(schema(base, "missing key entries")),
    }
}
