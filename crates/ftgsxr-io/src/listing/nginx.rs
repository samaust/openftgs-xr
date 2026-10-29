// SPDX-License-Identifier: Apache-2.0

//! nginx's `autoindex_format json` listing (§1.5, D-31; spec Appendix A row 35): an array
//! of `{name, type, mtime, size}`, `type` one of `directory`, `file`, `other`.

use serde_json::Value;
use url::Url;

use super::{Entry, EntryKind, classify_name, encode_segment, is_hidden, stem};
use crate::IoError;

/// Parses nginx's JSON listing of `base` (Q-36, Q-37).
// Q-36, Q-37
pub(super) fn parse(body: &str, base: &Url) -> Result<Vec<Entry>, IoError> {
    let not_nginx = || IoError::Listing(base.to_string());
    let Ok(Value::Array(items)) = serde_json::from_str::<Value>(body) else {
        return Err(not_nginx());
    };
    let mut entries = Vec::new();
    for item in &items {
        let Value::Object(map) = item else {
            return Err(not_nginx());
        };
        let (Some(Value::String(name)), Some(Value::String(kind))) =
            (map.get("name"), map.get("type"))
        else {
            return Err(not_nginx());
        };
        if is_hidden(name) {
            continue;
        }
        let (kind, suffix) = match kind.as_str() {
            "directory" => (EntryKind::Dir, "/"),
            "file" => match classify_name(name) {
                k @ (EntryKind::Exchange | EntryKind::Ply) => (k, ""),
                _ => continue,
            },
            _ => continue,
        };
        let location = base
            .join(&format!("{}{suffix}", encode_segment(name)))
            .map_err(|_| not_nginx())?;
        entries.push(Entry {
            name: stem(name, kind).to_owned(),
            location: location.into(),
            kind,
            size: match kind {
                EntryKind::Dir => None,
                _ => map.get("size").and_then(Value::as_u64),
            },
            mtime: map.get("mtime").and_then(Value::as_str).map(str::to_owned),
            sha256: None,
            thumbnail: None,
            audio: None,
            scene: None,
        });
    }
    Ok(entries)
}
