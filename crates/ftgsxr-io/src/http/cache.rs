// SPDX-License-Identifier: Apache-2.0

//! The download cache's entries on disk (§1.5, §1.11 Cache entries, Q-42): their paths,
//! their records and the times that order them.

use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use url::Url;

use crate::IoError;
use crate::util::{is_rfc3339, write_atomic};

/// The `format` of an entry's record (Q-42).
const FORMAT: &str = "ftgsxr-cache-entry";
/// The record's major version (Q-42).
const VERSION: u64 = 1;

/// The files of one URL's cache entry, `<cache dir>/http/<sha256 of the URL>/` (§1.11).
pub(crate) struct Entry {
    /// The entry's directory.
    pub(crate) dir: PathBuf,
    /// The complete file, `file` (§1.11).
    pub(crate) file: PathBuf,
    /// The complete file's record, `meta.json` (§1.11).
    pub(crate) meta: PathBuf,
    /// The download in progress, the temporary sibling `file.part` (§1.11, Q-42).
    pub(crate) part: PathBuf,
    /// The download in progress's record, `meta.part.json` (Q-42).
    pub(crate) part_meta: PathBuf,
}

impl Entry {
    /// The entry of `url` under the cache's `http/` directory; the key is the SHA-256 of the
    /// URL's serialization by the url crate (Q-42).
    pub(crate) fn of(entries: &Path, url: &Url) -> Entry {
        let key = ftgsxr_core::hex(&ftgsxr_core::sha256(url.as_str().as_bytes()));
        Entry::at(entries.join(key))
    }

    /// The entry whose directory is `dir`.
    pub(crate) fn at(dir: PathBuf) -> Entry {
        Entry {
            file: dir.join("file"),
            meta: dir.join("meta.json"),
            part: dir.join("file.part"),
            part_meta: dir.join("meta.part.json"),
            dir,
        }
    }

    /// The complete file's record, when it is readable, belongs to `url` (any URL when
    /// `None`), carries a digest and agrees with the file's size (Q-42).
    pub(crate) fn complete(&self, url: Option<&Url>) -> Option<Record> {
        let record = Record::read(&self.meta, url)?;
        let size = std::fs::metadata(&self.file).ok()?.len();
        (record.sha256.is_some() && record.size == Some(size)).then_some(record)
    }

    /// The download in progress: its record and the bytes `file.part` holds, when both
    /// exist and the record belongs to `url` (any URL when `None`).
    pub(crate) fn partial(&self, url: Option<&Url>) -> Option<(Record, u64)> {
        let record = Record::read(&self.part_meta, url)?;
        let len = std::fs::metadata(&self.part).ok()?.len();
        Some((record, len))
    }

    /// Removes the download in progress, if any.
    pub(crate) fn discard_partial(&self) -> Result<(), IoError> {
        remove_if_present(&self.part)?;
        remove_if_present(&self.part_meta)
    }
}

/// An entry's record, `meta.json` or `meta.part.json` (§1.11, Q-42).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Record {
    pub(crate) url: String,
    pub(crate) etag: Option<String>,
    pub(crate) last_modified: Option<String>,
    /// The file's size: the server's when the download starts, the bytes held once complete.
    pub(crate) size: Option<u64>,
    /// 64 lowercase hex digits once the file is complete.
    pub(crate) sha256: Option<String>,
    /// RFC 3339 UTC with nanoseconds, fixed width, so the strings order as the times do.
    pub(crate) fetched_at: String,
}

impl Record {
    /// The record at `path`, when it parses, is of this format and major version, has a
    /// well-formed `fetched_at` and belongs to `url` (any URL when `None`) (Q-42).
    pub(crate) fn read(path: &Path, url: Option<&Url>) -> Option<Record> {
        let value: Value = serde_json::from_slice(&std::fs::read(path).ok()?).ok()?;
        if value.get("format")?.as_str()? != FORMAT || value.get("version")?.as_u64()? != VERSION {
            return None;
        }
        let text = |key: &str| match value.get(key) {
            Some(Value::String(s)) => Some(Some(s.clone())),
            Some(Value::Null) | None => Some(None),
            Some(_) => None,
        };
        let size = match value.get("size") {
            Some(Value::Null) | None => None,
            Some(v) => Some(v.as_u64()?),
        };
        let record = Record {
            url: value.get("url")?.as_str()?.to_owned(),
            etag: text("etag")?,
            last_modified: text("last_modified")?,
            size,
            sha256: text("sha256")?,
            fetched_at: value.get("fetched_at")?.as_str()?.to_owned(),
        };
        let digest_ok = record.sha256.as_deref().is_none_or(is_digest);
        let time_ok = is_rfc3339(&record.fetched_at);
        let url_ok = url.is_none_or(|url| record.url == url.as_str());
        (digest_ok && time_ok && url_ok).then_some(record)
    }

    /// Writes the record to `path` through a temporary sibling (§1.11).
    pub(crate) fn write(&self, path: &Path) -> Result<(), IoError> {
        let value = json!({
            "format": FORMAT,
            "version": VERSION,
            "url": self.url,
            "etag": self.etag,
            "last_modified": self.last_modified,
            "size": self.size,
            "sha256": self.sha256,
            "fetched_at": self.fetched_at,
        });
        let bytes = serde_json::to_vec_pretty(&value).map_err(std::io::Error::other)?;
        write_atomic(path, &bytes)
    }
}

/// Whether `s` is 64 lowercase hex digits.
fn is_digest(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// Removes the file at `path`; a missing file is not an error.
pub(crate) fn remove_if_present(path: &Path) -> Result<(), IoError> {
    match std::fs::remove_file(path) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e.into()),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::now_rfc3339;

    #[test]
    fn record_round_trips_and_refuses_other_formats() {
        let dir = std::env::temp_dir().join("ftgsxr-http-record");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("meta.json");
        let record = Record {
            url: "http://example.org/a.safetensors".into(),
            etag: Some("\"x\"".into()),
            last_modified: None,
            size: Some(12),
            sha256: Some("ab".repeat(32)),
            fetched_at: now_rfc3339(),
        };
        record.write(&path).unwrap();
        assert_eq!(Record::read(&path, None), Some(record.clone()));
        let other = Url::parse("http://example.org/b.safetensors").unwrap();
        assert_eq!(Record::read(&path, Some(&other)), None);
        std::fs::write(&path, br#"{"format": "ftgsxr-cache-entry", "version": 2}"#).unwrap();
        assert_eq!(Record::read(&path, None), None);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
