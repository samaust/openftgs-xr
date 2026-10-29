// SPDX-License-Identifier: Apache-2.0

//! The download cache's entries on disk (§1.5, §1.11 Cache entries, Q-42): their paths,
//! their records and the times that order them.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};
use url::Url;

use crate::IoError;

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
        let time_ok = parse_rfc3339(&record.fetched_at).is_some();
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

/// Writes `bytes` to `<path>.tmp` and renames it over `path` (§1.11).
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), IoError> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Removes the file at `path`; a missing file is not an error.
pub(crate) fn remove_if_present(path: &Path) -> Result<(), IoError> {
    match std::fs::remove_file(path) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e.into()),
        _ => Ok(()),
    }
}

/// The current time as RFC 3339 UTC with nanoseconds (Q-42).
pub(crate) fn now_rfc3339() -> String {
    let since = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format_rfc3339(since.as_secs(), since.subsec_nanos())
}

/// `YYYY-MM-DDTHH:MM:SS.nnnnnnnnnZ` for a time `secs` + `nanos` after the Unix epoch.
fn format_rfc3339(secs: u64, nanos: u32) -> String {
    let (days, rem) = (secs / 86_400, secs % 86_400);
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{nanos:09}Z",
        rem / 3600,
        rem / 60 % 60,
        rem % 60
    )
}

/// The proleptic Gregorian (year, month, day) of the day `days` after 1970-01-01, by the
/// era arithmetic over 400-year cycles that start on 1 March of Howard Hinnant's published
/// `civil_from_days` algorithm, written from the arithmetic itself (no page or code
/// consulted; source log 48).
fn civil_from_days(days: u64) -> (u64, u64, u64) {
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z % 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + u64::from(month <= 2);
    (year, month, day)
}

/// Checks the fixed-width form `format_rfc3339` writes; `Some(())` when it holds.
fn parse_rfc3339(s: &str) -> Option<()> {
    let b = s.as_bytes();
    let digits =
        |r: std::ops::Range<usize>| b.get(r).is_some_and(|d| d.iter().all(u8::is_ascii_digit));
    let shape = b.len() == 30
        && digits(0..4)
        && b[4] == b'-'
        && digits(5..7)
        && b[7] == b'-'
        && digits(8..10)
        && b[10] == b'T'
        && digits(11..13)
        && b[13] == b':'
        && digits(14..16)
        && b[16] == b':'
        && digits(17..19)
        && b[19] == b'.'
        && digits(20..29)
        && b[29] == b'Z';
    shape.then_some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Expected values from Python's `datetime.fromtimestamp(t, timezone.utc).isoformat()`.
    #[test]
    fn format_rfc3339_matches_known_dates() {
        assert_eq!(format_rfc3339(0, 0), "1970-01-01T00:00:00.000000000Z");
        assert_eq!(
            format_rfc3339(951_782_400, 5),
            "2000-02-29T00:00:00.000000005Z"
        );
        assert_eq!(
            format_rfc3339(1_793_275_200, 123_456_789),
            "2026-10-29T12:00:00.123456789Z"
        );
        assert_eq!(
            format_rfc3339(4_107_542_399, 0),
            "2100-02-28T23:59:59.000000000Z"
        );
        assert!(parse_rfc3339(&now_rfc3339()).is_some());
        assert!(parse_rfc3339("2026-10-29T12:00:00Z").is_none());
    }

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
