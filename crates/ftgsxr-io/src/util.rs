// SPDX-License-Identifier: Apache-2.0

//! What the player's own files share (§1.11): the atomic write, the clock their times
//! come from and the versioning rule's `version` check, and the cancel error every source
//! gives. Placed before every module of §2.1, which may all use it.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

use crate::IoError;

/// The error of a cancelled read or download of `what`: E-IO of kind `Interrupted`, one
/// for every source, recognised by its kind (Q-43, Q-47).
pub(crate) fn cancelled(what: impl std::fmt::Display) -> IoError {
    IoError::Io(std::io::Error::new(
        std::io::ErrorKind::Interrupted,
        format!("{what}: cancelled"),
    ))
}

/// Writes `bytes` to `<path>.tmp`, flushes it to the disk and renames it over `path`, so a
/// crash leaves the old file intact (§1.11 Locations).
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), IoError> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    let mut file = File::create(&tmp)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Whether `version` names major version 1 (§1.11's versioning rule, which ignores a
/// minor version): a JSON integer, or a `"major[.minor]"` string of decimal parts without
/// leading zeros.
pub(crate) fn version_is_1(version: &Value) -> bool {
    fn decimal(part: &str) -> Option<u64> {
        let digits = !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit());
        let leading_zero = part.len() > 1 && part.starts_with('0');
        (digits && !leading_zero)
            .then(|| part.parse().ok())
            .flatten()
    }
    match version {
        Value::Number(n) => n.as_u64() == Some(1),
        Value::String(text) => {
            let mut parts = text.splitn(2, '.');
            let major = parts.next().and_then(decimal);
            let minor_ok = parts.next().is_none_or(|m| decimal(m).is_some());
            major == Some(1) && minor_ok
        }
        _ => false,
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
    let (year, month, day) = civil_date(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{nanos:09}Z",
        rem / 3600,
        rem / 60 % 60,
        rem % 60
    )
}

/// Whether `year` is a leap year of the Gregorian calendar: divisible by 4, except
/// centuries not divisible by 400.
fn is_leap(year: u64) -> bool {
    year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400))
}

/// The (year, month, day) of the Gregorian calendar of the day `days` after 1970-01-01:
/// whole years are counted off from 1970, then whole months of the year reached, and the
/// days left over give the day of the month.
fn civil_date(mut days: u64) -> (u64, u64, u64) {
    let mut year = 1970;
    loop {
        let length = if is_leap(year) { 366 } else { 365 };
        if days < length {
            break;
        }
        days -= length;
        year += 1;
    }
    let february = if is_leap(year) { 29 } else { 28 };
    let lengths = [31, february, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut month = 1;
    for length in lengths {
        if days < length {
            break;
        }
        days -= length;
        month += 1;
    }
    (year, month, days + 1)
}

/// Whether `s` has the fixed-width form `format_rfc3339` writes.
pub(crate) fn is_rfc3339(s: &str) -> bool {
    let b = s.as_bytes();
    let digits =
        |r: std::ops::Range<usize>| b.get(r).is_some_and(|d| d.iter().all(u8::is_ascii_digit));
    b.len() == 30
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
        && b[29] == b'Z'
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
        assert!(is_rfc3339(&now_rfc3339()));
        assert!(!is_rfc3339("2026-10-29T12:00:00Z"));
    }

    /// Hand-worked: day 365 is 1971-01-01 (1970 has 365 days); day 365 + 365 + 59 is
    /// 1972-02-29 (1972 is a leap year, 31 + 28 days before 29 February); day 10 957 is
    /// 2000-01-01 (thirty years, seven of them leap: 30 · 365 + 7).
    #[test]
    fn civil_date_crosses_years_and_leap_days() {
        assert_eq!(civil_date(364), (1970, 12, 31));
        assert_eq!(civil_date(365), (1971, 1, 1));
        assert_eq!(civil_date(365 + 365 + 59), (1972, 2, 29));
        assert_eq!(civil_date(365 + 365 + 60), (1972, 3, 1));
        assert_eq!(civil_date(10_957), (2000, 1, 1));
    }

    #[test]
    fn is_leap_follows_the_gregorian_rule() {
        assert!(is_leap(1972) && is_leap(2000) && is_leap(2400));
        assert!(!is_leap(1971) && !is_leap(1900) && !is_leap(2100));
    }

    #[test]
    fn write_atomic_replaces_the_file_and_leaves_no_temporary() {
        let dir =
            std::env::temp_dir().join(format!("ftgsxr-io-write-atomic-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.json");
        std::fs::write(&path, b"old").unwrap();
        write_atomic(&path, b"new").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"new");
        assert!(!dir.join("a.json.tmp").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn version_is_1_ignores_a_minor_version() {
        for ok in [Value::from(1), "1".into(), "1.0".into(), "1.12".into()] {
            assert!(version_is_1(&ok), "{ok}");
        }
        for bad in [
            Value::from(2),
            Value::from(1.5),
            "2".into(),
            "2.0".into(),
            "01".into(),
            "1.".into(),
            "1.2.3".into(),
            Value::Null,
        ] {
            assert!(!version_is_1(&bad), "{bad}");
        }
    }
}
