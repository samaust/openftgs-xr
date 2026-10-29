// SPDX-License-Identifier: Apache-2.0

//! The fixture server's header rules (Q-T15.1): percent-encoding, validators and byte
//! ranges, as pure functions of header values.

/// The `Last-Modified` date of every served file: the committed listing bodies' mtime.
pub(crate) const LAST_MODIFIED: &str = "Thu, 29 Oct 2026 12:00:00 GMT";

/// `path` with every byte outside RFC 3986's unreserved set and `/` percent-encoded.
pub(crate) fn percent_encode(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    for &b in path.as_bytes() {
        if b.is_ascii_alphanumeric() || b"-._~/".contains(&b) {
            out.push(char::from(b));
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// `path` with its `%XX` escapes decoded; `None` for a malformed escape or a result that
/// is not UTF-8.
pub(crate) fn percent_decode(path: &str) -> Option<String> {
    let bytes = path.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = bytes.get(i + 1..i + 3)?;
            let text = std::str::from_utf8(hex).ok()?;
            out.push(u8::from_str_radix(text, 16).ok()?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

/// Whether an `If-None-Match` value matches `etag`: `*`, or any listed tag equal to it
/// under weak comparison (a `W/` prefix ignored).
pub(crate) fn none_match_hits(header: &str, etag: &str) -> bool {
    header
        .split(',')
        .map(str::trim)
        .any(|tag| tag == "*" || tag.strip_prefix("W/").unwrap_or(tag) == etag)
}

/// Whether an `If-Range` value still names this representation: the same strong ETag, or
/// the same date as `Last-Modified`.
pub(crate) fn if_range_holds(header: &str, etag: &str) -> bool {
    let value = header.trim();
    value == etag || value == LAST_MODIFIED
}

/// How a request's `Range` header is answered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RangeAnswer {
    /// No usable single byte range: the whole representation, 200.
    Whole,
    /// Bytes `start..=end`, 206.
    Part { start: u64, end: u64 },
    /// No byte of the range exists, 416.
    Unsatisfiable,
}

/// The answer to `Range: <header>` for a representation of `len` bytes: a single
/// `bytes=a-b`, `bytes=a-` or `bytes=-n`; several ranges and malformed values are ignored.
pub(crate) fn range_answer(header: &str, len: u64) -> RangeAnswer {
    let header = header.trim();
    let Some(spec) = header
        .get(..6)
        .filter(|unit| unit.eq_ignore_ascii_case("bytes="))
        .map(|_| header[6..].trim())
    else {
        return RangeAnswer::Whole;
    };
    if spec.contains(',') {
        return RangeAnswer::Whole;
    }
    let Some((first, last)) = spec.split_once('-') else {
        return RangeAnswer::Whole;
    };
    let number = |text: &str| -> Option<u64> {
        (!text.is_empty() && text.bytes().all(|b| b.is_ascii_digit()))
            .then(|| text.parse().ok())
            .flatten()
    };
    if first.is_empty() {
        return match number(last) {
            None => RangeAnswer::Whole,
            Some(0) => RangeAnswer::Unsatisfiable,
            Some(_) if len == 0 => RangeAnswer::Unsatisfiable,
            Some(n) => RangeAnswer::Part {
                start: len.saturating_sub(n),
                end: len - 1,
            },
        };
    }
    let Some(start) = number(first) else {
        return RangeAnswer::Whole;
    };
    let end = if last.is_empty() {
        u64::MAX
    } else {
        match number(last) {
            Some(end) if end >= start => end,
            _ => return RangeAnswer::Whole,
        }
    };
    if start >= len {
        RangeAnswer::Unsatisfiable
    } else {
        RangeAnswer::Part {
            start,
            end: end.min(len - 1),
        }
    }
}

/// The `Content-Type` of a served file, by its extension.
pub(crate) fn content_type(name: &str) -> &'static str {
    let extension = name.rsplit_once('.').map_or("", |(_, e)| e);
    match extension {
        "json" => "application/json",
        "html" => "text/html; charset=utf-8",
        "txt" => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use RangeAnswer::*;

    #[test]
    fn percent_encode_keeps_slashes_and_unreserved_bytes() {
        assert_eq!(
            percent_encode("listing/models/probe p1.safetensors"),
            "listing/models/probe%20p1.safetensors"
        );
        assert_eq!(percent_encode("a~b_c-d.e/%é"), "a~b_c-d.e/%25%C3%A9");
    }

    #[test]
    fn percent_decode_inverts_encode_and_refuses_bad_escapes() {
        for path in ["probe p1.safetensors", "a/é/%x", ".hidden"] {
            assert_eq!(percent_decode(&percent_encode(path)).as_deref(), Some(path));
        }
        assert_eq!(percent_decode("a%2"), None);
        assert_eq!(percent_decode("a%zz"), None);
        assert_eq!(percent_decode("%ff"), None);
    }

    #[test]
    fn none_match_hits_uses_weak_comparison() {
        let etag = "\"abc\"";
        assert!(none_match_hits("\"abc\"", etag));
        assert!(none_match_hits("W/\"abc\"", etag));
        assert!(none_match_hits("\"x\", \"abc\"", etag));
        assert!(none_match_hits("*", etag));
        assert!(!none_match_hits("\"abd\"", etag));
    }

    #[test]
    fn if_range_holds_for_the_strong_tag_or_the_date() {
        let etag = "\"abc\"";
        assert!(if_range_holds("\"abc\"", etag));
        assert!(if_range_holds(LAST_MODIFIED, etag));
        assert!(!if_range_holds("W/\"abc\"", etag));
        assert!(!if_range_holds("Fri, 30 Oct 2026 12:00:00 GMT", etag));
    }

    #[test]
    fn range_answer_reads_one_byte_range() {
        assert_eq!(range_answer("bytes=0-7", 100), Part { start: 0, end: 7 });
        assert_eq!(range_answer("bytes=90-", 100), Part { start: 90, end: 99 });
        assert_eq!(
            range_answer("bytes=90-500", 100),
            Part { start: 90, end: 99 }
        );
        assert_eq!(range_answer("bytes=-10", 100), Part { start: 90, end: 99 });
        assert_eq!(range_answer("bytes=-500", 100), Part { start: 0, end: 99 });
        assert_eq!(range_answer("BYTES=1-1", 100), Part { start: 1, end: 1 });
    }

    #[test]
    fn range_answer_refuses_ranges_past_the_end() {
        assert_eq!(range_answer("bytes=100-", 100), Unsatisfiable);
        assert_eq!(range_answer("bytes=100-200", 100), Unsatisfiable);
        assert_eq!(range_answer("bytes=-0", 100), Unsatisfiable);
        assert_eq!(range_answer("bytes=-5", 0), Unsatisfiable);
    }

    #[test]
    fn range_answer_ignores_several_or_malformed_ranges() {
        for header in [
            "bytes=0-1,5-6",
            "bytes=7-3",
            "bytes=a-",
            "bytes=-",
            "bytes=5",
            "items=0-1",
            "bytes=+1-2",
            "",
        ] {
            assert_eq!(range_answer(header, 100), Whole, "{header:?}");
        }
    }

    #[test]
    fn content_type_by_extension() {
        assert_eq!(content_type("index.json"), "application/json");
        assert_eq!(content_type("apache.html"), "text/html; charset=utf-8");
        assert_eq!(content_type("notes.txt"), "text/plain; charset=utf-8");
        assert_eq!(
            content_type("probe p1.safetensors"),
            "application/octet-stream"
        );
        assert_eq!(content_type("README"), "application/octet-stream");
    }
}
