// SPDX-License-Identifier: Apache-2.0

//! An HTML directory listing (Apache's `mod_autoindex`, nginx's HTML autoindex; §1.5, D-31,
//! spec Appendix A row 35), read for its anchors' `href` values by a small tag scanner: no
//! HTML crate is in §2.3, and a listing needs only the `<a>` elements' attributes.

use url::Url;

use super::{Entry, EntryKind, classify_name, is_hidden, stem};
use crate::IoError;

/// Parses an HTML listing of `base` (Q-36, Q-37).
// Q-36, Q-37
pub(super) fn parse(body: &str, base: &Url) -> Result<Vec<Entry>, IoError> {
    let hrefs = anchors(body).ok_or_else(|| IoError::Listing(base.to_string()))?;
    let Ok(dir) = base.join(".") else {
        return Ok(Vec::new());
    };
    let mut entries: Vec<Entry> = Vec::new();
    for href in hrefs {
        let Some((location, segment, is_dir)) = child(base, &dir, &decode_references(&href)) else {
            continue;
        };
        let mut name = percent_decode(&segment);
        if is_dir {
            name.push('/');
        }
        let kind = classify_name(&name);
        if kind == EntryKind::Other || is_hidden(&name) {
            continue;
        }
        if entries.iter().any(|e| e.location == location) {
            continue;
        }
        entries.push(Entry {
            name: stem(&name, kind).to_owned(),
            location,
            kind,
            size: None,
            mtime: None,
            sha256: None,
            thumbnail: None,
            audio: None,
            scene: None,
        });
    }
    Ok(entries)
}

/// `href` resolved against `base`, when it names an entry of the directory `dir` (the same
/// origin, no query, one segment below `dir`'s path, optionally ending in `/`): the
/// location without its fragment, that segment (still percent-encoded) and whether it is a
/// directory. The parent link, the column-sort links and links elsewhere give `None`.
fn child(base: &Url, dir: &Url, href: &str) -> Option<(String, String, bool)> {
    let mut url = base.join(href).ok()?;
    url.set_fragment(None);
    if url.query().is_some() || url.origin() != dir.origin() {
        return None;
    }
    let rest = url.path().strip_prefix(dir.path())?;
    let (segment, is_dir) = match rest.strip_suffix('/') {
        Some(s) => (s, true),
        None => (rest, false),
    };
    if segment.is_empty() || segment.contains('/') {
        return None;
    }
    let segment = segment.to_owned();
    Some((url.into(), segment, is_dir))
}

/// The `href` values of every `<a>` element of `body`, in order, or `None` when the body
/// has no `<a>` element (not an HTML listing). Tag and attribute names are matched in any
/// case; values may be double-quoted, single-quoted or bare; comments are skipped.
fn anchors(body: &str) -> Option<Vec<String>> {
    let bytes = body.as_bytes();
    let mut hrefs = Vec::new();
    let mut seen_anchor = false;
    let mut i = 0;
    while let Some(off) = body[i..].find('<') {
        i += off + 1;
        if body[i..].starts_with("!--") {
            i = body[i..].find("-->").map_or(bytes.len(), |e| i + e + 3);
            continue;
        }
        let name_end = i + body[i..]
            .find(|c: char| !c.is_ascii_alphanumeric())
            .unwrap_or(body.len() - i);
        let is_anchor = body[i..name_end].eq_ignore_ascii_case("a");
        i = name_end;
        let mut href = None;
        // The attributes, up to the tag's `>`.
        loop {
            while i < bytes.len() && (bytes[i].is_ascii_whitespace() || bytes[i] == b'/') {
                i += 1;
            }
            if i >= bytes.len() || bytes[i] == b'>' {
                break;
            }
            let start = i;
            while i < bytes.len()
                && !matches!(bytes[i], b'=' | b'>' | b'/')
                && !bytes[i].is_ascii_whitespace()
            {
                i += 1;
            }
            let attr = &body[start..i];
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i >= bytes.len() || bytes[i] != b'=' {
                continue;
            }
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            let value = match bytes.get(i) {
                Some(&q @ (b'"' | b'\'')) => {
                    let end = body[i + 1..]
                        .find(char::from(q))
                        .map_or(bytes.len(), |e| i + 1 + e);
                    let v = &body[i + 1..end];
                    i = (end + 1).min(bytes.len());
                    v
                }
                _ => {
                    let start = i;
                    while i < bytes.len() && bytes[i] != b'>' && !bytes[i].is_ascii_whitespace() {
                        i += 1;
                    }
                    &body[start..i]
                }
            };
            if is_anchor && href.is_none() && attr.eq_ignore_ascii_case("href") {
                href = Some(value.to_owned());
            }
        }
        if is_anchor {
            seen_anchor = true;
            hrefs.extend(href);
        }
    }
    seen_anchor.then_some(hrefs)
}

/// `text` with its HTML character references decoded: `&amp;`, `&lt;`, `&gt;`, `&quot;`,
/// `&apos;` and the numeric `&#N;` and `&#xH;`; anything else is kept as written.
fn decode_references(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        rest = &rest[amp..];
        let decoded = rest.find(';').and_then(|semi| {
            let c = match &rest[1..semi] {
                "amp" => Some('&'),
                "lt" => Some('<'),
                "gt" => Some('>'),
                "quot" => Some('"'),
                "apos" => Some('\''),
                r => r
                    .strip_prefix("#x")
                    .or_else(|| r.strip_prefix("#X"))
                    .map(|h| u32::from_str_radix(h, 16))
                    .or_else(|| r.strip_prefix('#').map(str::parse::<u32>))
                    .and_then(Result::ok)
                    .and_then(char::from_u32),
            }?;
            Some((c, semi + 1))
        });
        match decoded {
            Some((c, len)) => {
                out.push(c);
                rest = &rest[len..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// `segment` with its `%XX` escapes decoded, as UTF-8 (invalid sequences replaced); a `%`
/// not followed by two hexadecimal digits is kept, and `+` stays a plus in a path.
pub(crate) fn percent_decode(segment: &str) -> String {
    let bytes = segment.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let hex = |b: u8| char::from(b).to_digit(16);
        match (
            bytes[i],
            bytes.get(i + 1).copied().and_then(hex),
            bytes.get(i + 2).copied().and_then(hex),
        ) {
            (b'%', Some(h), Some(l)) => {
                out.push((h * 16 + l) as u8);
                i += 3;
            }
            (b, _, _) => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchors_cases_quotes_comments() {
        let body = r#"<!-- <a href="no"> --><A HREF='one'>1</A><a class=x href=two>2</a><a name="n">
            <img src="x.png"><a  href = "three" >3</a>"#;
        assert_eq!(anchors(body).unwrap(), vec!["one", "two", "three"]);
        assert_eq!(anchors("<p>no anchors</p>"), None);
        assert_eq!(anchors(r#"[{"name": "a"}]"#), None);
        assert_eq!(anchors("<a>").unwrap(), Vec::<String>::new());
    }

    #[test]
    fn decode_references_named_and_numeric() {
        assert_eq!(
            decode_references("a&amp;b&#38;c&#x26;d&lt;&bogus;&"),
            "a&b&c&d<&bogus;&"
        );
    }

    #[test]
    fn percent_decode_utf8_and_stray_percent() {
        assert_eq!(percent_decode("probe%20p1%C3%A9%zz+%4"), "probe p1é%zz+%4");
    }
}
