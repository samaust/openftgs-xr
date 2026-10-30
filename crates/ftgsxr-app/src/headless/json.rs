// SPDX-License-Identifier: Apache-2.0

//! A small JSON value for the benchmark files (Q-67, Q-T122.2): written in a fixed key
//! order and read back for the baseline comparison. §2.1's row for `ftgsxr-app` names no
//! JSON crate, so the module writes and parses the subset the files use (RFC 8259 without
//! surrogate-pair escapes beyond the basic plane's `\uXXXX`).

use std::fmt::Write as _;

/// A JSON value; an object keeps its keys in insertion order.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    /// An object from `(key, value)` pairs.
    pub(crate) fn obj<const N: usize>(pairs: [(&str, Json); N]) -> Json {
        Json::Obj(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
    }

    /// A string value.
    pub(crate) fn str(s: impl Into<String>) -> Json {
        Json::Str(s.into())
    }

    /// The value of `key` in an object.
    pub(crate) fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Whether `self` and `other` are the same JSON value, objects compared as sets of
    /// members (RFC 8259: an object is unordered), so a baseline rewritten by another
    /// tool still matches.
    pub(crate) fn same(&self, other: &Json) -> bool {
        match (self, other) {
            (Json::Obj(a), Json::Obj(b)) => {
                a.len() == b.len()
                    && a.iter()
                        .all(|(k, v)| other.get(k).is_some_and(|w| v.same(w)))
            }
            (Json::Arr(a), Json::Arr(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(v, w)| v.same(w))
            }
            (a, b) => a == b,
        }
    }

    /// The number, if this is one.
    pub(crate) fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(v) => Some(*v),
            _ => None,
        }
    }

    /// The string, if this is one.
    pub(crate) fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    /// The elements, if this is an array.
    pub(crate) fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(v) => Some(v),
            _ => None,
        }
    }

    /// Compact JSON text; a non-finite number is written `null`.
    pub(crate) fn compact(&self) -> String {
        let mut out = String::new();
        self.write(&mut out, None, 0);
        out
    }

    /// JSON text indented by two spaces per level, one member or element per line.
    pub(crate) fn pretty(&self) -> String {
        let mut out = String::new();
        self.write(&mut out, Some(2), 0);
        out
    }

    fn write(&self, out: &mut String, indent: Option<usize>, level: usize) {
        let newline = |out: &mut String, level: usize| {
            if let Some(step) = indent {
                out.push('\n');
                out.extend(std::iter::repeat_n(' ', step * level));
            }
        };
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Num(v) if v.is_finite() => {
                let _ = write!(out, "{v}");
            }
            Json::Num(_) => out.push_str("null"),
            Json::Str(s) => write_string(out, s),
            Json::Arr(items) if items.is_empty() => out.push_str("[]"),
            Json::Arr(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    newline(out, level + 1);
                    item.write(out, indent, level + 1);
                }
                newline(out, level);
                out.push(']');
            }
            Json::Obj(pairs) if pairs.is_empty() => out.push_str("{}"),
            Json::Obj(pairs) => {
                out.push('{');
                for (i, (key, value)) in pairs.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    newline(out, level + 1);
                    write_string(out, key);
                    out.push(':');
                    if indent.is_some() {
                        out.push(' ');
                    }
                    value.write(out, indent, level + 1);
                }
                newline(out, level);
                out.push('}');
            }
        }
    }

    /// Parses JSON text; the error names the byte offset.
    pub(crate) fn parse(text: &str) -> Result<Json, String> {
        let mut p = Parser {
            bytes: text.as_bytes(),
            at: 0,
        };
        let value = p.value()?;
        p.space();
        if p.at != p.bytes.len() {
            return Err(format!("trailing characters at byte {}", p.at));
        }
        Ok(value)
    }
}

fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if u32::from(c) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", u32::from(c));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

struct Parser<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Parser<'_> {
    fn space(&mut self) {
        while self
            .bytes
            .get(self.at)
            .is_some_and(|b| matches!(b, b' ' | b'\t' | b'\n' | b'\r'))
        {
            self.at += 1;
        }
    }

    fn error(&self, what: &str) -> String {
        format!("{what} at byte {}", self.at)
    }

    fn expect(&mut self, word: &str) -> Result<(), String> {
        if self.bytes[self.at..].starts_with(word.as_bytes()) {
            self.at += word.len();
            Ok(())
        } else {
            Err(self.error(&format!("expected {word}")))
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        self.space();
        match self.bytes.get(self.at) {
            None => Err(self.error("unexpected end")),
            Some(b'n') => self.expect("null").map(|()| Json::Null),
            Some(b't') => self.expect("true").map(|()| Json::Bool(true)),
            Some(b'f') => self.expect("false").map(|()| Json::Bool(false)),
            Some(b'"') => self.string().map(Json::Str),
            Some(b'[') => {
                self.at += 1;
                let mut items = Vec::new();
                self.space();
                if self.bytes.get(self.at) == Some(&b']') {
                    self.at += 1;
                    return Ok(Json::Arr(items));
                }
                loop {
                    items.push(self.value()?);
                    self.space();
                    match self.bytes.get(self.at) {
                        Some(b',') => self.at += 1,
                        Some(b']') => {
                            self.at += 1;
                            return Ok(Json::Arr(items));
                        }
                        _ => return Err(self.error("expected , or ]")),
                    }
                }
            }
            Some(b'{') => {
                self.at += 1;
                let mut pairs = Vec::new();
                self.space();
                if self.bytes.get(self.at) == Some(&b'}') {
                    self.at += 1;
                    return Ok(Json::Obj(pairs));
                }
                loop {
                    self.space();
                    if self.bytes.get(self.at) != Some(&b'"') {
                        return Err(self.error("expected a key"));
                    }
                    let key = self.string()?;
                    self.space();
                    self.expect(":")?;
                    pairs.push((key, self.value()?));
                    self.space();
                    match self.bytes.get(self.at) {
                        Some(b',') => self.at += 1,
                        Some(b'}') => {
                            self.at += 1;
                            return Ok(Json::Obj(pairs));
                        }
                        _ => return Err(self.error("expected , or }")),
                    }
                }
            }
            Some(_) => self.number(),
        }
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.at;
        while self
            .bytes
            .get(self.at)
            .is_some_and(|b| matches!(b, b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9'))
        {
            self.at += 1;
        }
        std::str::from_utf8(&self.bytes[start..self.at])
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .filter(|v| v.is_finite())
            .map(Json::Num)
            .ok_or_else(|| format!("invalid value at byte {start}"))
    }

    fn string(&mut self) -> Result<String, String> {
        self.at += 1; // the opening quote
        let mut out = String::new();
        loop {
            let start = self.at;
            while self
                .bytes
                .get(self.at)
                .is_some_and(|&b| b != b'"' && b != b'\\' && b >= 0x20)
            {
                self.at += 1;
            }
            out.push_str(
                std::str::from_utf8(&self.bytes[start..self.at])
                    .map_err(|_| self.error("invalid UTF-8"))?,
            );
            match self.bytes.get(self.at) {
                Some(b'"') => {
                    self.at += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.at += 1;
                    let c = match self.bytes.get(self.at) {
                        Some(b'"') => '"',
                        Some(b'\\') => '\\',
                        Some(b'/') => '/',
                        Some(b'b') => '\u{8}',
                        Some(b'f') => '\u{c}',
                        Some(b'n') => '\n',
                        Some(b'r') => '\r',
                        Some(b't') => '\t',
                        Some(b'u') => {
                            let hex = self
                                .bytes
                                .get(self.at + 1..self.at + 5)
                                .and_then(|h| std::str::from_utf8(h).ok())
                                .and_then(|h| u32::from_str_radix(h, 16).ok())
                                .and_then(char::from_u32)
                                .ok_or_else(|| self.error("invalid \\u escape"))?;
                            self.at += 4;
                            hex
                        }
                        _ => return Err(self.error("invalid escape")),
                    };
                    self.at += 1;
                    out.push(c);
                }
                _ => return Err(self.error("unterminated string")),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_compact_writes_keys_in_order_and_numbers_shortest() {
        let v = Json::obj([
            ("b", Json::Num(90.0)),
            (
                "a",
                Json::Arr(vec![Json::Num(0.1), Json::Null, Json::Bool(true)]),
            ),
            ("s", Json::str("x\"y\\\n")),
            ("nan", Json::Num(f64::NAN)),
        ]);
        assert_eq!(
            v.compact(),
            r#"{"b":90,"a":[0.1,null,true],"s":"x\"y\\\n","nan":null}"#
        );
    }

    #[test]
    fn json_parse_reads_back_what_pretty_writes() {
        let v = Json::obj([
            ("format", Json::str("ftgsxr-bench")),
            ("version", Json::Num(1.0)),
            ("empty", Json::Arr(vec![])),
            (
                "o",
                Json::obj([("x", Json::Num(-1.5e-7)), ("é", Json::Obj(vec![]))]),
            ),
        ]);
        assert_eq!(Json::parse(&v.pretty()).unwrap(), v);
        assert_eq!(Json::parse(&v.compact()).unwrap(), v);
    }

    #[test]
    fn json_parse_handles_escapes_and_whitespace() {
        let v = Json::parse(" { \"k\" : [ 1 , 2e3 , \"\\u0041\\t\" ] } ").unwrap();
        assert_eq!(
            v,
            Json::obj([(
                "k",
                Json::Arr(vec![Json::Num(1.0), Json::Num(2000.0), Json::str("A\t")])
            )])
        );
    }

    #[test]
    fn json_same_ignores_the_order_of_members() {
        let a = Json::obj([
            ("n", Json::Num(1.0)),
            ("t", Json::obj([("x", Json::Null), ("y", Json::Bool(true))])),
        ]);
        let b = Json::obj([
            ("t", Json::obj([("y", Json::Bool(true)), ("x", Json::Null)])),
            ("n", Json::Num(1.0)),
        ]);
        assert!(a.same(&b));
        let c = Json::obj([
            ("n", Json::Num(2.0)),
            ("t", Json::obj([("x", Json::Null), ("y", Json::Bool(true))])),
        ]);
        assert!(!a.same(&c));
        assert!(!a.same(&Json::obj([("n", Json::Num(1.0))])));
        assert!(
            !Json::Arr(vec![Json::Num(1.0), Json::Num(2.0)])
                .same(&Json::Arr(vec![Json::Num(2.0), Json::Num(1.0)]))
        );
    }

    #[test]
    fn json_parse_refuses_malformed_text() {
        for bad in [
            "",
            "{",
            "[1,]",
            "{\"a\" 1}",
            "\"open",
            "1 2",
            "nul",
            "{1:2}",
        ] {
            assert!(Json::parse(bad).is_err(), "{bad:?}");
        }
    }
}
