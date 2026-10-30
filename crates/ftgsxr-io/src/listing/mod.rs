// SPDX-License-Identifier: Apache-2.0

//! The listings of §1.5's sources (D-31): the player's `index.json` (`ftgsxr-index` v1,
//! §1.11), nginx's `autoindex_format json` listing, an HTML directory listing read for its
//! anchors, and the one-level scan of a local directory, each giving `Entry` values.

use std::path::Path;

use url::Url;

use crate::IoError;

mod html;
mod index;
mod local;
mod nginx;

pub(crate) use local::{checked_header_len, header_info};

/// What a listed entry is (§1.5's kinds, and a directory to list on demand).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntryKind {
    /// A directory of an HTTP listing, listed on demand (§1.5, D-31).
    Dir,
    /// An exchange model `ftgs-if` (§1.2).
    Exchange,
    /// A checkpoint directory `ftgs-checkpoint` (§1.2).
    Checkpoint,
    /// A PLY snapshot (§1.2, D-41).
    Ply,
    /// None of the above: `classify_name`'s answer for any other name (Q-36).
    Other,
}

/// One entry of a listing (§1.5; the fields' content per flavor is Q-36's).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// The name the library shows: the file's stem, a directory's name, or the index's
    /// `name`.
    pub name: String,
    /// Where the entry is: an absolute URL for HTTP listings, a path for a local directory.
    pub location: String,
    /// The entry's kind.
    pub kind: EntryKind,
    /// The size in bytes, when the listing gives it.
    pub size: Option<u64>,
    /// The modification time as the listing states it (nginx's JSON listing).
    pub mtime: Option<String>,
    /// The file's SHA-256 as 64 lowercase hexadecimal digits (the index's `sha256`).
    pub sha256: Option<String>,
    /// The index's `thumbnail`, resolved to an absolute URL.
    pub thumbnail: Option<String>,
    /// The index's `audio` soundtrack (§1.10), resolved to an absolute URL.
    pub audio: Option<String>,
    /// The index's `scene` manifest, resolved to an absolute URL.
    pub scene: Option<String>,
}

/// Which listing an HTTP source reads (§1.5, D-31; `settings.toml`'s `flavor`, §1.11).
/// §2.1 shows it beside `HttpIndex`; its home is this module (§2.1's module table).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ListingFlavor {
    /// Try `index.json`, then nginx's JSON listing, then the HTML anchors (D-31).
    Auto,
    /// The player's `index.json` (`ftgsxr-index` v1).
    Index,
    /// nginx's `autoindex_format json` listing.
    NginxJson,
    /// An HTML directory listing (Apache's `mod_autoindex`, nginx's HTML autoindex).
    Html,
}

/// Parses an `ftgsxr-index` v1 manifest (§1.11) fetched from `base`: one entry per element
/// of `entries`, in order, `location`, `thumbnail`, `audio` and `scene` resolved against
/// `base`, `sha256` lowercase (Q-36). A body that is not JSON is E-JSON; one that breaks
/// the schema (format, major version, an entry's keys and types) is E-SCENE-FORMAT; unknown
/// keys are ignored (Q-37).
// Q-36, Q-37
pub fn parse_index_manifest(body: &str, base: &Url) -> Result<Vec<Entry>, IoError> {
    index::parse(body, base)
}

/// Parses nginx's `autoindex_format json` listing of `base`: its directories (`Dir`) and
/// the files `classify_name` calls `Exchange` or `Ply`, in order, dot-names skipped, each
/// named by its stem with its percent-encoded name resolved against `base`, with nginx's
/// `size` and `mtime` (Q-36). A body that is not an array of objects with string `name`
/// and `type` is E-NET-LISTING (Q-37).
// Q-36, Q-37
pub fn parse_nginx_json(body: &str, base: &Url) -> Result<Vec<Entry>, IoError> {
    nginx::parse(body, base)
}

/// Parses an HTML directory listing of `base` for its anchors' `href` values (§1.5): each
/// `href`, its character references decoded, is resolved against `base` and kept when it
/// names an entry of `base`'s directory (so the parent and column-sort links are skipped)
/// whose percent-decoded name ends with `.safetensors`, `.ply` or `/` and does not start
/// with `.`; repeated locations are kept once; no size (Q-36). A body without any `<a>`
/// element is E-NET-LISTING (Q-37).
// Q-36, Q-37
pub fn parse_html_listing(body: &str, base: &Url) -> Result<Vec<Entry>, IoError> {
    html::parse(body, base)
}

/// Lists a local directory one level deep (§1.5): the `*.safetensors` files whose header
/// says `format` `ftgs-if` (`Exchange`; the rest of the header is left to `info` and
/// loading, so a broken `ftgs-if` file is listed), the directories whose `manifest.json`
/// has `format` `ftgs-checkpoint` (`Checkpoint`) and the `*.ply` files (`Ply`), sorted by
/// file name, dot-names skipped. `location` is the entry's path, `size` the file's size (a
/// checkpoint's `manifest.json` and `params.safetensors` together). An entry that cannot be
/// read is skipped; a `path` that cannot be listed is E-IO (Q-38).
// Q-38
pub fn list_dir(path: &Path) -> Result<Vec<Entry>, IoError> {
    local::list(path)
}

/// The kind a name says: a trailing `/` is a directory, a `.safetensors` suffix an exchange
/// model, a `.ply` suffix a PLY snapshot (ASCII case ignored), anything else `Other`. A
/// name never says `Checkpoint`: only an index or a manifest does (Q-36).
// Q-36
pub fn classify_name(name: &str) -> EntryKind {
    if name.ends_with('/') {
        EntryKind::Dir
    } else if has_suffix(name, SAFETENSORS) {
        EntryKind::Exchange
    } else if has_suffix(name, PLY) {
        EntryKind::Ply
    } else {
        EntryKind::Other
    }
}

/// An exchange model's suffix.
const SAFETENSORS: &str = ".safetensors";
/// A PLY snapshot's suffix.
const PLY: &str = ".ply";

/// Whether `name` ends with `suffix`, ASCII case ignored, with something before it.
fn has_suffix(name: &str, suffix: &str) -> bool {
    name.len() > suffix.len()
        && name.as_bytes()[name.len() - suffix.len()..].eq_ignore_ascii_case(suffix.as_bytes())
}

/// The name a listing shows for a file or directory `name` of `kind`: a model's name
/// without its suffix, a directory's without its trailing `/` (§1.5's file stem).
fn stem(name: &str, kind: EntryKind) -> &str {
    match kind {
        EntryKind::Exchange => &name[..name.len() - SAFETENSORS.len()],
        EntryKind::Ply => &name[..name.len() - PLY.len()],
        _ => name.strip_suffix('/').unwrap_or(name),
    }
}

/// Whether a listing skips `name`: a dot-name, hidden by convention (U-27).
fn is_hidden(name: &str) -> bool {
    name.starts_with('.')
}

/// `name` percent-encoded as one URL path segment: every byte but the unreserved
/// characters `A–Z a–z 0–9 - . _ ~` becomes `%XX`.
fn encode_segment(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for &b in name.as_bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(char::from(b));
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}
