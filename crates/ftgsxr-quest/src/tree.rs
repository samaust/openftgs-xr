// SPDX-License-Identifier: Apache-2.0

//! The shim's `listTree` rows as `DocumentProvider` entries (§2.5 The shim, §1.5 Picked
//! files, D-49, Q-47, Q-T458.3): one `name\tsize\tmime\turi` line per document, classified
//! by its mime (a directory) and its name (`classify_name`). Built on the host too, so its
//! tests run there.

use ftgsxr_app::{Entry, EntryKind, IoError, classify_name};

/// The mime type Android's document providers give a directory
/// (`DocumentsContract.Document.MIME_TYPE_DIR`).
pub(crate) const DIR_MIME: &str = "vnd.android.document/directory";
/// A checkpoint directory's two documents (§1.2, D-51).
const MANIFEST: &str = "manifest.json";
const PARAMS: &str = "params.safetensors";

/// One document of a tree, as `listTree` gives it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Row {
    /// The display name.
    pub(crate) name: String,
    /// The size in bytes, when the provider gives one.
    pub(crate) size: Option<u64>,
    /// The mime type.
    pub(crate) mime: String,
    /// The document's content URI, built on the tree's (a directory's lists its children).
    pub(crate) uri: String,
}

/// `line` as a row: four tab-separated fields; a size that is empty or not a
/// non-negative integer is unknown. `None` for a line without four fields.
pub(crate) fn parse_row(line: &str) -> Option<Row> {
    let mut fields = line.splitn(4, '\t');
    let (name, size, mime, uri) = (
        fields.next()?,
        fields.next()?,
        fields.next()?,
        fields.next()?,
    );
    if name.is_empty() || uri.is_empty() {
        return None;
    }
    Some(Row {
        name: name.to_owned(),
        size: size.parse().ok(),
        mime: mime.to_owned(),
        uri: uri.to_owned(),
    })
}

/// The entries of a tree whose rows are `rows` (Q-T458.3), sorted by name, dot-names
/// skipped (as `list_dir`, U-27): a directory holding `manifest.json` and
/// `params.safetensors` is a `Checkpoint` (its size the two files' together, when both
/// are known), found by listing it through `list`; any other directory, or one that cannot
/// be listed, is a `Dir` whose location lists it on demand (Q-47); a file is
/// `classify_name`'s kind, named by its stem (Q-36). The location is the row's URI.
// Q-T458.3
pub(crate) fn entries(
    rows: Vec<Row>,
    list: &dyn Fn(&str) -> Result<Vec<Row>, IoError>,
) -> Vec<Entry> {
    let mut rows: Vec<Row> = rows
        .into_iter()
        .filter(|r| !r.name.starts_with('.'))
        .collect();
    rows.sort_by(|a, b| a.name.cmp(&b.name));
    rows.into_iter()
        .map(|row| {
            let (kind, name, size) = if row.mime == DIR_MIME {
                match list(&row.uri).ok().and_then(|c| checkpoint_size(&c)) {
                    Some(size) => (EntryKind::Checkpoint, row.name.clone(), size),
                    None => (EntryKind::Dir, row.name.clone(), None),
                }
            } else {
                let kind = classify_name(&row.name);
                (kind, stem(&row.name, kind).to_owned(), row.size)
            };
            Entry {
                name,
                location: row.uri,
                kind,
                size,
                mtime: None,
                sha256: None,
                thumbnail: None,
                audio: None,
                scene: None,
            }
        })
        .collect()
}

/// `Some` (the two files' size together, when both are known) when `children` hold a
/// checkpoint's two files.
fn checkpoint_size(children: &[Row]) -> Option<Option<u64>> {
    let file = |name: &str| {
        children
            .iter()
            .find(|r| r.name == name && r.mime != DIR_MIME)
    };
    let (manifest, params) = (file(MANIFEST)?, file(PARAMS)?);
    Some(manifest.size.zip(params.size).map(|(m, p)| m + p))
}

/// A file's name without the suffix its kind says (Q-36's stem, ASCII case ignored).
fn stem(name: &str, kind: EntryKind) -> &str {
    let suffix = match kind {
        EntryKind::Exchange => ".safetensors".len(),
        EntryKind::Ply => ".ply".len(),
        _ => 0,
    };
    &name[..name.len() - suffix]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(name: &str, size: Option<u64>, mime: &str, uri: &str) -> Row {
        Row {
            name: name.into(),
            size,
            mime: mime.into(),
            uri: uri.into(),
        }
    }

    fn entry(name: &str, uri: &str, kind: EntryKind, size: Option<u64>) -> Entry {
        Entry {
            name: name.into(),
            location: uri.into(),
            kind,
            size,
            mtime: None,
            sha256: None,
            thumbnail: None,
            audio: None,
            scene: None,
        }
    }

    #[test]
    fn parse_row_reads_the_four_fields() {
        assert_eq!(
            parse_row("probe p1.safetensors\t1234\tapplication/octet-stream\tcontent://a/b%2Fc"),
            Some(row(
                "probe p1.safetensors",
                Some(1234),
                "application/octet-stream",
                "content://a/b%2Fc"
            ))
        );
        // An unknown size (the provider's null, or -1) is none.
        assert_eq!(parse_row("d\t\tm\tu").unwrap().size, None);
        assert_eq!(parse_row("d\t-1\tm\tu").unwrap().size, None);
        assert_eq!(parse_row("d\t1\tm"), None);
        assert_eq!(parse_row("\t1\tm\tu"), None);
        assert_eq!(parse_row(""), None);
    }

    #[test]
    fn entries_classify_by_mime_and_name() {
        let rows = vec![
            row("snapshot.PLY", Some(7), "application/octet-stream", "u:ply"),
            row("tiny", None, DIR_MIME, "u:tiny"),
            row(
                "probe p1.safetensors",
                Some(9),
                "application/octet-stream",
                "u:probe",
            ),
            row(
                ".hidden.safetensors",
                Some(1),
                "application/octet-stream",
                "u:hidden",
            ),
            row("notes.txt", Some(3), "text/plain", "u:notes"),
            row("plain", None, DIR_MIME, "u:plain"),
            row("broken", None, DIR_MIME, "u:broken"),
        ];
        let list = |uri: &str| match uri {
            "u:tiny" => Ok(vec![
                row("manifest.json", Some(10), "application/json", "u:tiny/m"),
                row(
                    "params.safetensors",
                    Some(20),
                    "application/octet-stream",
                    "u:tiny/p",
                ),
            ]),
            "u:plain" => Ok(vec![row(
                "manifest.json",
                Some(10),
                "application/json",
                "u:pm",
            )]),
            _ => Err(IoError::Io(std::io::ErrorKind::NotFound.into())),
        };
        assert_eq!(
            entries(rows, &list),
            vec![
                entry("broken", "u:broken", EntryKind::Dir, None),
                entry("notes.txt", "u:notes", EntryKind::Other, Some(3)),
                entry("plain", "u:plain", EntryKind::Dir, None),
                entry("probe p1", "u:probe", EntryKind::Exchange, Some(9)),
                entry("snapshot", "u:ply", EntryKind::Ply, Some(7)),
                entry("tiny", "u:tiny", EntryKind::Checkpoint, Some(30)),
            ]
        );
    }

    #[test]
    fn a_checkpoint_listing_names_its_members_as_document_source_finds_them() {
        // Q-T458.1: `DocumentSource` looks for the `Other` entry `manifest.json` and the
        // `Exchange` entry `params` among the directory's entries.
        let rows = vec![
            row(
                "params.safetensors",
                None,
                "application/octet-stream",
                "u:p",
            ),
            row("manifest.json", Some(10), "application/json", "u:m"),
        ];
        let none = |_: &str| Ok(Vec::new());
        assert_eq!(
            entries(rows, &none),
            vec![
                entry("manifest.json", "u:m", EntryKind::Other, Some(10)),
                entry("params", "u:p", EntryKind::Exchange, None),
            ]
        );
        // A checkpoint whose sizes the provider does not give has none.
        let rows = vec![row("c", None, DIR_MIME, "u:c")];
        let list = |_: &str| {
            Ok(vec![
                row("manifest.json", None, "application/json", "u:m"),
                row(
                    "params.safetensors",
                    Some(2),
                    "application/octet-stream",
                    "u:p",
                ),
            ])
        };
        assert_eq!(
            entries(rows, &list),
            vec![entry("c", "u:c", EntryKind::Checkpoint, None)]
        );
    }
}
