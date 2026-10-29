// SPDX-License-Identifier: Apache-2.0

//! The one-level scan of a local directory (§1.5's local directory paragraph).

use std::fs::{self, File};
use std::io::Read;
use std::path::Path;

use ftgsxr_core::{CoreError, ModelInfo, SAFETENSORS_PREFIX, header_len, read_info};
use serde_json::Value;

use super::{Entry, EntryKind, classify_name, is_hidden, stem};
use crate::IoError;

/// A checkpoint directory's manifest file (§1.2).
const MANIFEST: &str = "manifest.json";
/// A checkpoint directory's parameters file, the other file its reader reads (Q-34).
const PARAMS: &str = "params.safetensors";
/// The checkpoint's `format`.
const CHECKPOINT_FORMAT: &str = "ftgs-checkpoint";
/// The largest safetensors header read (`checked_header_len`): the `safetensors` crate's own limit (source log 39).
const MAX_HEADER: u64 = 100_000_000;
/// The largest `manifest.json` read; openftgs's are a few kilobytes.
const MAX_MANIFEST: u64 = 16 << 20;

/// Lists `path` one level deep (Q-38).
// Q-38
pub(super) fn list(path: &Path) -> Result<Vec<Entry>, IoError> {
    let read = fs::read_dir(path).map_err(|e| {
        IoError::Io(std::io::Error::new(
            e.kind(),
            format!("{}: {e}", path.display()),
        ))
    })?;
    let mut found: Vec<(String, Entry)> = Vec::new();
    for item in read.flatten() {
        let Ok(file_name) = item.file_name().into_string() else {
            continue;
        };
        if is_hidden(&file_name) {
            continue;
        }
        let full = path.join(&file_name);
        let Some(location) = full.to_str().map(str::to_owned) else {
            continue;
        };
        // `fs::metadata` follows symbolic links.
        let Ok(meta) = fs::metadata(&full) else {
            continue;
        };
        let found_kind = if meta.is_dir() {
            checkpoint_size(&full).map(|size| (EntryKind::Checkpoint, size))
        } else {
            match classify_name(&file_name) {
                EntryKind::Exchange if is_exchange(&full, meta.len()) => {
                    Some((EntryKind::Exchange, meta.len()))
                }
                EntryKind::Ply => Some((EntryKind::Ply, meta.len())),
                _ => None,
            }
        };
        let Some((kind, size)) = found_kind else {
            continue;
        };
        let name = stem(&file_name, kind).to_owned();
        found.push((
            file_name,
            Entry {
                name,
                location,
                kind,
                size: Some(size),
                mtime: None,
                sha256: None,
                thumbnail: None,
                audio: None,
                scene: None,
            },
        ));
    }
    found.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(found.into_iter().map(|(_, e)| e).collect())
}

/// The header length `prefix` gives for a safetensors file of `len` bytes; a header longer
/// than the file or than `MAX_HEADER` is E-FILE-FORMAT.
pub(crate) fn checked_header_len(
    prefix: &[u8; SAFETENSORS_PREFIX],
    len: u64,
) -> Result<u64, IoError> {
    let n = header_len(prefix);
    if n > MAX_HEADER || n > len.saturating_sub(SAFETENSORS_PREFIX as u64) {
        return Err(
            CoreError::Format(format!("a header of {n} bytes in a file of {len} bytes")).into(),
        );
    }
    Ok(n)
}

/// Whether the safetensors file `path` of `len` bytes has a header `read_info` accepts:
/// the 8-byte prefix, then the header alone (Q-33).
fn is_exchange(path: &Path, len: u64) -> bool {
    File::open(path).is_ok_and(|mut file| header_info(&mut file, len).is_ok())
}

/// The header-only information of the exchange model `reader` starts, a file of `len`
/// bytes: its 8-byte prefix, then the header alone, read and given to `read_info` (§1.5,
/// Q-33); nothing past the header is read. A header longer than the file or than
/// `MAX_HEADER` is E-FILE-FORMAT, a failed read E-IO.
// Q-33
pub(crate) fn header_info(reader: &mut impl Read, len: u64) -> Result<ModelInfo, IoError> {
    let mut prefix = [0u8; SAFETENSORS_PREFIX];
    reader.read_exact(&mut prefix)?;
    let n = checked_header_len(&prefix, len)?;
    let mut header = vec![0u8; n as usize];
    reader.read_exact(&mut header)?;
    Ok(read_info(&header, len)?)
}

/// The size of the checkpoint directory `dir` (its manifest and parameters files, Q-34),
/// or `None` when its `manifest.json` does not have `format` `ftgs-checkpoint`.
fn checkpoint_size(dir: &Path) -> Option<u64> {
    let manifest = dir.join(MANIFEST);
    let manifest_len = fs::metadata(&manifest).ok().filter(|m| m.is_file())?.len();
    if manifest_len > MAX_MANIFEST {
        return None;
    }
    let value: Value = serde_json::from_slice(&fs::read(&manifest).ok()?).ok()?;
    if value.get("format").and_then(Value::as_str) != Some(CHECKPOINT_FORMAT) {
        return None;
    }
    let params_len = fs::metadata(dir.join(PARAMS)).map_or(0, |m| m.len());
    Some(manifest_len + params_len)
}
