// SPDX-License-Identifier: Apache-2.0

//! `HttpClient::download_checkpoint`: a checkpoint's two files downloaded into one cache
//! entry, which is itself the checkpoint directory `read_checkpoint` reads (§1.5 Loading and
//! unloading, §1.11 `index.json` and Cache entries, D-51, Q-42, Q-57, Q-59), and
//! `HttpClient::copy_checkpoint`: a picked checkpoint's two documents copied into one
//! entry of the same layout (Q-61, Q-T458.1).

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::Value;
use sha2::{Digest, Sha256};
use url::Url;

use super::cache::{Entry, Record, entry_dir, remove_if_present};
use super::download::{parse_digest, reuse};
use super::{HttpClient, Progress, lock};
use crate::IoError;
use crate::util::{cancelled, now_rfc3339};

/// The first member of a checkpoint's entry: its manifest (§1.11 `index.json`, D-51).
pub(crate) const MANIFEST: &str = "manifest.json";
/// The second member: its parameters, the file `read_checkpoint` activates (§1.2, D-51).
pub(crate) const PARAMS: &str = "params.safetensors";

/// The largest `manifest.json` read for the params digest; openftgs's are a few kilobytes.
const MAX_MANIFEST: u64 = 16 << 20;

impl HttpClient {
    /// Downloads the checkpoint at the directory URL `dir_url` (ending in `/`, Q-36, Q-60) into its
    /// cache entry `<cache dir>/http/<sha256 of dir_url>/` and returns the entry's directory
    /// (D-51, Q-42): `manifest.json` first, then `params.safetensors` against `params_sha256`
    /// (the index entry's digest) when given, else the manifest's digest for it, a mismatch
    /// being E-NET-CHECKSUM (Q-57); a mismatch with the manifest's digest discards the
    /// manifest too, so a retry fetches both (owner, 2026-10-29, #97). Each member is cached,
    /// revalidated and resumed as `download`'s file is (§1.5, Q-43); both downloads hold the
    /// cache lock, so `prune` never sees half an entry. `progress` is cumulative over both
    /// files (Q-59).
    // D-51, Q-42, Q-57, Q-59
    pub(crate) fn download_checkpoint(
        &self,
        dir_url: &Url,
        params_sha256: Option<&str>,
        progress: &mut dyn FnMut(Progress),
        cancel: &AtomicBool,
    ) -> Result<PathBuf, IoError> {
        let join = |name: &str| {
            dir_url
                .join(name)
                .map_err(|e| IoError::Fetch(format!("{dir_url}{name} ({e})")))
        };
        let (manifest_url, params_url) = (join(MANIFEST)?, join(PARAMS)?);
        let given = match params_sha256 {
            None => None,
            // Q-43: a malformed digest is refused before any request.
            Some(digest) => Some(parse_digest(digest).ok_or_else(|| {
                IoError::Checksum(format!(
                    "{dir_url}: expected digest {digest:?} is not SHA-256"
                ))
            })?),
        };
        let _guard = lock(&self.cache_lock);
        let dir = entry_dir(&self.entries, dir_url);
        let manifest = Entry::member(&dir, MANIFEST);
        // Q-59: the params size is not known yet.
        let path = self.download_into(
            &manifest,
            &manifest_url,
            None,
            &mut |p| {
                progress(Progress {
                    done: p.done,
                    total: None,
                })
            },
            cancel,
        )?;
        let held = std::fs::metadata(&path)?.len();
        let expected = given.clone().or_else(|| params_digest(&path));
        let params = Entry::member(&dir, PARAMS);
        let result = self.download_into(
            &params,
            &params_url,
            expected,
            &mut |p| {
                progress(Progress {
                    done: held + p.done,
                    total: p.total.map(|t| held + t),
                })
            },
            cancel,
        );
        // Q-57: a manifest whose digest the params disagree with may itself be the bad file,
        // kept by every 304; dropping it lets the retry E-NET-CHECKSUM offers fetch it again.
        if given.is_none() && matches!(result, Err(IoError::Checksum(_))) {
            remove_if_present(&manifest.meta)?;
            remove_if_present(&manifest.file)?;
        }
        result?;
        Ok(dir)
    }
}

/// One document of a picked checkpoint directory (Q-61, Q-T458.1): its content URI, the
/// size its listing gives, and how to open it.
pub(crate) struct DocumentMember<'a> {
    /// The document's content URI, the member's record `url`.
    pub(crate) uri: Url,
    /// The size the provider's listing gives, which a reused copy must still have.
    pub(crate) size: Option<u64>,
    /// Opens the document for reading.
    pub(crate) open: &'a dyn Fn() -> Result<File, IoError>,
}

/// The bytes a document member is copied in between two cancel checks (Q-47's chunk).
const COPY_CHUNK: usize = 1 << 20;

impl HttpClient {
    /// Copies the picked checkpoint directory `dir_uri` (a document URI, Q-61) into its
    /// cache entry `<cache dir>/http/<sha256 of dir_uri>/` and returns the entry's
    /// directory, as `download_checkpoint` fills one from a URL (D-51, Q-42): `manifest.json`
    /// first, then `params.safetensors` against the manifest's digest for it, a mismatch
    /// being E-NET-CHECKSUM that discards both members (Q-57). A member whose complete
    /// record names its URI, has the size the listing gives now and, for the params, the
    /// manifest's digest, is reused without opening the document; any other is copied again
    /// (Q-T458.1). Both copies hold the cache lock, so `prune` never sees half an entry;
    /// `progress` is cumulative over both (Q-59); `cancel` is read before each chunk, and a
    /// cancelled copy keeps nothing (E-IO `Interrupted`, Q-47).
    // D-51, Q-57, Q-59, Q-61, Q-T458.1
    pub(crate) fn copy_checkpoint(
        &self,
        dir_uri: &Url,
        manifest: &DocumentMember<'_>,
        params: &DocumentMember<'_>,
        progress: &mut dyn FnMut(Progress),
        cancel: &AtomicBool,
    ) -> Result<PathBuf, IoError> {
        let _guard = lock(&self.cache_lock);
        let dir = entry_dir(&self.entries, dir_uri);
        std::fs::create_dir_all(&dir)?;
        let manifest_entry = Entry::member(&dir, MANIFEST);
        // Q-59: the params size is not known yet.
        let path = copy_member(
            &manifest_entry,
            manifest,
            None,
            &mut |p| {
                progress(Progress {
                    done: p.done,
                    total: None,
                })
            },
            cancel,
        )?;
        let held = std::fs::metadata(&path)?.len();
        let expected = params_digest(&path);
        let result = copy_member(
            &Entry::member(&dir, PARAMS),
            params,
            expected,
            &mut |p| {
                progress(Progress {
                    done: held + p.done,
                    total: p.total.map(|t| held + t),
                })
            },
            cancel,
        );
        // Q-57: the manifest the params disagree with may be the bad file; a retry copies
        // both.
        if matches!(result, Err(IoError::Checksum(_))) {
            remove_if_present(&manifest_entry.meta)?;
            remove_if_present(&manifest_entry.file)?;
        }
        result?;
        Ok(dir)
    }
}

/// One member of `copy_checkpoint`: reused when its complete record is current
/// (Q-T458.1), else copied through `<name>.part` and checked against `expected`.
// Q-T458.1
fn copy_member(
    entry: &Entry,
    member: &DocumentMember<'_>,
    expected: Option<String>,
    progress: &mut dyn FnMut(Progress),
    cancel: &AtomicBool,
) -> Result<PathBuf, IoError> {
    let uri = &member.uri;
    if cancel.load(Ordering::Relaxed) {
        return Err(cancelled(uri));
    }
    let current = entry.complete(Some(uri)).filter(|record| {
        member.size.is_none_or(|size| record.size == Some(size))
            && expected
                .as_ref()
                .is_none_or(|e| record.sha256.as_ref() == Some(e))
    });
    if let Some(record) = current {
        return reuse(entry, record, progress);
    }
    entry.discard_partial()?;
    let mut source = (member.open)()?;
    let total = source.metadata()?.len();
    let mut file = File::create(&entry.part)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; COPY_CHUNK];
    let mut done = 0u64;
    loop {
        if cancel.load(Ordering::Relaxed) {
            drop(file);
            entry.discard_partial()?;
            return Err(cancelled(uri));
        }
        let n = match source.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => {
                drop(file);
                entry.discard_partial()?;
                return Err(e.into());
            }
        };
        file.write_all(&buf[..n])?;
        hasher.update(&buf[..n]);
        done += n as u64;
        progress(Progress {
            done,
            total: Some(total),
        });
    }
    file.sync_all()?;
    drop(file);
    let digest = ftgsxr_core::hex(&hasher.finalize().into());
    if expected.as_ref().is_some_and(|e| *e != digest) {
        entry.discard_partial()?;
        return Err(IoError::Checksum(uri.to_string()));
    }
    std::fs::rename(&entry.part, &entry.file)?;
    let record = Record {
        url: uri.as_str().to_owned(),
        etag: None,
        last_modified: None,
        size: Some(done),
        sha256: Some(digest),
        fetched_at: now_rfc3339(),
    };
    record.write(&entry.meta)?;
    Ok(entry.file.clone())
}

/// The manifest's digest of `params.safetensors` (`files`, openftgs `ftgs-checkpoint`), or
/// `None` when the manifest cannot be read or gives none: `read_checkpoint` then reports
/// the fault (Q-34, Q-57).
// Q-57
fn params_digest(manifest: &Path) -> Option<String> {
    if std::fs::metadata(manifest).ok()?.len() > MAX_MANIFEST {
        return None;
    }
    let value: Value = serde_json::from_slice(&std::fs::read(manifest).ok()?).ok()?;
    parse_digest(value.get("files")?.get(PARAMS)?.as_str()?)
}
