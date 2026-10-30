// SPDX-License-Identifier: Apache-2.0

//! `HttpClient::download_checkpoint`: a checkpoint's two files downloaded into one cache
//! entry, which is itself the checkpoint directory `read_checkpoint` reads (§1.5 Loading and
//! unloading, §1.11 `index.json` and Cache entries, D-51, Q-42, Q-57, Q-59).

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use serde_json::Value;
use url::Url;

use super::cache::{Entry, entry_dir};
use super::download::parse_digest;
use super::{HttpClient, Progress, lock};
use crate::IoError;

/// The first member of a checkpoint's entry: its manifest (§1.11 `index.json`, D-51).
pub(crate) const MANIFEST: &str = "manifest.json";
/// The second member: its parameters, the file `read_checkpoint` activates (§1.2, D-51).
pub(crate) const PARAMS: &str = "params.safetensors";

/// The largest `manifest.json` read for the params digest; openftgs's are a few kilobytes.
const MAX_MANIFEST: u64 = 16 << 20;

impl HttpClient {
    /// Downloads the checkpoint at the directory URL `dir_url` (ending in `/`, Q-36) into its
    /// cache entry `<cache dir>/http/<sha256 of dir_url>/` and returns the entry's directory
    /// (D-51, Q-42): `manifest.json` first, then `params.safetensors` against `params_sha256`
    /// (the index entry's digest) when given, else the manifest's digest for it, a mismatch
    /// being E-NET-CHECKSUM (Q-57). Each member is cached,
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
        let expected = given.or_else(|| params_digest(&path));
        let params = Entry::member(&dir, PARAMS);
        self.download_into(
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
        )?;
        Ok(dir)
    }
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
