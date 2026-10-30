// SPDX-License-Identifier: Apache-2.0

//! `HttpClient::prune` and `set_in_use`: the download cache's size limit (§1.5, §1.11 Cache
//! entries, Q-4, Q-45).

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use super::cache::Entry;
use super::checkpoint::{MANIFEST, PARAMS};
use super::{HttpClient, lock};
use crate::IoError;

/// One cache entry as `prune` sees it.
struct Candidate {
    dir: PathBuf,
    /// The bytes of every file in the entry.
    bytes: u64,
    /// Its last use, `fetched_at`; empty (the oldest) without a readable record.
    used: String,
    in_use: bool,
}

impl HttpClient {
    /// Replaces the set of cached files `prune` must keep, the paths `download` returned for
    /// the models now loaded, or a fetched checkpoint's entry directory (Q-4, Q-56); the
    /// application calls it before pruning.
    pub fn set_in_use(&self, paths: &[PathBuf]) {
        *lock(&self.in_use) = paths.iter().map(|p| canonical(p)).collect();
    }

    /// Evicts whole cache entries, least recently used first, until the download cache holds
    /// at most `max_bytes`, never one whose directory or one of whose files is in use (§1.5,
    /// §1.11, Q-4, Q-45, Q-56);
    /// returns the bytes freed.
    pub fn prune(&self) -> Result<u64, IoError> {
        let _guard = lock(&self.cache_lock);
        let in_use = lock(&self.in_use).clone();
        let mut candidates = self.candidates(&in_use)?;
        candidates.sort_by(|a, b| (&a.used, &a.dir).cmp(&(&b.used, &b.dir)));
        let mut total: u64 = candidates.iter().map(|c| c.bytes).sum();
        let mut freed = 0;
        for candidate in candidates {
            if total <= self.max_bytes {
                break;
            }
            if candidate.in_use {
                continue;
            }
            std::fs::remove_dir_all(&candidate.dir)?;
            total -= candidate.bytes;
            freed += candidate.bytes;
        }
        Ok(freed)
    }

    /// Every entry directory under `http/`, with its size, last use and whether it is in use.
    fn candidates(&self, in_use: &HashSet<PathBuf>) -> Result<Vec<Candidate>, IoError> {
        let mut candidates = Vec::new();
        for dir in std::fs::read_dir(&self.entries)? {
            let dir = dir?;
            if !dir.file_type()?.is_dir() {
                continue;
            }
            let dir = dir.path();
            let mut bytes = 0;
            // Q-56: the entry's directory, or any file in it.
            let mut used_now = in_use.contains(&canonical(&dir));
            for file in std::fs::read_dir(&dir)? {
                let file = file?;
                let metadata = file.metadata()?;
                if metadata.is_file() {
                    bytes += metadata.len();
                    used_now |= in_use.contains(&canonical(&file.path()));
                }
            }
            // Q-42: the latest use of a single file's entry or of a checkpoint's members.
            let used = [
                Entry::at(dir.clone()),
                Entry::member(&dir, MANIFEST),
                Entry::member(&dir, PARAMS),
            ]
            .iter()
            .filter_map(|entry| {
                entry
                    .complete(None)
                    .map(|r| r.fetched_at)
                    .or_else(|| entry.partial(None).map(|(r, _)| r.fetched_at))
            })
            .max()
            .unwrap_or_default();
            candidates.push(Candidate {
                dir,
                bytes,
                used,
                in_use: used_now,
            });
        }
        Ok(candidates)
    }
}

/// `path` canonicalized when it exists, else as given (Q-45).
fn canonical(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}
