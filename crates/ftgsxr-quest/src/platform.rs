// SPDX-License-Identifier: Apache-2.0

//! `QuestPlatform` (§2.1, §2.5): the `Platform` and `DocumentProvider` ports over the
//! JNI shim. Request ids are a counter; a pick or a permission is polled by the caller
//! (the library and `tracking` poll once per frame on the main thread); a picked file is a
//! `ModelRef::Document` and a picked tree a `ModelRef::Document` the library adds as a
//! `documents` source; `listTree`'s rows become entries (`tree`, Q-T458.3);
//! `openDocument`'s descriptor becomes a `File`. The camera's frames are M-7's.

use std::collections::HashMap;
use std::fs::File;
use std::os::fd::FromRawFd;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::Poll;

use bevy::log::{info, warn};
use ftgsxr_app::{
    AppPaths, CameraFrames, DocumentProvider, Entry, IoError, ModelRef, Permission,
    PermissionHandle, PickHandle, PickKind, Platform,
};

use crate::{shim, tree};

/// The Quest's platform (§2.1): the paths and the shim's requests in flight. Cloning
/// shares them, so the platform is its own document provider.
#[derive(Clone)]
pub struct QuestPlatform {
    inner: Arc<Inner>,
}

struct Inner {
    paths: AppPaths,
    next: AtomicU32,
    picks: Mutex<HashMap<u32, PickState>>,
    /// Each permission handle's shim requests: one per Android permission.
    permissions: Mutex<HashMap<u32, Vec<i32>>>,
}

enum PickState {
    Pending(PickKind),
    Failed(String),
}

/// The Android permissions each `Permission` stands for (§2.5's manifest): light
/// estimation's camera needs both camera permissions.
fn android_permissions(p: Permission) -> &'static [&'static str] {
    match p {
        Permission::Camera => &[
            "horizonos.permission.HEADSET_CAMERA",
            "android.permission.CAMERA",
        ],
        Permission::Scene => &["com.oculus.permission.USE_SCENE"],
        Permission::HandTracking => &["com.oculus.permission.HAND_TRACKING"],
        Permission::Anchors => &["com.oculus.permission.USE_ANCHOR_API"],
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// E-IO with `message`.
fn io_error(kind: std::io::ErrorKind, message: String) -> IoError {
    IoError::Io(std::io::Error::new(kind, message))
}

impl QuestPlatform {
    /// The platform over `paths` (Q-15's layout).
    pub(crate) fn new(paths: AppPaths) -> QuestPlatform {
        QuestPlatform {
            inner: Arc::new(Inner {
                paths,
                next: AtomicU32::new(1),
                picks: Mutex::new(HashMap::new()),
                permissions: Mutex::new(HashMap::new()),
            }),
        }
    }

    /// The next request id, kept positive for Java's `int`.
    fn next_id(&self) -> u32 {
        self.inner.next.fetch_add(1, Ordering::Relaxed) & 0x7FFF_FFFF
    }

    /// A tree's rows, parsed (a malformed line skipped).
    fn rows(tree_uri: &str) -> Result<Vec<tree::Row>, IoError> {
        let lines = shim::list_tree(tree_uri)
            .map_err(|e| io_error(std::io::ErrorKind::Other, format!("{tree_uri}: {e}")))?;
        Ok(lines.iter().filter_map(|l| tree::parse_row(l)).collect())
    }
}

impl Platform for QuestPlatform {
    fn paths(&self) -> &AppPaths {
        &self.inner.paths
    }

    fn pick(&self, kind: PickKind) -> PickHandle {
        let id = self.next_id();
        let state = match shim::pick_documents(id as i32, kind == PickKind::Folder) {
            Ok(()) => PickState::Pending(kind),
            Err(e) => PickState::Failed(format!("the document picker could not start: {e}")),
        };
        lock(&self.inner.picks).insert(id, state);
        PickHandle(id)
    }

    fn poll_pick(&self, h: PickHandle) -> Poll<Result<Vec<ModelRef>, String>> {
        let mut picks = lock(&self.inner.picks);
        let kind = match picks.get(&h.0) {
            None => return Poll::Ready(Err(format!("no pick {}", h.0))),
            Some(PickState::Failed(_)) => {
                let Some(PickState::Failed(e)) = picks.remove(&h.0) else {
                    unreachable!("invariant: the state was just seen");
                };
                return Poll::Ready(Err(e));
            }
            Some(PickState::Pending(kind)) => *kind,
        };
        match shim::poll_pick(h.0 as i32) {
            Ok(None) => Poll::Pending,
            Ok(Some(uris)) => {
                picks.remove(&h.0);
                if kind == PickKind::Folder {
                    // Q-273: the owner reads the granted tree's URI here for Dv-10.
                    for uri in &uris {
                        info!("picked tree {uri}");
                    }
                }
                Poll::Ready(Ok(uris.into_iter().map(ModelRef::Document).collect()))
            }
            Err(e) => {
                picks.remove(&h.0);
                Poll::Ready(Err(e))
            }
        }
    }

    fn request_permission(&self, p: Permission) -> PermissionHandle {
        let h = self.next_id();
        let mut ids = Vec::new();
        for name in android_permissions(p) {
            let id = self.next_id() as i32;
            match shim::request_permission(id, name) {
                Ok(()) => ids.push(id),
                Err(e) => {
                    warn!("requesting {name}: {e}");
                    // The shim's unknown request answers denied.
                    ids.push(-1);
                }
            }
        }
        lock(&self.inner.permissions).insert(h, ids);
        PermissionHandle(h)
    }

    fn poll_permission(&self, h: PermissionHandle) -> Poll<bool> {
        let mut permissions = lock(&self.inner.permissions);
        let Some(ids) = permissions.get(&h.0) else {
            return Poll::Ready(false);
        };
        let mut granted = true;
        for &id in ids {
            let answer = if id < 0 {
                0
            } else {
                shim::poll_permission(id).unwrap_or(0)
            };
            match answer {
                -1 => return Poll::Pending,
                1 => {}
                _ => granted = false,
            }
        }
        permissions.remove(&h.0);
        Poll::Ready(granted)
    }

    fn document_provider(&self) -> Option<Arc<dyn DocumentProvider>> {
        Some(Arc::new(self.clone()))
    }

    fn camera_frames(&self) -> Option<Arc<dyn CameraFrames>> {
        // M-7: the shim's camera methods.
        None
    }
}

// Q-T458.3
impl DocumentProvider for QuestPlatform {
    fn list_tree(&self, tree_uri: &str) -> Result<Vec<Entry>, IoError> {
        let rows = QuestPlatform::rows(tree_uri)?;
        Ok(tree::entries(rows, &QuestPlatform::rows))
    }

    fn open(&self, uri: &str) -> Result<File, IoError> {
        let fd = shim::open_document(uri)
            .map_err(|e| io_error(std::io::ErrorKind::Other, format!("{uri}: {e}")))?;
        if fd < 0 {
            return Err(io_error(
                std::io::ErrorKind::NotFound,
                format!("{uri}: the document could not be opened"),
            ));
        }
        // SAFETY: `openDocument` returns `ParcelFileDescriptor.detachFd()`: a descriptor
        // freshly opened for this call whose ownership passed to the caller, so nothing
        // else closes it and the `File` is its only owner.
        Ok(unsafe { File::from_raw_fd(fd) })
    }
}
