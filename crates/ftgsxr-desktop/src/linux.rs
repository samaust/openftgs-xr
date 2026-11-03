// SPDX-License-Identifier: Apache-2.0

//! `LinuxPlatform` (§2.1 `ftgsxr_desktop`, §2.5 Desktop): rfd's dialogs on a helper
//! thread, polled by `poll_pick`; every permission granted; no documents; no camera; the
//! XDG locations of §1.11 (`AppPaths::linux`).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::task::Poll;

use ftgsxr_app::{
    AppPaths, CameraFrames, DocumentProvider, IoError, ModelRef, Permission, PermissionHandle,
    PickHandle, PickKind, Platform,
};

/// A pick's answer: `None` while its dialog is open.
type Answer = Option<Result<Vec<ModelRef>, String>>;

/// The desktop's platform (§2.5).
pub struct LinuxPlatform {
    paths: AppPaths,
    next: AtomicU32,
    picks: Arc<Mutex<HashMap<u32, Answer>>>,
}

impl LinuxPlatform {
    /// The platform with §1.11's XDG locations; E-IO without a home directory (Q-14).
    // Q-T277.5
    pub fn new() -> Result<LinuxPlatform, IoError> {
        Ok(Self::with_paths(AppPaths::linux()?))
    }

    /// The platform with the given locations.
    fn with_paths(paths: AppPaths) -> LinuxPlatform {
        LinuxPlatform {
            paths,
            next: AtomicU32::new(0),
            picks: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn handle(&self) -> u32 {
        self.next.fetch_add(1, Ordering::Relaxed)
    }
}

/// The picked paths as models: each file, or the folder (Q-160: a dialog closed or one
/// that could not open gives no path, so an empty pick).
fn picked(kind: PickKind) -> Vec<ModelRef> {
    let dialog = rfd::FileDialog::new();
    let paths: Vec<PathBuf> = match kind {
        PickKind::Files => dialog.pick_files().unwrap_or_default(),
        PickKind::Folder => dialog.pick_folder().into_iter().collect(),
    };
    paths.into_iter().map(ModelRef::File).collect()
}

impl Platform for LinuxPlatform {
    fn paths(&self) -> &AppPaths {
        &self.paths
    }

    fn pick(&self, kind: PickKind) -> PickHandle {
        let h = self.handle();
        self.picks
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(h, None);
        let picks = Arc::clone(&self.picks);
        let spawned = std::thread::Builder::new()
            .name("ftgsxr-pick".into())
            .spawn(move || {
                let answer = Ok(picked(kind));
                picks
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert(h, Some(answer));
            });
        if let Err(e) = spawned {
            self.picks.lock().unwrap_or_else(|e| e.into_inner()).insert(
                h,
                Some(Err(format!("the file dialog could not start: {e}"))),
            );
        }
        PickHandle(h)
    }

    fn poll_pick(&self, h: PickHandle) -> Poll<Result<Vec<ModelRef>, String>> {
        let mut picks = self.picks.lock().unwrap_or_else(|e| e.into_inner());
        match picks.get(&h.0) {
            Some(None) => Poll::Pending,
            Some(Some(_)) => Poll::Ready(
                picks
                    .remove(&h.0)
                    .flatten()
                    .expect("invariant: the answer was just seen"),
            ),
            None => Poll::Ready(Err(format!("no pick {}", h.0))),
        }
    }

    fn request_permission(&self, _: Permission) -> PermissionHandle {
        PermissionHandle(self.handle())
    }

    fn poll_permission(&self, _: PermissionHandle) -> Poll<bool> {
        Poll::Ready(true)
    }

    fn document_provider(&self) -> Option<Arc<dyn DocumentProvider>> {
        None
    }

    fn camera_frames(&self) -> Option<Arc<dyn CameraFrames>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn platform() -> LinuxPlatform {
        let root = PathBuf::from("/nonexistent/ftgsxr");
        LinuxPlatform::with_paths(AppPaths::android(&root, None, &root))
    }

    #[test]
    fn every_permission_is_granted() {
        let p = platform();
        for perm in [
            Permission::Camera,
            Permission::Scene,
            Permission::HandTracking,
            Permission::Anchors,
        ] {
            let h = p.request_permission(perm);
            assert_eq!(p.poll_permission(h), Poll::Ready(true));
        }
    }

    #[test]
    fn no_documents_and_no_camera() {
        let p = platform();
        assert!(p.document_provider().is_none());
        assert!(p.camera_frames().is_none());
    }

    #[test]
    fn a_pick_is_pending_until_answered_then_read_once() {
        let p = platform();
        p.picks.lock().unwrap().insert(7, None);
        assert_eq!(p.poll_pick(PickHandle(7)), Poll::Pending);
        let file = ModelRef::File("/m.safetensors".into());
        p.picks
            .lock()
            .unwrap()
            .insert(7, Some(Ok(vec![file.clone()])));
        assert_eq!(p.poll_pick(PickHandle(7)), Poll::Ready(Ok(vec![file])));
        assert!(matches!(p.poll_pick(PickHandle(7)), Poll::Ready(Err(_))));
    }
}
