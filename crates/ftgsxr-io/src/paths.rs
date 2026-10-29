// SPDX-License-Identifier: Apache-2.0

//! Where the player keeps its own files (§1.11 Locations, D-23): the XDG base and user
//! directories on the desktop, the app's internal, external and cache directories on the
//! Quest.

use std::path::{Path, PathBuf};

use directories::{ProjectDirs, UserDirs};

use crate::IoError;

/// The directory name the desktop's locations end in (§1.11, D-23).
const APP_DIR: &str = "openftgs-xr"; // D-23

/// The six locations of §1.11 (D-23).
///
/// `config` holds `settings.toml` and `poses.json`, `scenes` the scene files; `cache`
/// is the cache root, which holds `http/` and `thumbnails/`, and `logs` its `logs/`;
/// `models` is the default model directory and `screenshots` where screenshots go.
/// The paths are computed only: no directory is created (Q-14).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppPaths {
    /// Settings and the pose store (§1.11).
    pub config: PathBuf,
    /// The cache root: the HTTP cache and thumbnails (§1.11).
    pub cache: PathBuf,
    /// The default model directory (§1.5, §1.11).
    pub models: PathBuf,
    /// Scene files (§1.11).
    pub scenes: PathBuf,
    /// Screenshots (§1.11).
    pub screenshots: PathBuf,
    /// Logs (§1.11).
    pub logs: PathBuf,
}

impl AppPaths {
    /// The desktop's locations: §1.11's XDG base directories through `directories`, and
    /// the screenshots under the pictures directory (Q-14). E-IO without a home
    /// directory.
    pub fn linux() -> Result<AppPaths, IoError> {
        let no_home =
            || std::io::Error::new(std::io::ErrorKind::NotFound, "no home directory found");
        let project = ProjectDirs::from_path(PathBuf::from(APP_DIR)).ok_or_else(no_home)?;
        let config = project.config_dir().to_path_buf();
        let cache = project.cache_dir().to_path_buf();
        Ok(AppPaths {
            scenes: config.join("scenes"),
            logs: cache.join("logs"),
            models: project.data_dir().join("models"),
            screenshots: pictures_dir().ok_or_else(no_home)?.join(APP_DIR),
            config,
            cache,
        })
    }

    /// The Quest's locations (§1.11) from the directories the Android entry point
    /// passes: the internal files directory, the app-external files directory
    /// (`Android/data/<application id>/files/`) when shared storage is available, and
    /// the cache directory. Without the external directory, models and screenshots stay
    /// in the internal one (Q-15).
    pub fn android(internal: &Path, external: Option<&Path>, cache: &Path) -> AppPaths {
        // Q-15
        let shared = external.unwrap_or(internal);
        AppPaths {
            config: internal.to_path_buf(),
            cache: cache.to_path_buf(),
            models: shared.join("models"),
            scenes: internal.join("scenes"),
            screenshots: shared.join("screenshots"),
            logs: cache.join("logs"),
        }
    }
}

/// The desktop's pictures directory: `$XDG_PICTURES_DIR` when absolute, else the
/// user-dirs file's entry, else `$HOME/Pictures`; `None` without a home directory.
// Q-14
fn pictures_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("XDG_PICTURES_DIR").map(PathBuf::from)
        && dir.is_absolute()
    {
        return Some(dir);
    }
    let user = UserDirs::new()?;
    Some(match user.picture_dir() {
        Some(dir) => dir.to_path_buf(),
        None => user.home_dir().join("Pictures"),
    })
}
