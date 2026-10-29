// SPDX-License-Identifier: Apache-2.0

//! openftgs-xr foundation: paths, settings, scene and pose files, listings, HTTP and
//! model sources (§1.5, §1.11, §2.1). M-1 fills its modules.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod error;
pub mod http;
pub mod listing;
pub mod paths;
pub mod settings;
pub mod sources;

pub use error::IoError;
pub use http::{HttpClient, Progress, TlsRoots};
pub use listing::{
    Entry, EntryKind, ListingFlavor, classify_name, list_dir, parse_html_listing,
    parse_index_manifest, parse_nginx_json,
};
pub use paths::AppPaths;
pub use settings::{
    AudioSettings, CacheSettings, DesktopSettings, DisplaySettings, EnvironmentSettings,
    GeneralSettings, LocomotionSettings, ManualLighting, OverlaySettings, PlaybackSettings,
    Settings, SoundSwitches, XrSettings, load_settings, save_settings,
};
pub use sources::{
    DocumentProvider, DocumentSource, HttpIndex, LocalDir, ModelBytes, ModelRef, ModelSource,
    SourceSpec, open_ref, open_source,
};
