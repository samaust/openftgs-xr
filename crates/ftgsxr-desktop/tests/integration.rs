// SPDX-License-Identifier: Apache-2.0

//! `ftgsxr-desktop`'s integration tests, one test binary (#413, §2.7 Organisation): each
//! `tests/*.rs` file is a module of it, and the support modules are declared here once.

use std::panic::Location;
use std::path::Path;

/// The stem of the `tests/*.rs` file that calls it, `u31_sources` for
/// `tests/u31_sources.rs`: through `#[track_caller]` helpers, the first caller that isn't
/// one. Each file passes it to `scratch_dir` as `binary`, so that it keeps a scratch
/// folder of its own (Q-62, #413), and names its tests by it in a child run.
#[track_caller]
fn test_file_stem() -> &'static str {
    let file = Path::new(Location::caller().file());
    assert!(
        file.parent().and_then(Path::file_name) == Some("tests".as_ref()),
        "{} is not a tests/*.rs file (a support helper without #[track_caller]?)",
        file.display()
    );
    file.file_stem()
        .and_then(|s| s.to_str())
        .expect("a test file's stem is UTF-8")
}

mod common;
mod winit_support;

mod cli;
mod discard_undrawn;
mod play_cli;
mod play_exit;
mod present_undrawn;
mod window_resize;
