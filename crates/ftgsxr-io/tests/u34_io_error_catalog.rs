// SPDX-License-Identifier: Apache-2.0

//! U-34, the `IoError` part (§1.15, `cpu` lane): every variant has a §1.15 code and a
//! non-empty user text. The variant → code mapping is PRD #1's ("Error codes"); `Core(e)`
//! carries the inner error's code. The `CoreError` part is in `ftgsxr-core`.

use ftgsxr_core::CoreError;
use ftgsxr_io::IoError;

/// Every code of §1.15's table.
const CATALOG: &[&str] = &[
    "E-FILE-FORMAT",
    "E-FILE-VERSION",
    "E-FILE-KEYS",
    "E-FILE-SHAPE",
    "E-FILE-VALUE",
    "E-FILE-SINGLE",
    "E-FILE-SH",
    "E-FILE-CHECKSUM",
    "E-NET-CHECKSUM",
    "E-IO",
    "E-JSON",
    "E-SCENE-FORMAT",
    "E-GPU-SHADER",
    "E-REQ-UNIT",
    "E-REQ-DEGREE",
    "E-MEM-BUDGET",
    "E-NET-FETCH",
    "E-NET-LISTING",
    "E-GPU-ADAPTER",
    "E-GPU-LOST",
    "E-XR-RUNTIME",
    "E-XR-LOST",
    "E-XR-PERMISSION",
    "E-XR-UNSUPPORTED",
    "E-AUDIO",
    "E-SCENE-MISSING",
    "E-SETTINGS",
    "E-NUMERIC",
];

/// One value of every variant, with the code its §1.15 condition names; `Core` once with
/// a JSON error (a player file that is not JSON, PRD #1) and once with a format error.
fn every_variant() -> Vec<(IoError, &'static str)> {
    vec![
        (
            IoError::Fetch("http://example.invalid/a.safetensors (404)".into()),
            "E-NET-FETCH",
        ),
        (
            IoError::Listing("http://example.invalid/models/".into()),
            "E-NET-LISTING",
        ),
        (IoError::Checksum("a.safetensors".into()), "E-NET-CHECKSUM"),
        (
            IoError::Settings("settings.toml: unknown major version 2".into()),
            "E-SETTINGS",
        ),
        (
            IoError::Scene("scenes/a.ftgsxr-scene.json (missing clock)".into()),
            "E-SCENE-FORMAT",
        ),
        (
            IoError::Core(CoreError::Json("poses.json".into())),
            "E-JSON",
        ),
        (
            IoError::Core(CoreError::Format("format \"x\"".into())),
            "E-FILE-FORMAT",
        ),
        (
            IoError::Io(std::io::Error::from(std::io::ErrorKind::PermissionDenied)),
            "E-IO",
        ),
    ]
}

#[test]
fn u34_every_io_error_has_its_code_and_a_user_text() {
    for (error, code) in every_variant() {
        assert_eq!(error.code(), code, "{error:?}");
        assert!(
            CATALOG.contains(&error.code()),
            "{} is not in §1.15",
            error.code()
        );
        assert!(
            !error.user_text().trim().is_empty(),
            "{error:?} has no user text"
        );
    }
}

#[test]
fn u34_io_error_core_carries_the_inner_user_text() {
    let inner = CoreError::Keys("position".into());
    let text = inner.user_text();
    assert_eq!(IoError::Core(inner).user_text(), text);
}

#[test]
fn u34_every_io_error_variant_is_covered() {
    // A variant added without a row here fails to compile: the match is exhaustive.
    for (error, _) in every_variant() {
        match error {
            IoError::Fetch(_)
            | IoError::Listing(_)
            | IoError::Checksum(_)
            | IoError::Settings(_)
            | IoError::Scene(_)
            | IoError::Core(_)
            | IoError::Io(_) => {}
        }
    }
}
