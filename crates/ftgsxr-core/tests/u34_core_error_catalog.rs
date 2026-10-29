// SPDX-License-Identifier: Apache-2.0

//! U-34, the `CoreError` part (§1.15, `cpu` lane): every variant has a §1.15 code and a
//! non-empty user text. The `IoError`, `RenderError`, `XrError`, `HeadlessError` and
//! `exit_code` clauses belong to their crates.

use ftgsxr_core::CoreError;

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

/// One value of every variant, with the code its §1.15 condition names.
fn every_variant() -> Vec<(CoreError, &'static str)> {
    vec![
        (CoreError::Format("format \"x\"".into()), "E-FILE-FORMAT"),
        (CoreError::Version("2".into()), "E-FILE-VERSION"),
        (CoreError::Keys("position".into()), "E-FILE-KEYS"),
        (
            CoreError::Shape("scale has shape [4, 2]".into()),
            "E-FILE-SHAPE",
        ),
        (CoreError::Value("duration[0] = 0".into()), "E-FILE-VALUE"),
        (
            CoreError::SingleTimestamp("velocity[0]".into()),
            "E-FILE-SINGLE",
        ),
        (CoreError::Sh("sh_degree 4".into()), "E-FILE-SH"),
        (
            CoreError::Checksum("params.safetensors".into()),
            "E-FILE-CHECKSUM",
        ),
        (
            CoreError::RequestUnit {
                request: "ms".into(),
                model: "s".into(),
            },
            "E-REQ-UNIT",
        ),
        (
            CoreError::Io(std::io::Error::from(std::io::ErrorKind::NotFound)),
            "E-IO",
        ),
        (CoreError::Json("settings.json".into()), "E-JSON"),
    ]
}

#[test]
fn u34_every_core_error_has_its_code_and_a_user_text() {
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
fn u34_every_core_error_variant_is_covered() {
    // A variant added without a row here fails to compile: the match is exhaustive.
    for (error, _) in every_variant() {
        match error {
            CoreError::Format(_)
            | CoreError::Version(_)
            | CoreError::Keys(_)
            | CoreError::Shape(_)
            | CoreError::Value(_)
            | CoreError::SingleTimestamp(_)
            | CoreError::Sh(_)
            | CoreError::Checksum(_)
            | CoreError::RequestUnit { .. }
            | CoreError::Io(_)
            | CoreError::Json(_) => {}
        }
    }
    assert_eq!(every_variant().len(), 11);
}
