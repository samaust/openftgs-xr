// SPDX-License-Identifier: Apache-2.0

//! U-34, the `HeadlessError` part (§1.15, `cpu` lane): every variant has a §1.15 code and
//! a non-empty user text; `exit_code` maps Input and Environment to 2 and Numeric to 3.
//! Input and Environment carry their cause's code in the message's leading `E-…:` prefix
//! (Q-64), and fall back to a code of their own without one (Q-75).

use ftgsxr_app::{HeadlessError, exit_code};

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

/// One value of every variant, with the code and exit code §1.15 and §2.1 give it.
fn every_variant() -> Vec<(HeadlessError, &'static str, i32)> {
    vec![
        (
            HeadlessError::Input(
                "E-FILE-VALUE: m.safetensors: opacity has an invalid value".into(),
            ),
            "E-FILE-VALUE",
            2,
        ),
        (
            HeadlessError::Input("E-REQ-UNIT: Request time unit ms, model s".into()),
            "E-REQ-UNIT",
            2,
        ),
        (
            HeadlessError::Environment("E-GPU-ADAPTER: No Vulkan adapter found".into()),
            "E-GPU-ADAPTER",
            2,
        ),
        (
            HeadlessError::Numeric("rgb/0 holds a NaN".into()),
            "E-NUMERIC",
            3,
        ),
        // Q-75: without a §1.15 prefix, Input is E-FILE-FORMAT and Environment
        // E-GPU-ADAPTER.
        (HeadlessError::Input("no prefix".into()), "E-FILE-FORMAT", 2),
        (
            HeadlessError::Environment("E-NOT-A-CODE: something".into()),
            "E-GPU-ADAPTER",
            2,
        ),
    ]
}

#[test]
fn u34_every_headless_error_has_a_code_and_a_user_text() {
    for (e, code, _) in every_variant() {
        assert!(
            CATALOG.contains(&e.code()),
            "{e:?}: {} is not a §1.15 code",
            e.code()
        );
        assert_eq!(e.code(), code, "{e:?}");
        assert!(!e.user_text().trim().is_empty(), "{e:?} has no user text");
    }
}

#[test]
fn u34_exit_code_maps_input_and_environment_to_2_numeric_to_3() {
    for (e, _, exit) in every_variant() {
        assert_eq!(exit_code(&e), exit, "{e:?}");
    }
}

#[test]
fn u34_user_text_drops_the_code_prefix() {
    // Q-64: the prefix holds the code; the user text is the rest.
    let e = HeadlessError::Input("E-FILE-KEYS: m.safetensors: Missing key sh".into());
    assert_eq!(e.user_text(), "m.safetensors: Missing key sh");
    let e = HeadlessError::Numeric("rgb/0 holds a NaN".into());
    assert_eq!(
        e.user_text(),
        "Render produced invalid values (rgb/0 holds a NaN)"
    );
}
