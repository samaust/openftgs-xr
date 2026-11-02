// SPDX-License-Identifier: Apache-2.0

//! U-34, the `XrError` part (§1.15, `cpu` lane): every variant has a §1.15 code and a
//! non-empty user text. The variant → code mapping and the texts are Q-119's.

use ftgsxr_app::Permission;
use ftgsxr_xr::XrError;

/// One value of every variant, with the code and user text Q-119 gives it.
fn every_variant() -> Vec<(XrError, &'static str, &'static str)> {
    vec![
        (
            XrError::Unsupported("XR_FB_passthrough"),
            "E-XR-UNSUPPORTED",
            "Not available on this runtime (XR_FB_passthrough)",
        ),
        (
            XrError::Permission(Permission::Camera),
            "E-XR-PERMISSION",
            "Permission Camera denied; feature unavailable",
        ),
        (
            XrError::Runtime("no active runtime".into()),
            "E-XR-RUNTIME",
            "No OpenXR runtime (no active runtime)",
        ),
        (XrError::SessionLost, "E-XR-LOST", "XR session lost"),
    ]
}

#[test]
fn u34_every_xr_error_has_its_code_and_a_user_text() {
    for (error, code, text) in every_variant() {
        assert_eq!(error.code(), code, "{error:?}");
        assert!(!error.user_text().trim().is_empty(), "{error:?}");
        assert_eq!(error.user_text(), text, "{error:?}");
    }
}

#[test]
fn u34_every_permission_names_itself_in_the_text() {
    for (p, name) in [
        (Permission::Camera, "Camera"),
        (Permission::Scene, "Scene"),
        (Permission::HandTracking, "HandTracking"),
        (Permission::Anchors, "Anchors"),
    ] {
        let e = XrError::Permission(p);
        assert_eq!(e.code(), "E-XR-PERMISSION");
        assert_eq!(
            e.user_text(),
            format!("Permission {name} denied; feature unavailable")
        );
    }
}

#[test]
fn u34_every_xr_error_variant_is_covered() {
    for (error, _, _) in every_variant() {
        match error {
            XrError::Unsupported(_)
            | XrError::Permission(_)
            | XrError::Runtime(_)
            | XrError::SessionLost => {}
        }
    }
    assert_eq!(every_variant().len(), 4);
}
