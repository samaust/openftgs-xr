// SPDX-License-Identifier: Apache-2.0

//! U-34, the `RenderError` part (§1.15, `cpu` lane): every variant has a §1.15 code and a
//! non-empty user text. The variant → code mapping is Q-64's; `Core(e)` carries the inner
//! error's code and text.

use ftgsxr_core::CoreError;
use ftgsxr_render::RenderError;

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

/// One value of every variant, with the code Q-64 gives it.
fn every_variant() -> Vec<(RenderError, &'static str)> {
    vec![
        (RenderError::NoAdapter, "E-GPU-ADAPTER"),
        (RenderError::DeviceLost, "E-GPU-LOST"),
        (
            RenderError::Budget {
                needed: 3_000_000_000,
                budget: 1_610_612_736,
            },
            "E-MEM-BUDGET",
        ),
        (
            RenderError::Shader("cull.wgsl: unknown identifier".into()),
            "E-GPU-SHADER",
        ),
        (RenderError::NonFinite, "E-NUMERIC"),
        (
            RenderError::Core(CoreError::Sh("sh_degree 4".into())),
            "E-FILE-SH",
        ),
    ]
}

#[test]
fn u34_every_render_error_has_its_code_and_a_user_text() {
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
fn u34_render_error_core_keeps_the_inner_text() {
    let inner = CoreError::Keys("position".into());
    let text = inner.user_text();
    assert_eq!(RenderError::Core(inner).user_text(), text);
}

#[test]
fn u34_budget_user_text_gives_both_figures_in_mb() {
    // §1.15 E-MEM-BUDGET: "Needs X MB, budget Y MB; reduce to budget or choose a tier".
    // 3 000 000 000 bytes is 3000 MB; 1.5 GiB = 1 610 612 736 bytes is 1610 MB (MB = 10⁶ bytes).
    let e = RenderError::Budget {
        needed: 3_000_000_000,
        budget: 1_610_612_736,
    };
    assert_eq!(
        e.user_text(),
        "Needs 3000 MB, budget 1610 MB; reduce to budget or choose a tier"
    );
}

#[test]
fn u34_every_render_error_variant_is_covered() {
    // A variant added without a row here fails to compile: the match is exhaustive.
    for (error, _) in every_variant() {
        match error {
            RenderError::NoAdapter
            | RenderError::DeviceLost
            | RenderError::Budget { .. }
            | RenderError::Shader(_)
            | RenderError::NonFinite
            | RenderError::Core(_) => {}
        }
    }
    assert_eq!(every_variant().len(), 6);
}
