// SPDX-License-Identifier: Apache-2.0

//! U-1, the conventions clause (§1.2, openftgs D-73, `cpu` lane), on a JSON string:
//! `render_conventions` recovered with openftgs's values; an extra key ignored; `to_json`
//! compact in openftgs's key order. The file-level clauses of U-1 are the loader's.

use ftgsxr_core::{CoreError, RenderConventions};
use serde_json::Value;

/// The block exactly as openftgs 9362202 writes it: `json.dumps(render_conventions(),
/// separators=(",", ":"))` (source-log row 11).
const OPENFTGS_BLOCK: &str = concat!(
    r#"{"alpha_max":0.999,"eps2d":0.3,"near":0.01,"far":10000000000.0,"#,
    r#""alpha_skip":0.00392156862745098,"transmittance_stop":0.0001,"pixel_center_offset":0.5,"#,
    r#""background":[0.0,0.0,0.0],"jacobian_clamp_margin":0.15,"screen_extent_rule":"opacity_aware","#,
    r#""screen_extent_max":3.33,"rasterize_mode":"classic"}"#
);

/// openftgs's twelve keys in the order it writes them (§1.2).
const KEYS: [&str; 12] = [
    "alpha_max",
    "eps2d",
    "near",
    "far",
    "alpha_skip",
    "transmittance_stop",
    "pixel_center_offset",
    "background",
    "jacobian_clamp_margin",
    "screen_extent_rule",
    "screen_extent_max",
    "rasterize_mode",
];

/// openftgs's values as §1.2 lists them, in float32.
fn openftgs_values() -> RenderConventions {
    RenderConventions {
        alpha_max: 0.999,
        eps2d: 0.3,
        near: 0.01,
        far: 1e10,
        alpha_skip: 1.0 / 255.0,
        transmittance_stop: Some(1e-4),
        pixel_center_offset: 0.5,
        background: [0.0, 0.0, 0.0],
        jacobian_clamp_margin: 0.15,
        screen_extent_rule: "opacity_aware".into(),
        screen_extent_max: 3.33,
        rasterize_mode: "classic".into(),
    }
}

#[test]
fn u01_openftgs_block_recovers_openftgs_values() {
    let conv = RenderConventions::from_json(OPENFTGS_BLOCK).expect("openftgs's block parses");
    assert_eq!(conv, openftgs_values());
    assert_eq!(RenderConventions::default(), openftgs_values());
    assert!(conv.warnings().is_empty());
}

#[test]
fn u01_unknown_key_is_ignored() {
    let with_extra = OPENFTGS_BLOCK.replacen('{', r#"{"foo_extra":[1,{"x":2}],"#, 1);
    assert_eq!(
        RenderConventions::from_json(&with_extra).expect("an unknown key is ignored"),
        openftgs_values()
    );
}

#[test]
fn u01_background_recovered() {
    let white = OPENFTGS_BLOCK.replace("[0.0,0.0,0.0]", "[1.0,0.5,0.25]");
    let conv = RenderConventions::from_json(&white).unwrap();
    assert_eq!(conv.background, [1.0, 0.5, 0.25]);
}

#[test]
fn u01_to_json_is_compact_in_openftgs_key_order() {
    for conv in [openftgs_values(), {
        let mut c = openftgs_values();
        c.background = [1.0, 0.5, 0.25];
        c.transmittance_stop = None;
        c
    }] {
        let json = conv.to_json();
        assert!(
            !json.chars().any(char::is_whitespace),
            "not compact: {json}"
        );
        // The keys appear in openftgs's order, each once, and no other key.
        let positions: Vec<usize> = KEYS
            .iter()
            .map(|k| {
                let quoted = format!("\"{k}\":");
                assert_eq!(json.matches(&quoted).count(), 1, "{k} in {json}");
                json.find(&quoted).unwrap()
            })
            .collect();
        assert!(positions.windows(2).all(|w| w[0] < w[1]), "order: {json}");
        let value: Value = serde_json::from_str(&json).expect("to_json writes JSON");
        assert_eq!(value.as_object().unwrap().len(), KEYS.len());
        // Every number reads back to the same float32, and `null` stands for no stop.
        assert_eq!(RenderConventions::from_json(&json).unwrap(), conv);
        assert_eq!(
            value["transmittance_stop"].is_null(),
            conv.transmittance_stop.is_none()
        );
    }
}

#[test]
fn u01_missing_entries_take_openftgs_values() {
    let conv = RenderConventions::from_json(r#"{"background":[0.0,1.0,0.0]}"#).unwrap();
    let mut expected = openftgs_values();
    expected.background = [0.0, 1.0, 0.0];
    assert_eq!(conv, expected);
    assert_eq!(
        RenderConventions::from_json("{}").unwrap(),
        openftgs_values()
    );
}

#[test]
fn u01_honored_numbers_are_read() {
    // D-39: the numeric entries of another producer are honored, not replaced.
    let other = OPENFTGS_BLOCK.replace("\"eps2d\":0.3", "\"eps2d\":0.1");
    assert_eq!(RenderConventions::from_json(&other).unwrap().eps2d, 0.1);
}

#[test]
fn u01_bad_blocks_are_value_errors() {
    // §1.2: a block that is not an object, a background that is not three numbers in
    // [0, 1] (E-FILE-VALUE, §1.15); Q-8 for the non-object's code and typed entries.
    for bad in [
        "[1, 2, 3]",
        "not json",
        r#"{"background":[1.5,0.0,0.0]}"#,
        r#"{"background":[-0.1,0.0,0.0]}"#,
        r#"{"background":[0.0,0.0]}"#,
        r#"{"background":"black"}"#,
        r#"{"alpha_max":"0.999"}"#,
        r#"{"rasterize_mode":3}"#,
    ] {
        let err = RenderConventions::from_json(bad).expect_err(bad);
        assert!(matches!(err, CoreError::Value(_)), "{bad}: {err:?}");
        assert_eq!(err.code(), "E-FILE-VALUE");
    }
}

#[test]
fn u01_foreign_rule_and_mode_warn() {
    // E6, D-39: a rule or mode other than openftgs's is reported, one warning each.
    let mut conv = openftgs_values();
    conv.screen_extent_rule = "fixed_3sigma".into();
    assert_eq!(conv.warnings().len(), 1);
    conv.rasterize_mode = "antialiased".into();
    assert_eq!(conv.warnings().len(), 2);
}
