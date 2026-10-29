// SPDX-License-Identifier: Apache-2.0

//! U-26, `info_json` (§1.7, openftgs §3.3's adapter contract, D-7, `cpu` lane): the keys
//! and values of the reply, `transmittance_stop` null.

use ftgsxr_core::{InfoReply, RenderConventions, info_json};
use serde_json::{Value, json};

/// The reply `ftgsxr info` gives (§1.7): openftgs's conventions but for the stop (D-7).
fn reply() -> InfoReply {
    InfoReply {
        name: "openftgs-xr".to_owned(),
        version: "0.1.0".to_owned(),
        query_kinds: vec![],
        rasterizer_name: "openftgs-xr wgpu-blend".to_owned(),
        rasterizer_version: "0.1.0".to_owned(),
        render_conventions: RenderConventions {
            transmittance_stop: None,
            ..RenderConventions::default()
        },
    }
}

/// The position of `"key":` in `text`, to check the order keys are written in.
fn at(text: &str, key: &str) -> usize {
    text.find(&format!("\"{key}\":"))
        .unwrap_or_else(|| panic!("no key {key} in {text}"))
}

#[test]
fn u26_the_keys_and_values_of_the_reply() {
    let text = info_json(&reply());
    let v: Value = serde_json::from_str(&text).unwrap();
    let top: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
    let mut expected = vec![
        "name",
        "version",
        "query_kinds",
        "rasterizer",
        "render_conventions",
    ];
    // openftgs's order (its `ftgs-adapter info`).
    let positions: Vec<usize> = expected.iter().map(|k| at(&text, k)).collect();
    assert!(positions.windows(2).all(|w| w[0] < w[1]), "{text}");
    expected.sort();
    let mut top_sorted = top.clone();
    top_sorted.sort();
    assert_eq!(top_sorted, expected);

    assert_eq!(v["name"], "openftgs-xr");
    assert_eq!(v["version"], "0.1.0");
    assert_eq!(v["query_kinds"], json!([]));
    assert_eq!(
        v["rasterizer"],
        json!({"name": "openftgs-xr wgpu-blend", "version": "0.1.0"})
    );
    let rc = &v["render_conventions"];
    assert_eq!(rc.as_object().unwrap().len(), 12);
    let f = |k: &str| rc[k].as_f64().unwrap() as f32;
    assert_eq!(f("alpha_max"), 0.999);
    assert_eq!(f("eps2d"), 0.3);
    assert_eq!(f("near"), 0.01);
    assert_eq!(f("far"), 1e10);
    assert_eq!(f("alpha_skip"), 1.0 / 255.0);
    assert_eq!(f("pixel_center_offset"), 0.5);
    assert_eq!(rc["background"], json!([0.0, 0.0, 0.0]));
    assert_eq!(f("jacobian_clamp_margin"), 0.15);
    assert_eq!(rc["screen_extent_rule"], "opacity_aware");
    assert_eq!(f("screen_extent_max"), 3.33);
    assert_eq!(rc["rasterize_mode"], "classic");
}

#[test]
fn u26_transmittance_stop_is_null() {
    let v: Value = serde_json::from_str(&info_json(&reply())).unwrap();
    let rc = v["render_conventions"].as_object().unwrap();
    assert_eq!(rc.get("transmittance_stop"), Some(&Value::Null));
}

#[test]
fn u26_query_kinds_and_strings_are_written_as_given() {
    let mut r = reply();
    r.query_kinds = vec!["motion".to_owned(), "sh_color".to_owned()];
    r.version = "0.1.0 \"dev\"".to_owned();
    let v: Value = serde_json::from_str(&info_json(&r)).unwrap();
    assert_eq!(v["query_kinds"], json!(["motion", "sh_color"]));
    assert_eq!(v["version"], "0.1.0 \"dev\"");
}
