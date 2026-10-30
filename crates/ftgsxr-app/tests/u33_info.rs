// SPDX-License-Identifier: Apache-2.0

//! U-33, the `cmd_info` clause (§1.7, openftgs §3.3's adapter reply, D-7, `cpu` lane): with
//! `--json` the reply as one compact JSON object; without it one `key: value` line per
//! field (Q-71), strings bare and other values as their JSON text (Q-77).

use ftgsxr_app::cmd_info;

#[test]
fn u33_info_json_is_the_adapter_reply() {
    assert_eq!(
        cmd_info("0.1.0", true),
        concat!(
            r#"{"name":"openftgs-xr","version":"0.1.0","query_kinds":[],"#,
            r#""rasterizer":{"name":"openftgs-xr wgpu-blend","version":"0.1.0"},"#,
            r#""render_conventions":{"alpha_max":0.999,"eps2d":0.3,"near":0.01,"far":10000000000.0,"#,
            r#""alpha_skip":0.003921569,"transmittance_stop":null,"pixel_center_offset":0.5,"#,
            r#""background":[0.0,0.0,0.0],"jacobian_clamp_margin":0.15,"#,
            r#""screen_extent_rule":"opacity_aware","screen_extent_max":3.33,"rasterize_mode":"classic"}}"#,
        )
    );
}

#[test]
fn u33_info_text_is_one_line_per_field() {
    assert_eq!(
        cmd_info("0.1.0", false),
        "name: openftgs-xr
version: 0.1.0
query_kinds: []
rasterizer: openftgs-xr wgpu-blend 0.1.0
render_conventions.alpha_max: 0.999
render_conventions.eps2d: 0.3
render_conventions.near: 0.01
render_conventions.far: 10000000000.0
render_conventions.alpha_skip: 0.003921569
render_conventions.transmittance_stop: null
render_conventions.pixel_center_offset: 0.5
render_conventions.background: [0.0,0.0,0.0]
render_conventions.jacobian_clamp_margin: 0.15
render_conventions.screen_extent_rule: opacity_aware
render_conventions.screen_extent_max: 3.33
render_conventions.rasterize_mode: classic"
    );
}
