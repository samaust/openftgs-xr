// SPDX-License-Identifier: Apache-2.0

//! `ftgsxr-xr`'s integration tests, one test binary (#413, §2.7 Organisation): each
//! `tests/*.rs` file is a module of it, and the support modules are declared here once.

mod dv_support;
mod xr_support;

mod dv10_storage;
mod dv12_display;
mod dv13_sleep_wake;
mod dv1_install_launch;
mod dv2_stereo;
mod dv3_sort;
mod dv4_conformance;
mod dv6_passthrough;
mod dv7_depth;
mod dv8_hands;
mod dv9_anchors;
mod i13_xr_unsupported_toast;
mod i15_xr_session;
mod i16_xr_input;
mod i17_reference_spaces;
mod r18_contact_shadow;
mod u20_light;
mod u32_hands_to_intents;
mod u34_xr_error_catalog;
mod u37_panel_placement;
mod xr_controllers;
mod xr_display;
mod xr_hands;
mod xr_passthrough_unsupported;
mod xr_rig_binding;
