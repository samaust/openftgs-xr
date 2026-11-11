// SPDX-License-Identifier: Apache-2.0

//! §3.2 R-9's fixtures (§2.7, Q-402, Q-T719.1): `probes/R9_<P>.request.json` holds, for every
//! item of probe P1 … P10, the left eye's item `<id>L` and the right eye's `<id>R` at the
//! item's τ, two eyes 64 mm apart with parallel axes and the item's intrinsics, and
//! `probes/R9_<P>.renders.safetensors` openftgs's reference renders of them.

use approx::assert_abs_diff_eq;
use ftgsxr_core::{Camera, DMat3, DVec3, RenderItem, RenderRequest, read_request};
use ftgsxr_testkit::{fixtures_dir, probe, reference_render};

const PROBES: [&str; 10] = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"];

/// The R-9 request of probe `p`, read through the verified fixtures directory.
fn r9_request(p: &str) -> RenderRequest {
    let path = fixtures_dir().join(format!("probes/R9_{p}.request.json"));
    read_request(&std::fs::read(&path).unwrap()).unwrap()
}

/// E2's camera centre −R_wᵀ t_w.
fn center(camera: &Camera) -> DVec3 {
    let m = camera.world_to_camera;
    let r = DMat3::from_mat4(m);
    -(r.transpose() * m.w_axis.truncate())
}

fn rotation(camera: &Camera) -> DMat3 {
    DMat3::from_mat4(camera.world_to_camera)
}

fn item<'a>(request: &'a RenderRequest, id: &str) -> &'a RenderItem {
    request.items.iter().find(|i| i.id == id).unwrap()
}

/// The probe camera of §3.2's preamble, at (0, −4, 0) looking along +y with world z up:
/// OpenCV rows right = +x, down = −z, forward = +y.
fn probe_rotation() -> DMat3 {
    DMat3::from_cols(
        DVec3::new(1.0, 0.0, 0.0),
        DVec3::new(0.0, 0.0, 1.0),
        DVec3::new(0.0, -1.0, 0.0),
    )
}

fn assert_eye(request: &RenderRequest, id: &str, expected: DVec3) {
    let camera = &item(request, id).camera;
    let c = center(camera);
    for k in 0..3 {
        assert_abs_diff_eq!(c[k], expected[k], epsilon = 1e-6);
    }
    let r = rotation(camera);
    for (a, b) in r
        .to_cols_array()
        .iter()
        .zip(probe_rotation().to_cols_array())
    {
        assert_abs_diff_eq!(*a, b, epsilon = 1e-12);
    }
}

// Q-402: P1's eyes' midpoint is on the optical axis 1 m from the origin, (0, −1, 0), the eyes
// ± 32 mm along the camera's x axis, world +x.
#[test]
fn r9_eyes_of_p1s_first_item_worked_by_hand() {
    let request = r9_request("P1");
    assert_eye(&request, "r0000L", DVec3::new(-0.032, -1.0, 0.0));
    assert_eye(&request, "r0000R", DVec3::new(0.032, -1.0, 0.0));
}

// Q-402: P6 keeps the probe camera's own position, (0, −4, 0), as the midpoint.
#[test]
fn r9_eyes_of_p6s_first_item_worked_by_hand() {
    let request = r9_request("P6");
    assert_eye(&request, "r0000L", DVec3::new(-0.032, -4.0, 0.0));
    assert_eye(&request, "r0000R", DVec3::new(0.032, -4.0, 0.0));
}

// Every item of the probe becomes a left and a right item at its τ, with its intrinsics and
// image size, parallel axes, 64 mm apart along the camera's x axis; the request keeps the
// probe's time unit, background and degree.
#[test]
fn r9_requests_pair_every_probe_item() {
    for p in PROBES {
        let (_, probe_request) = probe(p);
        let request = r9_request(p);
        assert_eq!(request.items.len(), 2 * probe_request.items.len(), "{p}");
        assert_eq!(
            (&request.time_unit, request.background, request.sh_degree),
            (
                &probe_request.time_unit,
                probe_request.background,
                probe_request.sh_degree
            ),
            "{p}"
        );
        for (k, source) in probe_request.items.iter().enumerate() {
            let left = &request.items[2 * k];
            let right = &request.items[2 * k + 1];
            assert_eq!(left.id, format!("{}L", source.id), "{p}");
            assert_eq!(right.id, format!("{}R", source.id), "{p}");
            for eye in [left, right] {
                assert_eq!(eye.timestamp, source.timestamp, "{p}/{}", eye.id);
                assert_eq!(eye.camera.k, source.camera.k, "{p}/{}", eye.id);
                assert_eq!(
                    (eye.camera.width, eye.camera.height),
                    (source.camera.width, source.camera.height)
                );
                let (a, b) = (rotation(&eye.camera), rotation(&source.camera));
                assert!(a.abs_diff_eq(b, 1e-12), "{p}/{}", eye.id);
            }
            let baseline = center(&right.camera) - center(&left.camera);
            let x = rotation(&source.camera).row(0);
            assert!(baseline.abs_diff_eq(0.064 * x, 1e-9), "{p}/{}", source.id);
            // P1–P5, P7–P10: the midpoint 1 m from the origin on the item's optical axis,
            // between the camera and the origin.
            if p != "P6" {
                let mid = 0.5 * (center(&left.camera) + center(&right.camera));
                let forward = rotation(&source.camera).row(2);
                assert_abs_diff_eq!(mid.length(), 1.0, epsilon = 1e-9);
                assert!(mid.abs_diff_eq(-forward, 1e-9), "{p}/{}", source.id);
            }
        }
    }
}

// The references are read through `reference_render`, which checks the request's SHA-256,
// one render per item at the item's size.
#[test]
fn r9_reference_renders_cover_every_item() {
    for p in PROBES {
        let request = r9_request(p);
        let renders = reference_render(&format!("R9_{p}"));
        let ids: Vec<&str> = renders.iter().map(|(id, _)| id.as_str()).collect();
        let mut expected: Vec<&str> = request.items.iter().map(|i| i.id.as_str()).collect();
        expected.sort();
        assert_eq!(ids, expected, "{p}");
        for (id, image) in &renders {
            let camera = &item(&request, id).camera;
            assert_eq!((image.width, image.height), (camera.width, camera.height));
        }
    }
}
