// SPDX-License-Identifier: Apache-2.0

//! R-9 on the device, the app's entry for the Quest runner (§3.2 R-9, Q-414):
//! `cmd_render_stereo` reads a request whose items come in left and right pairs (R-9's
//! `R9_<P>` requests, Q-415), renders each pair in the given sort mode and writes an
//! `ftgs-renders` file with the request's SHA-256, as `cmd_render` does. A request whose
//! items do not pair is refused before the GPU with E-FILE-VALUE, exit 2, and no file
//! (`cpu` lane, Q-T728.1); the end to end write is the `gpu` lane's.

use std::fs;
use std::path::{Path, PathBuf};

use ftgsxr_app::{HeadlessError, QualityTier, SortMode, cmd_render_stereo, exit_code};
use ftgsxr_core::{read_renders, read_request};
use ftgsxr_testkit::{fixtures_dir, layer_messages, scratch_dir};
use serde_json::Value;

const VERSION: &str = "0.1.0-test";

fn probes(name: &str) -> PathBuf {
    fixtures_dir().join("probes").join(name)
}

fn scratch(name: &str) -> PathBuf {
    scratch_dir(env!("CARGO_TARGET_TMPDIR"), crate::test_file_stem(), name)
}

/// `R9_P1`'s request with its items changed by `edit`, written into `dir`.
fn altered_request(dir: &Path, edit: impl FnOnce(&mut Vec<Value>)) -> PathBuf {
    let text = fs::read_to_string(probes("R9_P1.request.json")).unwrap();
    let mut request: Value = serde_json::from_str(&text).unwrap();
    edit(request["items"].as_array_mut().unwrap());
    let path = dir.join("request.json");
    fs::write(&path, serde_json::to_string(&request).unwrap()).unwrap();
    path
}

/// `cmd_render_stereo` on P1 and `request` refuses with `Input`, E-FILE-VALUE and exit 2,
/// leaves no file, and names the items.
fn refused(request: &Path, dir: &Path) -> HeadlessError {
    let out = dir.join("out.safetensors");
    let e = cmd_render_stereo(
        &probes("P1.safetensors"),
        request,
        &out,
        QualityTier::conformance(),
        SortMode::SharedCenter,
        VERSION,
    )
    .expect_err("refused");
    assert!(matches!(e, HeadlessError::Input(_)), "{e:?}");
    assert_eq!(e.code(), "E-FILE-VALUE", "{e}");
    assert_eq!(exit_code(&e), 2);
    assert!(e.user_text().contains("items = "), "{e}");
    assert!(!out.exists(), "no file is written");
    e
}

#[test]
fn cmd_render_stereo_refuses_an_odd_item_count() {
    // Q-T728.1
    let dir = scratch("odd");
    let request = altered_request(&dir, |items| {
        items.pop();
    });
    refused(&request, &dir);
}

/// A change to a request's items.
type Edit = fn(&mut Vec<Value>);

#[test]
fn cmd_render_stereo_refuses_ids_that_do_not_pair() {
    // Q-T728.1: the right eye first; two different stems; a left id without its letter.
    let cases: [(&str, Edit); 3] = [
        ("swapped", |items| items.swap(0, 1)),
        ("stems", |items| items[1]["id"] = "r9999R".into()),
        ("letter", |items| items[0]["id"] = "r0000".into()),
    ];
    for (name, edit) in cases {
        let dir = scratch(name);
        let request = altered_request(&dir, edit);
        refused(&request, &dir);
    }
}

#[test]
fn cmd_render_stereo_refuses_a_pair_at_two_times() {
    // Q-T728.1: `render_stereo` renders both eyes at one τ.
    let dir = scratch("times");
    let request = altered_request(&dir, |items| {
        let t = items[0]["timestamp"].as_f64().unwrap();
        items[1]["timestamp"] = (t + 0.1).into();
    });
    refused(&request, &dir);
}

#[test]
fn cmd_render_stereo_keeps_cmd_renders_request_checks() {
    // Q-414 "as `cmd_render` does": a request degree above P1's 0 is E-REQ-DEGREE.
    let dir = scratch("degree");
    let text = fs::read_to_string(probes("R9_P1.request.json")).unwrap();
    let mut request: Value = serde_json::from_str(&text).unwrap();
    request["sh_degree"] = 1.into();
    let path = dir.join("request.json");
    fs::write(&path, serde_json::to_string(&request).unwrap()).unwrap();
    let out = dir.join("out.safetensors");
    let e = cmd_render_stereo(
        &probes("P1.safetensors"),
        &path,
        &out,
        QualityTier::conformance(),
        SortMode::PerView,
        VERSION,
    )
    .expect_err("refused");
    assert_eq!(e.code(), "E-REQ-DEGREE", "{e}");
    assert!(!out.exists(), "no file is written");
}

/// Both modes write one image per item, in the request's pairs, with the request's
/// SHA-256 and each eye's own size; per view and shared-center draw P1's primitive.
#[test]
#[ignore = "gpu"]
fn gpu_cmd_render_stereo_writes_both_eyes_of_every_pair() {
    let _layer = layer_messages(); // R-16, Q-105
    let request_path = probes("R9_P1.request.json");
    let request = read_request(&fs::read(&request_path).unwrap()).unwrap();
    for (name, sort) in [
        ("per_view", SortMode::PerView),
        ("shared", SortMode::SharedCenter),
    ] {
        let dir = scratch(name);
        let out = dir.join("R9_P1.renders.safetensors");
        cmd_render_stereo(
            &probes("P1.safetensors"),
            &request_path,
            &out,
            QualityTier::conformance(),
            sort,
            VERSION,
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"));
        let renders = read_renders(&fs::read(&out).unwrap(), Some(request.sha256))
            .expect("a ftgs-renders file with the request's SHA-256");
        assert_eq!(renders.len(), request.items.len(), "{name}");
        for item in &request.items {
            let (_, image) = renders
                .iter()
                .find(|(id, _)| *id == item.id)
                .unwrap_or_else(|| panic!("{name}: no render for {}", item.id));
            assert_eq!(
                (image.width, image.height),
                (item.camera.width, item.camera.height)
            );
        }
        let drawn = renders
            .iter()
            .filter(|(_, image)| image.alpha.iter().any(|&a| a > 0.1))
            .count();
        assert!(drawn > 0, "{name}: P1's primitive is drawn");
    }
}
