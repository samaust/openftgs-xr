// SPDX-License-Identifier: Apache-2.0

//! §2.7 Organisation, Q-1, Q-T15.3: `probe` and `reference_render` return every committed
//! probe of `tests/fixtures/probes/` by name, and `metrics` of a reference render against
//! itself is exact (E15).

use std::collections::BTreeSet;

use ftgsxr_testkit::{fixtures_dir, metrics, probe, reference_render};

/// The probe names, from the committed request files (README Layout: `NAME.request.json`).
fn committed_probes() -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(fixtures_dir().join("probes"))
        .unwrap()
        .filter_map(|e| {
            let name = e.unwrap().file_name().into_string().unwrap();
            name.strip_suffix(".request.json").map(str::to_owned)
        })
        .collect();
    names.sort();
    names
}

#[test]
fn every_committed_probe_is_listed() {
    // README Layout: P1 … P10, five R2 probes, R19 and the two merged models.
    assert_eq!(committed_probes().len(), 18, "{:?}", committed_probes());
}

#[test]
fn probe_returns_the_model_and_request_of_every_committed_probe() {
    for name in committed_probes() {
        let (model, request) = probe(&name);
        assert!(!model.position.is_empty(), "{name}: an empty model");
        assert!(!request.items.is_empty(), "{name}: an empty request");
        let bytes =
            std::fs::read(fixtures_dir().join(format!("probes/{name}.request.json"))).unwrap();
        assert_eq!(request.sha256, ftgsxr_core::sha256(&bytes), "{name}");
    }
}

#[test]
fn reference_render_returns_one_render_per_request_item() {
    for name in committed_probes() {
        let (_, request) = probe(&name);
        let renders = reference_render(&name);
        let ids: BTreeSet<&str> = renders.iter().map(|(id, _)| id.as_str()).collect();
        let items: BTreeSet<&str> = request.items.iter().map(|i| i.id.as_str()).collect();
        assert_eq!(ids, items, "{name}");
        for (id, image) in &renders {
            let item = request.items.iter().find(|i| &i.id == id).unwrap();
            assert_eq!(
                (image.width, image.height),
                (item.camera.width, item.camera.height),
                "{name}/{id}"
            );
        }
    }
}

#[test]
fn metrics_of_a_reference_render_against_itself_is_exact() {
    for name in committed_probes() {
        for (id, image) in reference_render(&name) {
            let m = metrics(&image, &image);
            assert_eq!(m.psnr, f64::INFINITY, "{name}/{id}");
            assert_eq!((m.share, m.alpha), (0.0, 0.0), "{name}/{id}");
        }
    }
}

#[test]
#[should_panic(expected = "probes/no_such_probe")]
fn probe_names_the_missing_fixture() {
    probe("no_such_probe");
}
