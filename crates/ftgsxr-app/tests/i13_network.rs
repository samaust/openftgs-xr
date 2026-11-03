// SPDX-License-Identifier: Apache-2.0

//! I-13's network clause (§1.15, `cpu`): an unreachable URL (a closed localhost port)
//! gives E-NET-FETCH. A load without a library entry reports its failure in the library's
//! own `LibraryFailure` message, with the code and the full text, for a fetch, read or
//! upload failure alike (Q-173); an entry's load reports in its state.

mod support;

use std::fs;
use std::net::TcpListener;

use ftgsxr_app::{EntryState, Library, LibraryCommand, LibraryFailure, LoadedModel, ModelRef, Url};
use ftgsxr_io::{ListingFlavor, SourceSpec};
use support::TestApp;

/// A localhost URL whose port was bound and closed again.
fn closed_url(path: &str) -> String {
    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    format!("http://127.0.0.1:{port}/{path}")
}

/// Runs frames until the library sends a failure, at most `max`; every one sent.
fn failures(t: &mut TestApp, max: usize) -> Vec<LibraryFailure> {
    let mut seen = Vec::new();
    for _ in 0..max {
        t.app.update();
        let msgs = t
            .app
            .world()
            .resource::<bevy::ecs::message::Messages<LibraryFailure>>();
        // A listing's failure (the scratch models directory may not exist) is not a load's.
        seen.extend(
            msgs.iter_current_update_messages()
                .filter(|f| f.reference.is_some())
                .cloned(),
        );
        if !seen.is_empty() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    seen
}

fn no_instance(t: &mut TestApp) {
    t.frames(10);
    let mut q = t.app.world_mut().query::<&LoadedModel>();
    assert_eq!(q.iter(t.app.world()).count(), 0, "no partial instance");
}

// Q-173
#[test]
fn i13_an_unreachable_url_is_e_net_fetch() {
    let mut t = TestApp::new("i13-unreachable");
    let url = Url::parse(&closed_url("m.safetensors")).unwrap();
    t.command(LibraryCommand::LoadRef(ModelRef::Http(url.clone())));
    let seen = failures(&mut t, 3000);
    assert_eq!(seen.len(), 1, "{seen:?}");
    let f = &seen[0];
    assert_eq!(f.code, "E-NET-FETCH");
    assert!(
        f.text.starts_with("E-NET-FETCH: Could not reach"),
        "{}",
        f.text
    );
    assert_eq!(f.reference, Some(ModelRef::Http(url)));
    assert_eq!(f.name, "m");
    no_instance(&mut t);
}

#[test]
fn i13_an_unreachable_entry_fails_with_e_net_fetch_and_a_load_retries() {
    let mut t = TestApp::new("i13-entry");
    let base = t.server.url_of("listing/models/");
    t.command(LibraryCommand::AddSource(SourceSpec {
        kind: "http".into(),
        location: base,
        name: "fixtures".into(),
        flavor: Some(ListingFlavor::Index),
    }));
    assert!(t.until(2000, |app| {
        !app.world().resource::<Library>().entries.is_empty()
    }));
    let closed = closed_url("probe%20p1.safetensors");
    let good = {
        let mut lib = t.app.world_mut().resource_mut::<Library>();
        let e = &mut lib.entries[0];
        std::mem::replace(&mut e.entry.location, closed)
    };
    t.command(LibraryCommand::Load {
        entry: 0,
        reduce: false,
    });
    assert!(t.until(3000, |app| matches!(
        app.world().resource::<Library>().entries[0].state,
        EntryState::Failed(_)
    )));
    let EntryState::Failed(text) = t.app.world().resource::<Library>().entries[0].state.clone()
    else {
        unreachable!()
    };
    assert!(text.starts_with("E-NET-FETCH: "), "{text}");
    no_instance(&mut t);
    t.app.world_mut().resource_mut::<Library>().entries[0]
        .entry
        .location = good;
    t.command(LibraryCommand::Load {
        entry: 0,
        reduce: false,
    });
    assert!(t.until(3000, |app| {
        app.world_mut()
            .query::<&LoadedModel>()
            .iter(app.world())
            .count()
            == 1
    }));
}

// Q-173
#[test]
fn i13_a_missing_file_reports_its_read_failure() {
    let mut t = TestApp::new("i13-missing");
    let path = t.dir.join("absent.safetensors");
    t.command(LibraryCommand::LoadRef(ModelRef::File(path.clone())));
    let seen = failures(&mut t, 600);
    assert_eq!(seen.len(), 1, "{seen:?}");
    assert_eq!(seen[0].code, "E-IO");
    assert!(seen[0].text.starts_with("E-IO: "), "{}", seen[0].text);
    assert_eq!(seen[0].reference, Some(ModelRef::File(path)));
    no_instance(&mut t);
}

// Q-173
#[test]
fn i13_a_dropped_non_model_reports_its_validation_failure() {
    let mut t = TestApp::new("i13-invalid-ref");
    let path = t.dir.join("notes.safetensors");
    fs::write(&path, "not a model\n").unwrap();
    t.command(LibraryCommand::LoadRef(ModelRef::File(path.clone())));
    let seen = failures(&mut t, 600);
    assert_eq!(seen.len(), 1, "{seen:?}");
    assert!(seen[0].code.starts_with("E-FILE"), "{:?}", seen[0]);
    assert!(seen[0].text.starts_with(&format!("{}: ", seen[0].code)));
    assert_eq!(seen[0].name, "notes");
    no_instance(&mut t);
}

// Q-173, Q-117
#[test]
#[ignore = "gpu"]
fn gpu_i13_an_entry_less_load_over_budget_reports_e_mem_budget_with_n() {
    let _layer = ftgsxr_testkit::layer_messages(); // R-16, Q-105
    use ftgsxr_app::QualityTier;
    use ftgsxr_core::memory_estimate;
    use ftgsxr_testkit::{fixtures_dir, probe};
    use support::Setup;
    let (model, _) = probe("P1");
    let n = model.len() as u64;
    let mut tier = QualityTier::by_name("desktop").unwrap();
    let full = memory_estimate(n, model.sh_degree, &tier.layout, false).static_bytes;
    tier.memory_budget_bytes = full / 2;
    let mut t = TestApp::gpu_with(
        "i13-budget-ref",
        Setup {
            tier,
            ..Setup::default()
        },
    );
    let path = fixtures_dir().join("probes/P1.safetensors");
    t.command(LibraryCommand::LoadRef(ModelRef::File(path.clone())));
    let seen = failures(&mut t, 1200);
    assert_eq!(seen.len(), 1, "{seen:?}");
    assert_eq!(seen[0].code, "E-MEM-BUDGET");
    assert!(
        seen[0].text.starts_with("E-MEM-BUDGET: ")
            && seen[0].text.ends_with(&format!(" (N = {n})")),
        "{}",
        seen[0].text
    );
    assert_eq!(seen[0].reference, Some(ModelRef::File(path)));
    no_instance(&mut t);
}
