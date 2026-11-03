// SPDX-License-Identifier: Apache-2.0

//! I-13 (§1.15), this ticket's clauses: an invalid file listed by the library gives
//! `EntryState::Failed` with its E-FILE code (`cpu`); a model over the budget fails with
//! Q-117's text, and `Load { reduce: true }` then loads it reduced (`gpu`; Q-158).

mod support;

use std::fs;

use ftgsxr_app::{EntryState, Library, LibraryCommand, LoadedModel, QualityTier};
use ftgsxr_core::memory_estimate;
use ftgsxr_testkit::{exchange_bytes, fixtures_dir, layer_messages, probe};
use support::{Setup, TestApp};

fn state(t: &TestApp, i: usize) -> Option<EntryState> {
    t.app
        .world()
        .resource::<Library>()
        .entries
        .get(i)
        .map(|e| e.state.clone())
}

#[test]
fn i13_an_invalid_file_fails_with_its_e_file_code() {
    let mut t = TestApp::new("i13-file");
    let (mut model, _) = probe("P1");
    model.position[0][0] = f32::NAN;
    let models = t.dir.join("models");
    fs::create_dir_all(&models).unwrap();
    fs::write(models.join("bad.safetensors"), exchange_bytes(&model)).unwrap();
    t.command(LibraryCommand::Refresh(0));
    assert!(
        t.until(300, |app| app.world().resource::<Library>().entries.len()
            == 1)
    );
    t.command(LibraryCommand::Load {
        entry: 0,
        reduce: false,
    });
    let ok = t.until(600, |app| {
        matches!(
            app.world().resource::<Library>().entries[0].state,
            EntryState::Failed(_)
        )
    });
    assert!(ok, "the load fails");
    let Some(EntryState::Failed(text)) = state(&t, 0) else {
        unreachable!()
    };
    assert!(text.starts_with("E-FILE-VALUE: "), "{text}");
    let mut q = t.app.world_mut().query::<&LoadedModel>();
    assert_eq!(q.iter(t.app.world()).count(), 0, "no partial instance");
}

#[test]
#[ignore = "gpu"]
fn gpu_i13_over_budget_then_reduced() {
    let _layer = layer_messages(); // R-16, Q-105
    let (model, _) = probe("P1");
    let n = model.len() as u64;
    let mut tier = QualityTier::by_name("desktop").unwrap();
    let full = memory_estimate(n, model.sh_degree, &tier.layout, false).static_bytes;
    tier.memory_budget_bytes = full / 2;
    let mut t = TestApp::gpu_with(
        "i13-budget",
        Setup {
            tier,
            ..Setup::default()
        },
    );
    let models = t.dir.join("models");
    fs::create_dir_all(&models).unwrap();
    fs::copy(
        fixtures_dir().join("probes/P1.safetensors"),
        models.join("P1.safetensors"),
    )
    .unwrap();
    t.command(LibraryCommand::Refresh(0));
    assert!(t.until(300, |app| {
        app.world()
            .resource::<Library>()
            .entries
            .first()
            .is_some_and(|e| e.info.is_some())
    }));
    t.command(LibraryCommand::Load {
        entry: 0,
        reduce: false,
    });
    let failed = |app: &mut bevy::app::App| {
        matches!(
            app.world().resource::<Library>().entries[0].state,
            EntryState::Failed(_)
        )
    };
    assert!(t.until(600, failed), "refused");
    let Some(EntryState::Failed(text)) = state(&t, 0) else {
        unreachable!()
    };
    assert!(text.starts_with("E-MEM-BUDGET: Needs "), "{text}");
    assert!(
        text.ends_with(&format!("reduce to budget or choose a tier (N = {n})")),
        "{text}"
    );
    t.command(LibraryCommand::Load {
        entry: 0,
        reduce: true,
    });
    let loaded = |app: &mut bevy::app::App| {
        matches!(
            app.world().resource::<Library>().entries[0].state,
            EntryState::Loaded(_)
        )
    };
    assert!(t.until(600, loaded), "loaded reduced: {:?}", state(&t, 0));
    let mut q = t.app.world_mut().query::<&LoadedModel>();
    let m = q.single(t.app.world()).unwrap();
    assert!(m.reduced);
    assert!(m.info.n < n);
}
