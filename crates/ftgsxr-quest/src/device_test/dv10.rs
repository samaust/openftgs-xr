// SPDX-License-Identifier: Apache-2.0

//! Dv-10 on the device (§3.4, §1.5, D-49, Q-273). Clause 1: the probe the driver pushed as
//! `models/dv10-probe.safetensors` is listed by the library's first source, the
//! app-external models directory (Q-15); the runner prints the listed names. Clause 2,
//! given a granted tree's URI in the extra `ftgsxr-tree` (Q-273): the tree is listed
//! through the shim's `listTree`, its first model file's header is read through
//! `openDocument`, and its first model (a checkpoint through Q-61's cache copy) is added
//! as a `documents` source and loaded through the library. Without the extra the clause is
//! reported as not run, not as FAIL.

use std::io::Read;
use std::time::{Duration, Instant};

use bevy::prelude::*;
use ftgsxr_app::{EntryKind, EntryState, Library, LibraryCommand, PlatformRes, SourceSpec};

use crate::device_test::verdict;
use crate::jni_calls;

/// The test's name in the intent extra.
pub(crate) const NAME: &str = "dv10";
/// The extra holding the granted tree's URI (Q-273).
const TREE_EXTRA: &str = "ftgsxr-tree";
/// The pushed probe's stem, the library's name for it.
const PROBE: &str = "dv10-probe";
/// How long each clause may take before FAIL.
const TIMEOUT: Duration = Duration::from_secs(60);

enum Phase {
    /// Waiting for the probe in the first source's entries.
    Listing,
    /// The tree added as a source; waiting for its entry `location`.
    Added {
        location: String,
    },
    /// The entry asked to load.
    Loading {
        location: String,
    },
    Done,
}

#[derive(Resource)]
struct Dv10 {
    started: Instant,
    tree: Option<String>,
    /// Whether the runner added the tree as a source (and so removes it).
    added: bool,
    listed: String,
    phase: Phase,
}

/// Adds the test.
pub(super) fn add(app: &mut App) {
    let tree = jni_calls::intent_extra(TREE_EXTRA).filter(|t| !t.is_empty());
    app.insert_resource(Dv10 {
        started: Instant::now(),
        tree,
        added: false,
        listed: String::new(),
        phase: Phase::Listing,
    })
    .add_systems(Update, run);
}

fn fail(test: &mut Dv10, detail: &str) {
    verdict(NAME, false, detail);
    test.phase = Phase::Done;
}

fn run(
    mut test: ResMut<Dv10>,
    library: Res<Library>,
    platform: Res<PlatformRes>,
    mut commands: MessageWriter<LibraryCommand>,
) {
    if test.started.elapsed() > TIMEOUT && !matches!(test.phase, Phase::Done) {
        let detail = match &test.phase {
            Phase::Listing => format!("{PROBE} not listed after {} s", TIMEOUT.as_secs()),
            _ => format!("the tree's model not loaded after {} s", TIMEOUT.as_secs()),
        };
        fail(&mut test, &detail);
        return;
    }
    // Each arm puts the phase back unless the test is over.
    match std::mem::replace(&mut test.phase, Phase::Done) {
        Phase::Done => {}
        Phase::Listing => {
            // The models directory's source, wherever the settings put it (#285).
            let models = platform.0.paths().models.to_string_lossy().into_owned();
            let names: Vec<&str> = library
                .entries
                .iter()
                .filter(|e| {
                    library
                        .sources
                        .get(e.source)
                        .is_some_and(|(spec, _)| spec.kind == "dir" && spec.location == models)
                })
                .map(|e| e.entry.name.as_str())
                .collect();
            if !names.contains(&PROBE) {
                test.phase = Phase::Listing;
                return;
            }
            test.listed = format!("models directory lists {names:?}");
            info!("dv10: {}", test.listed);
            let Some(tree) = test.tree.clone() else {
                let detail = format!(
                    "{}; tree clause not run: no {TREE_EXTRA} extra (Q-273)",
                    test.listed
                );
                verdict(NAME, true, &detail);
                return;
            };
            match tree_clause(&platform, &tree) {
                Ok(location) => {
                    // The owner's grant may have persisted the tree as a source already
                    // (Q-273); only a missing one is added, and removed at the end.
                    test.added = !library
                        .sources
                        .iter()
                        .any(|(spec, _)| spec.location == tree);
                    if test.added {
                        commands.write(LibraryCommand::AddSource(SourceSpec {
                            kind: "documents".into(),
                            location: tree.clone(),
                            name: "dv10".into(),
                            flavor: None,
                        }));
                    }
                    test.tree = Some(tree);
                    test.started = Instant::now();
                    test.phase = Phase::Added { location };
                }
                Err(e) => verdict(NAME, false, &format!("{}; tree: {e}", test.listed)),
            }
        }
        Phase::Added { location } => {
            match library
                .entries
                .iter()
                .position(|e| e.entry.location == location)
            {
                Some(entry) => {
                    commands.write(LibraryCommand::Load {
                        entry,
                        reduce: false,
                    });
                    test.phase = Phase::Loading { location };
                }
                None => test.phase = Phase::Added { location },
            }
        }
        Phase::Loading { location } => {
            let state = library
                .entries
                .iter()
                .find(|e| e.entry.location == location)
                .map(|e| e.state.clone());
            match state {
                Some(EntryState::Loaded(_)) => {
                    let detail = format!("{}; tree's model {location} loaded", test.listed);
                    verdict(NAME, true, &detail);
                    remove_added(&test, &library, &mut commands);
                }
                Some(EntryState::Failed(e)) => {
                    verdict(NAME, false, &format!("{}; {location}: {e}", test.listed));
                    remove_added(&test, &library, &mut commands);
                }
                _ => test.phase = Phase::Loading { location },
            }
        }
    }
}

/// Removes the tree's source when the runner added it, so the settings keep none of it.
fn remove_added(test: &Dv10, library: &Library, commands: &mut MessageWriter<LibraryCommand>) {
    let Some(tree) = test.tree.as_deref().filter(|_| test.added) else {
        return;
    };
    if let Some(i) = library.sources.iter().position(|(s, _)| s.location == tree) {
        commands.write(LibraryCommand::RemoveSource(i));
    }
}

/// Lists `tree` through the provider (the shim's `listTree`), reads the first model file's
/// 8-byte prefix through `openDocument`, and returns the first model's location.
fn tree_clause(platform: &PlatformRes, tree: &str) -> Result<String, String> {
    let provider = platform
        .0
        .document_provider()
        .ok_or("no document provider")?;
    let entries = provider.list_tree(tree).map_err(|e| e.to_string())?;
    let names: Vec<(&str, EntryKind)> = entries.iter().map(|e| (e.name.as_str(), e.kind)).collect();
    info!("dv10: tree {tree} lists {names:?}");
    let is_model = |k: EntryKind| {
        matches!(
            k,
            EntryKind::Exchange | EntryKind::Ply | EntryKind::Checkpoint
        )
    };
    if let Some(file) = entries
        .iter()
        .find(|e| matches!(e.kind, EntryKind::Exchange | EntryKind::Ply))
    {
        let mut prefix = [0u8; 8];
        provider
            .open(&file.location)
            .map_err(|e| e.to_string())?
            .read_exact(&mut prefix)
            .map_err(|e| format!("{}: {e}", file.location))?;
        info!("dv10: read {}'s first 8 bytes", file.name);
    }
    entries
        .iter()
        .find(|e| is_model(e.kind))
        .map(|e| e.location.clone())
        .ok_or_else(|| format!("no model in {names:?}"))
}
