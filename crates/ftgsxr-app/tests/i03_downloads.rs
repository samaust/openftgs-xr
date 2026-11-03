// SPDX-License-Identifier: Apache-2.0

//! I-3 (§1.5 Download cache, `cpu`): a fetch shows its progress in
//! `EntryState::Fetching`, a cancel stops it within one chunk, a load after a cut
//! resumes with `Range`, and a checksum mismatch is `EntryState::Failed` with
//! E-NET-CHECKSUM. Also the index entry's `audio` (§1.10) and `scene` manifest (§1.7),
//! and the cache's pruning after a load (Q-4, Q-45, Q-56). Progress and cancel are seen
//! through a test-local server that holds its body after a first part.

mod support;

use std::fs;

use bevy::prelude::*;
use ftgsxr_app::{
    EntryState, Library, LibraryCommand, LoadedModel, ModelRef, Settings, SettingsRes,
};
use ftgsxr_io::{EntryKind, ListingFlavor, Progress, SourceSpec};
use support::{Setup, TestApp};

const PROBE: &str = "/listing/models/probe%20p1.safetensors";
const PROBE_SIZE: u64 = 1092;
/// `HttpClient`'s chunk between two cancel checks (U-28's "one chunk").
const CHUNK: u64 = 64 << 10;

/// Adds the fixture server's index as source 1 and waits for its entries; the probe's
/// index.
fn http_source(t: &mut TestApp) -> usize {
    let base = t.server.url_of("listing/models/");
    t.command(LibraryCommand::AddSource(SourceSpec {
        kind: "http".into(),
        location: base,
        name: "fixtures".into(),
        flavor: Some(ListingFlavor::Index),
    }));
    let ok = t.until(2000, |app| {
        app.world()
            .resource::<Library>()
            .entries
            .iter()
            .any(|e| e.source == 1 && e.entry.kind == EntryKind::Exchange && e.info.is_some())
    });
    // The header read is done: a cut armed later falls on the fetch.
    assert!(ok, "the index lists and the probe's header is read");
    entry_of(t, EntryKind::Exchange)
}

fn entry_of(t: &TestApp, kind: EntryKind) -> usize {
    t.app
        .world()
        .resource::<Library>()
        .entries
        .iter()
        .position(|e| e.source == 1 && e.entry.kind == kind)
        .unwrap()
}

fn state(t: &TestApp, i: usize) -> EntryState {
    t.app.world().resource::<Library>().entries[i].state.clone()
}

fn load(t: &mut TestApp, entry: usize) {
    t.command(LibraryCommand::Load {
        entry,
        reduce: false,
    });
}

fn instances(t: &mut TestApp) -> Vec<LoadedModel> {
    let mut q = t.app.world_mut().query::<&LoadedModel>();
    q.iter(t.app.world()).cloned().collect()
}

/// Waits until the entry leaves `Fetching`, recording every state seen.
fn fetched(t: &mut TestApp, i: usize) -> Vec<EntryState> {
    let mut seen = Vec::new();
    t.until(3000, |app| {
        let s = app.world().resource::<Library>().entries[i].state.clone();
        let done = !matches!(s, EntryState::Fetching(_));
        seen.push(s);
        done
    });
    seen
}

/// The bytes the held server sends of the model before it waits.
const FIRST: usize = 500;

/// A one-model HTTP server that sends the first `FIRST` bytes of its model's body and
/// holds the rest until released, so a fetch's progress and a cancel are seen between two
/// frames (`serve_fixtures` sends each body whole). It serves `index.json` listing the
/// model `m` (`kind`, no digest), byte ranges of the model whole (header reads), and 404
/// for anything else, a `HEAD` included (so the ETag check before a header read, Q-230,
/// finds none and never takes the held body).
struct HeldServer {
    base: String,
    release: std::sync::mpsc::Sender<()>,
}

impl HeldServer {
    fn start(body: Vec<u8>, kind: &'static str) -> HeldServer {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let (release, held) = std::sync::mpsc::channel::<()>();
        let file = if kind == "ply" {
            "m.ply"
        } else {
            "m.safetensors"
        };
        let held = std::sync::Arc::new(std::sync::Mutex::new(held));
        let body = std::sync::Arc::new(body);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { return };
                let (held, body) = (held.clone(), body.clone());
                std::thread::spawn(move || serve(stream, &body, kind, file, &held));
            }
        });
        HeldServer {
            base: format!("http://127.0.0.1:{port}/m/"),
            release,
        }
    }

    fn release(&self) {
        let _ = self.release.send(());
    }
}

/// Answers the requests of one connection (`HeldServer`).
fn serve(
    mut stream: std::net::TcpStream,
    body: &[u8],
    kind: &str,
    file: &str,
    held: &std::sync::Mutex<std::sync::mpsc::Receiver<()>>,
) {
    use std::io::{BufRead, BufReader, Write};
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 {
            return;
        }
        let path = line.split(' ').nth(1).unwrap_or("").to_owned();
        let is_head = line.starts_with("HEAD ");
        let mut range = None;
        loop {
            let mut h = String::new();
            if reader.read_line(&mut h).unwrap_or(0) == 0 || h.trim().is_empty() {
                break;
            }
            if let Some(v) = h.to_ascii_lowercase().strip_prefix("range: bytes=") {
                let (a, b) = v.trim().split_once('-').unwrap();
                let a: usize = a.parse().unwrap();
                let b = b
                    .parse::<usize>()
                    .map_or(body.len() - 1, |b| b.min(body.len() - 1));
                range = Some((a, b));
            }
        }
        let head = |status: &str, len: usize, extra: &str| {
            format!("HTTP/1.1 {status}\r\nContent-Length: {len}\r\n{extra}\r\n")
        };
        if is_head {
            let _ = write!(stream, "{}", head("404 Not Found", 0, ""));
        } else if path == "/m/index.json" {
            let index = format!(
                r#"{{"format":"ftgsxr-index","version":1,"entries":[{{"name":"m","path":"{file}","kind":"{kind}","size":{}}}]}}"#,
                body.len()
            );
            let _ = write!(stream, "{}{index}", head("200 OK", index.len(), ""));
        } else if path == format!("/m/{file}") {
            if let Some((a, b)) = range {
                let range = format!("Content-Range: bytes {a}-{b}/{}\r\n", body.len());
                let _ = write!(stream, "{}", head("206 Partial Content", b + 1 - a, &range));
                let _ = stream.write_all(&body[a..=b]);
            } else {
                let _ = write!(stream, "{}", head("200 OK", body.len(), ""));
                let _ = stream.write_all(&body[..FIRST]);
                let _ = stream.flush();
                let _ = held.lock().unwrap().recv();
                let _ = stream.write_all(&body[FIRST..]);
            }
        } else {
            let _ = write!(stream, "{}", head("404 Not Found", 0, ""));
        }
        let _ = stream.flush();
    }
}

/// Adds the held server as source 1 and starts loading its model; waits for the
/// fetch's progress at the bytes sent.
fn held_fetch(t: &mut TestApp, server: &HeldServer, size: u64) -> usize {
    t.command(LibraryCommand::AddSource(SourceSpec {
        kind: "http".into(),
        location: server.base.clone(),
        name: "held".into(),
        flavor: Some(ListingFlavor::Index),
    }));
    assert!(t.until(2000, |app| {
        app.world()
            .resource::<Library>()
            .entries
            .iter()
            .any(|e| e.source == 1)
    }));
    let i = t
        .app
        .world()
        .resource::<Library>()
        .entries
        .iter()
        .position(|e| e.source == 1)
        .unwrap();
    load(t, i);
    let at = EntryState::Fetching(Progress {
        done: FIRST as u64,
        total: Some(size),
    });
    let ok = t.until(3000, |app| {
        app.world().resource::<Library>().entries[i].state == at
    });
    assert!(ok, "the fetch's progress at the bytes sent");
    i
}

#[test]
// Needs bevy's `multi_threaded` task pools (a dev-dependency feature, #291): a fetch then
// runs beside the frames, so the held body holds only the fetch.
fn i03_progress_reaches_the_entry() {
    let body = fs::read(ftgsxr_testkit::fixtures_dir().join("probes/P1.safetensors")).unwrap();
    let size = body.len() as u64;
    let server = HeldServer::start(body, "ftgs-if");
    let mut t = TestApp::new("i03-progress");
    let i = held_fetch(&mut t, &server, size);
    // Held: the progress stays.
    t.frames(5);
    assert!(matches!(state(&t, i), EntryState::Fetching(p) if p.done == FIRST as u64));
    server.release();
    let seen = fetched(&mut t, i);
    for s in &seen {
        if let EntryState::Fetching(p) = s {
            assert!(p.done >= FIRST as u64 && p.done <= size, "{seen:?}");
        }
    }
    assert!(
        matches!(
            seen.last(),
            Some(EntryState::Loading | EntryState::Loaded(_))
        ),
        "{seen:?}"
    );
    assert!(t.until(600, |app| {
        app.world_mut()
            .query::<&LoadedModel>()
            .iter(app.world())
            .count()
            == 1
    }));
}

#[test]
// Needs bevy's `multi_threaded` task pools (a dev-dependency feature, #291): a fetch then
// runs beside the frames, so the held body holds only the fetch.
fn i03_cancel_stops_within_one_chunk() {
    let size = 4 * CHUNK;
    let server = HeldServer::start(vec![0; size as usize], "ply");
    let mut t = TestApp::new("i03-cancel");
    let i = held_fetch(&mut t, &server, size);
    t.command(LibraryCommand::Cancel(i));
    t.frames(2);
    server.release();
    let seen = fetched(&mut t, i);
    assert_eq!(seen.last(), Some(&EntryState::Listed), "{seen:?}");
    t.frames(20);
    assert!(instances(&mut t).is_empty(), "no instance");
    // The partial file the cache keeps (Q-43): the bytes before the cancel and at most
    // one chunk after it.
    let kept: u64 = fs::read_dir(t.dir.join("cache/http"))
        .unwrap()
        .flat_map(|d| fs::read_dir(d.unwrap().path()).unwrap())
        .map(|f| f.unwrap())
        .filter(|f| !f.file_name().to_string_lossy().ends_with(".json"))
        .map(|f| f.metadata().unwrap().len())
        .sum();
    assert!(
        kept >= FIRST as u64 && kept <= FIRST as u64 + CHUNK,
        "{kept} bytes kept"
    );
}

#[test]
fn i03_a_cut_download_resumes() {
    let mut t = TestApp::new("i03-resume");
    let i = http_source(&mut t);
    t.server.truncate_next(500);
    load(&mut t, i);
    let seen = fetched(&mut t, i);
    let Some(EntryState::Failed(text)) = seen.last() else {
        panic!("the cut download fails: {seen:?}");
    };
    assert!(text.starts_with("E-NET-FETCH: "), "{text}");
    let before = t.server.requests().len();
    load(&mut t, i);
    let seen = fetched(&mut t, i);
    assert!(
        matches!(
            seen.last(),
            Some(EntryState::Loading | EntryState::Loaded(_))
        ),
        "{seen:?}"
    );
    let resumed = t.server.requests()[before..]
        .iter()
        .find(|r| r.path == PROBE && r.method == "GET")
        .cloned()
        .expect("the second GET");
    assert_eq!(resumed.header("Range"), Some("bytes=500-"), "{resumed:?}");
    assert_eq!(resumed.status, 206);
    assert_eq!(resumed.body_bytes, PROBE_SIZE - 500);
    assert!(t.until(600, |app| {
        app.world_mut()
            .query::<&LoadedModel>()
            .iter(app.world())
            .count()
            == 1
    }));
}

#[test]
fn i03_a_checksum_mismatch_is_e_net_checksum_and_a_refresh_retries() {
    let mut t = TestApp::new("i03-checksum");
    let i = http_source(&mut t);
    t.app.world_mut().resource_mut::<Library>().entries[i]
        .entry
        .sha256 = Some("00".repeat(32));
    load(&mut t, i);
    let seen = fetched(&mut t, i);
    let Some(EntryState::Failed(text)) = seen.last() else {
        panic!("the mismatch fails: {seen:?}");
    };
    assert!(text.starts_with("E-NET-CHECKSUM: "), "{text}");
    t.frames(20);
    assert!(instances(&mut t).is_empty(), "no instance");
    // A refresh relists: the index's digest returns, and a new load succeeds.
    t.command(LibraryCommand::Refresh(1));
    assert!(t.until(2000, |app| {
        app.world().resource::<Library>().entries[i]
            .entry
            .sha256
            .as_deref()
            != Some("00".repeat(32).as_str())
    }));
    load(&mut t, i);
    let seen = fetched(&mut t, i);
    assert!(
        matches!(
            seen.last(),
            Some(EntryState::Loading | EntryState::Loaded(_))
        ),
        "{seen:?}"
    );
}

// §1.10: the index entry's soundtrack is downloaded with the model and given to the
// instance as its cached file. Any served file stands in for the audio: the library does
// not decode it.
#[test]
fn i03_the_index_entry_audio_is_downloaded_with_the_model() {
    // A cache limit of one byte: the prune after the load keeps both files in use.
    let mut settings = Settings::default();
    settings.cache.max_bytes = 1;
    let mut t = TestApp::with("i03-audio", Setup::default(), move |app: &mut App| {
        app.insert_resource(SettingsRes(settings));
    });
    let i = http_source(&mut t);
    let url = t.server.url_of("listing/models/notes.txt");
    t.app.world_mut().resource_mut::<Library>().entries[i]
        .entry
        .audio = Some(url);
    load(&mut t, i);
    assert!(t.until(3000, |app| {
        app.world_mut()
            .query::<&LoadedModel>()
            .iter(app.world())
            .count()
            == 1
    }));
    let m = instances(&mut t).remove(0);
    assert!(matches!(m.source, ModelRef::Http(_)));
    let Some(ModelRef::File(path)) = m.audio else {
        panic!("a local soundtrack: {:?}", m.audio);
    };
    assert!(
        path.starts_with(t.dir.join("cache/http")),
        "{}",
        path.display()
    );
    let served = fs::read(ftgsxr_testkit::fixtures_dir().join("listing/models/notes.txt")).unwrap();
    assert_eq!(fs::read(&path).unwrap(), served);
    t.frames(30);
    assert_eq!(
        cached(&t.dir),
        2,
        "the model's and the soundtrack's entries stay"
    );
    assert!(path.is_file());
}

/// The cache entry directories under `<cache>/http/`.
fn cached(dir: &std::path::Path) -> usize {
    fs::read_dir(dir.join("cache/http"))
        .map(|d| {
            d.filter_map(Result::ok)
                .filter(|e| e.path().is_dir())
                .count()
        })
        .unwrap_or(0)
}

// Q-4, Q-45, Q-56
#[test]
fn i03_the_cache_is_pruned_after_a_load_never_a_loaded_models_file() {
    let mut settings = Settings::default();
    settings.cache.max_bytes = 1;
    let mut t = TestApp::with("i03-prune", Setup::default(), move |app: &mut App| {
        app.insert_resource(SettingsRes(settings));
    });
    let probe = http_source(&mut t);
    let ply = entry_of(&t, EntryKind::Ply);
    load(&mut t, probe);
    assert!(t.until(3000, |app| {
        app.world_mut()
            .query::<&LoadedModel>()
            .iter(app.world())
            .count()
            == 1
    }));
    t.frames(20);
    assert_eq!(cached(&t.dir), 1, "the loaded probe's file stays");
    t.command(LibraryCommand::UnloadAll);
    t.frames(2);
    load(&mut t, ply);
    assert!(t.until(3000, |app| {
        app.world_mut()
            .query::<&LoadedModel>()
            .iter(app.world())
            .count()
            == 1
    }));
    let dir = t.dir.clone();
    let ok = t.until(600, |_| cached(&dir) == 1);
    assert!(
        ok,
        "the unloaded probe's entry is evicted, the loaded PLY's kept: {:?}",
        fs::read_dir(t.dir.join("cache/http"))
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect::<Vec<_>>()
    );
    let m = instances(&mut t).remove(0);
    assert_eq!(m.name, "snapshot");
}

/// The cache entry directories holding a file with `bytes`.
fn cached_with(dir: &std::path::Path, bytes: &[u8]) -> usize {
    fs::read_dir(dir.join("cache/http"))
        .map(|d| {
            d.filter_map(Result::ok)
                .filter(|e| {
                    fs::read_dir(e.path()).is_ok_and(|files| {
                        files
                            .filter_map(Result::ok)
                            .any(|f| fs::read(f.path()).is_ok_and(|b| b == bytes))
                    })
                })
                .count()
        })
        .unwrap_or(0)
}

// §1.7 Training cameras and requests ("or an index entry names one"), §1.11 `index.json`
// (`scene`), Q-4: the index entry's `scene` manifest is downloaded with the model and stays
// in the cache while the instance is loaded; once it is unloaded, a later prune evicts it.
#[test]
fn i03_the_index_entry_scene_is_downloaded_with_the_model_and_kept_in_use() {
    // A cache limit of one byte: every prune evicts whatever is not in use.
    let mut settings = Settings::default();
    settings.cache.max_bytes = 1;
    let mut t = TestApp::with("i03-scene", Setup::default(), move |app: &mut App| {
        app.insert_resource(SettingsRes(settings));
    });
    let probe = http_source(&mut t);
    let ply = entry_of(&t, EntryKind::Ply);
    let url = t.server.url_of("checkpoint/scene/scene.json");
    t.app.world_mut().resource_mut::<Library>().entries[probe]
        .entry
        .scene = Some(url);
    load(&mut t, probe);
    assert!(t.until(3000, |app| {
        app.world_mut()
            .query::<&LoadedModel>()
            .iter(app.world())
            .count()
            == 1
    }));
    t.frames(30);
    let manifest =
        fs::read(ftgsxr_testkit::fixtures_dir().join("checkpoint/scene/scene.json")).unwrap();
    assert_eq!(cached_with(&t.dir, &manifest), 1, "the manifest is cached");
    assert_eq!(
        cached(&t.dir),
        2,
        "the model's and the manifest's entries stay"
    );
    // Unloaded, the manifest leaves the in-use set: the prune after the next load evicts it.
    t.command(LibraryCommand::UnloadAll);
    t.frames(2);
    load(&mut t, ply);
    assert!(t.until(3000, |app| {
        app.world_mut()
            .query::<&LoadedModel>()
            .iter(app.world())
            .count()
            == 1
    }));
    let dir = t.dir.clone();
    assert!(
        t.until(600, |_| cached(&dir) == 1),
        "only the PLY's entry stays"
    );
    assert_eq!(cached_with(&t.dir, &manifest), 0, "the manifest is evicted");
}

// §1.7, #374: an index `scene` that fails to download leaves the model to load without
// training cameras, as a soundtrack's failure does (§1.10).
#[test]
fn i03_an_index_scene_that_fails_to_download_still_loads_the_model() {
    let mut t = TestApp::new("i03-scene-404");
    let probe = http_source(&mut t);
    let url = t.server.url_of("listing/models/missing-scene.json");
    t.app.world_mut().resource_mut::<Library>().entries[probe]
        .entry
        .scene = Some(url);
    load(&mut t, probe);
    assert!(
        t.until(3000, |app| {
            app.world_mut()
                .query::<&LoadedModel>()
                .iter(app.world())
                .count()
                == 1
        }),
        "the model loads: {:?}",
        t.app.world().resource::<Library>().entries[probe].state
    );
    let m = instances(&mut t).remove(0);
    assert!(matches!(m.source, ModelRef::Http(_)));
}
