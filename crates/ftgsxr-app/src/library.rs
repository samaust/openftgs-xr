// SPDX-License-Identifier: Apache-2.0

//! The library (§2.1 `ftgsxr_app::library`, §1.5): the model sources, their listings
//! and header-only information, loads, unloads, the pickers and dropped files (§1.7).
//! Listings, header reads and fetches run on the I/O task pool, the CPU part of a load on
//! the compute pool through `SplatLoad` (§2.4 Threads, Q-99). No toast is raised here:
//! the entries' states carry the texts, and a failure with no entry to carry it is a
//! `LibraryFailure` message that `ui` turns into a toast (PRD #275, Q-173).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::Poll;

use bevy::app::{App, Plugin, PostUpdate, Update};
use bevy::asset::{Assets, Handle};
use bevy::camera::visibility::Visibility;
use bevy::ecs::prelude::*;
use bevy::image::Image;
use bevy::log::warn;
use bevy::tasks::{IoTaskPool, Task, futures::check_ready};
use bevy::transform::components::Transform;
use bevy::window::FileDragAndDrop;
use ftgsxr_core::{ModelInfo, ModelKind, ModelTime, memory_estimate};
use ftgsxr_io::{
    DocumentProvider, Entry, EntryKind, HttpClient, IoError, ModelBytes, ModelRef, ModelSource,
    Progress, SourceSpec, TlsRoots, Url, open_ref, open_source,
};
use ftgsxr_render::{SplatAsset, SplatEvent, SplatGpu, SplatLoad, SplatModel, SplatTierState};

use crate::platform::{PickHandle, PickKind, PlatformRes};

/// A library entry's state (§1.5).
#[derive(Clone, Debug, PartialEq)]
pub enum EntryState {
    /// Listed, not loaded.
    Listed,
    /// Being fetched, with its progress.
    Fetching(Progress),
    /// Being read, validated, packed or uploaded.
    Loading,
    /// Loaded as this instance.
    Loaded(Entity),
    /// The load failed: `<code>: <user text>` (§1.15).
    Failed(String),
    /// Named by a scene, offered by no source (§1.5 Scene files).
    Missing,
}

/// One model a source offers (§1.5 Library).
#[derive(Clone, Debug, PartialEq)]
pub struct LibraryEntry {
    /// The index of its source in `Library.sources`.
    pub source: usize,
    /// The listing's entry.
    pub entry: Entry,
    /// The header-only information, once read (or once the model was loaded).
    pub info: Option<ModelInfo>,
    /// The model's identity (64 lowercase hex digits), once loaded.
    pub identity: Option<String>,
    /// The thumbnail, once rendered.
    pub thumbnail: Option<Handle<Image>>,
    /// Its state.
    pub state: EntryState,
}

/// The configured sources and every entry they list (§1.5). `persist` sets the sources
/// from the settings (#285); until then the platform's models directory alone.
#[derive(Resource, Default)]
pub struct Library {
    /// The sources, each with its specification.
    pub sources: Vec<(SourceSpec, Arc<dyn ModelSource>)>,
    /// Their entries, in source order, each source's in listing order.
    pub entries: Vec<LibraryEntry>,
}

/// What the UI and the command line ask of the library (§1.5, §1.7). Like `Intent`, it
/// derives `Message` (written with a `MessageWriter`, read in the frame's order) and
/// `Event` (Q-166).
// Q-166
#[derive(Message, Event, Clone, Debug, PartialEq)]
pub enum LibraryCommand {
    /// Adds a source and lists it.
    AddSource(SourceSpec),
    /// Removes a source and its entries.
    RemoveSource(usize),
    /// Lists a source again.
    Refresh(usize),
    /// Loads an entry; `reduce` keeps the primitives the budget holds (§1.2 option 4).
    Load {
        /// The entry's index in `Library.entries`.
        entry: usize,
        /// Reduce to the budget (D-13, Q-158).
        reduce: bool,
    },
    /// Loads a model by reference (§1.7 `play`, picked and dropped files).
    LoadRef(ModelRef),
    /// Cancels an entry's fetch.
    Cancel(usize),
    /// Unloads an instance; its entry stays.
    Unload(Entity),
    /// Unloads every instance.
    UnloadAll,
    /// Opens a picker (§1.5 Picked files).
    Pick(PickKind),
}

/// A failure the library has no entry state for (Q-173): a load without a library entry
/// (`LoadRef`, picked and dropped files) whose fetch, read, validation or upload failed,
/// or a source's listing that failed (Q-193). `ui` turns it into a toast.
// Q-173, Q-193
#[derive(Message, Event, Clone, Debug, PartialEq)]
pub struct LibraryFailure {
    /// The model's name (its file stem), or the source's label for a listing.
    pub name: String,
    /// The model's reference; `None` for a listing.
    pub reference: Option<ModelRef>,
    /// The §1.15 code, `E-NET-FETCH` say.
    pub code: String,
    /// The full text, `<code>: <user text>`, an E-MEM-BUDGET one followed by ` (N = n)`
    /// (Q-117).
    pub text: String,
}

impl LibraryFailure {
    fn new(name: String, reference: Option<ModelRef>, text: String) -> LibraryFailure {
        let code = text.split(": ").next().unwrap_or("").to_owned();
        LibraryFailure {
            name,
            reference,
            code,
            text,
        }
    }
}

/// What an instance is (§1.5 Instances), inserted when its model is read (Q-148).
#[derive(Component, Clone, Debug, PartialEq)]
pub struct LoadedModel {
    /// Where it came from.
    pub source: ModelRef,
    /// Its identity, 64 lowercase hex digits (D-40).
    pub identity: String,
    /// Its information (after any reduction, `n` the rows kept).
    pub info: ModelInfo,
    /// Its name, the file stem.
    pub name: String,
    /// Its time mapping (E9).
    pub time: ModelTime,
    /// Whether it was reduced to the budget (§1.2 option 4, #205).
    pub reduced: bool,
    /// Its soundtrack (§1.10): a local or document file; an HTTP index entry's is
    /// downloaded with the model and given as its cached file.
    pub audio: Option<ModelRef>,
    /// §1.6's Δ_frame in seconds (D-32), the asset's (Q-174).
    pub frame_step_s: f64,
}

/// The system that reads `LibraryCommand`s, for the modules that write them to order
/// themselves before it (persist's first settings sync, #279).
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct LibraryCommands;

/// The library's systems and observers, added by `AppPlugin` with its `initial` models.
pub(crate) struct LibraryPlugin {
    pub(crate) initial: Vec<ModelRef>,
    /// The HTTP cache's size limit, `Settings.cache.max_bytes` (`AppPlugin` maps it).
    pub(crate) cache_max_bytes: u64,
}

impl Plugin for LibraryPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<LibraryCommand>()
            .add_message::<LibraryFailure>()
            .add_message::<FileDragAndDrop>()
            .add_message::<LocateEntry>()
            .init_resource::<Pending>()
            .add_observer(on_model)
            .add_observer(on_splat_event)
            .add_systems(
                Update,
                (
                    read_drops,
                    run_locates,
                    run_commands.in_set(LibraryCommands),
                    poll_tasks,
                    poll_picks,
                )
                    .chain(),
            )
            .add_systems(PostUpdate, follow_in_use);
        let paths = app.world().resource::<PlatformRes>().0.paths().clone();
        let max_bytes = self.cache_max_bytes;
        // The client of HTTP sources and references, and of `set_in_use` (Q-4).
        let client = match HttpClient::new(paths.cache.clone(), TlsRoots::WebPki, max_bytes) {
            Ok(c) => Some(Arc::new(c)),
            Err(e) => {
                warn!(
                    "the HTTP cache {}: {}: {e}",
                    paths.cache.display(),
                    e.code()
                );
                None
            }
        };
        app.insert_resource(Client(client));
        let world = app.world_mut();
        if !world.contains_resource::<Library>() {
            // Until `persist` (#285) sets the sources: the platform's models directory.
            let spec = SourceSpec {
                kind: "dir".into(),
                location: paths.models.to_string_lossy().into_owned(),
                name: "models".into(),
                flavor: None,
            };
            let mut library = Library::default();
            match open_source(&spec, None, None) {
                Ok(source) => library.sources.push((spec, source)),
                Err(e) => warn!("{}: {}: {e}", spec.location, e.code()),
            }
            world.insert_resource(library);
        }
        let n = world.resource::<Library>().sources.len();
        for i in 0..n {
            world.write_message(LibraryCommand::Refresh(i));
        }
        for r in &self.initial {
            world.write_message(LibraryCommand::LoadRef(r.clone()));
        }
    }
}

/// "Locate…" on a missing entry (§1.5 Scene files, PRD #275's Library decision): opens
/// the file picker; the picked file loads only if its identity is the entry's, and its
/// load clears the entry (Q-225). The UI writes it.
// Q-225
#[derive(Message, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LocateEntry(pub(crate) usize);

/// Opens a picker for each `LocateEntry` on a missing entry with an identity.
fn run_locates(
    mut reader: MessageReader<LocateEntry>,
    library: Res<Library>,
    mut pending: ResMut<Pending>,
    platform: Res<PlatformRes>,
) {
    for LocateEntry(i) in reader.read() {
        let Some(e) = library.entries.get(*i) else {
            continue;
        };
        if e.state != EntryState::Missing {
            continue;
        }
        let Some(id) = e.identity.clone() else {
            continue;
        };
        let h = platform.0.pick(PickKind::Files);
        pending.locates.push((id, h));
    }
}

/// Whether an instance not yet uploaded shows when its upload completes (Q-148): true
/// from the library, false for a scene's hidden instance (`persist`); removed at the
/// upload.
// Q-202
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub(crate) struct ShowOnUpload(pub(crate) bool);

/// The HTTP client the sources and references share (Q-4, Q-47).
#[derive(Resource)]
struct Client(Option<Arc<HttpClient>>);

/// What an instance's load knows that its model does not (Q-148).
#[derive(Component, Clone, Debug)]
struct LoadCtx {
    source: ModelRef,
    name: String,
    /// The entry's source id and location, for a load from the library.
    entry: Option<(u64, String)>,
    /// The file's N, when known before the load.
    file_n: Option<u64>,
    reduce_to: Option<usize>,
    audio: Option<ModelRef>,
    /// The identity a "Locate…" pick must have (Q-225).
    locate: Option<String>,
}

type ListTask = Task<Result<Vec<Entry>, IoError>>;
/// A subdirectory's listing: its source's id, the directory entry's location, the task.
type SubTask = (u64, String, ListTask);
type InfoTask = Task<Result<ModelInfo, IoError>>;
type Fetched = Result<(ModelRef, ModelBytes, Option<ModelRef>), IoError>;

/// A fetch on the I/O task pool.
struct Fetch {
    ctx: LoadCtx,
    kind_hint: Option<ModelKind>,
    cancel: Arc<AtomicBool>,
    progress: Arc<Mutex<Option<Progress>>>,
    task: Task<Fetched>,
}

/// The library's tasks in flight, and the sources' stable ids.
#[derive(Resource, Default)]
struct Pending {
    /// One id per source of `Library.sources`, in its order.
    ids: Vec<u64>,
    next_id: u64,
    listings: Vec<(u64, ListTask)>,
    /// Subdirectories listed on demand (D-31, Q-192).
    sublistings: Vec<SubTask>,
    infos: Vec<(u64, String, InfoTask)>,
    fetches: Vec<Fetch>,
    picks: Vec<(PickKind, PickHandle)>,
    /// A fetch completed since the last prune (Q-4, Q-45).
    prune: bool,
    /// The prune on the I/O pool, until it finishes (#291).
    pruning: Option<Task<()>>,
    /// The commands read while it ran, in their order (#291).
    deferred: Vec<LibraryCommand>,
    /// The open "Locate…" pickers, each with the identity it looks for (Q-225).
    locates: Vec<(String, PickHandle)>,
    /// Picked files answering a "Locate…", until their `LoadRef` is read.
    locating: Vec<(ModelRef, String)>,
}

impl Pending {
    /// Whether a prune is still running on the I/O pool.
    fn is_pruning(&mut self) -> bool {
        if self.pruning.as_ref().is_some_and(Task::is_finished) {
            self.pruning = None;
        }
        self.pruning.is_some()
    }
}

impl Pending {
    /// Keeps `ids` one per source: sources the caller pushed get new ids.
    fn sync(&mut self, sources: usize) {
        while self.ids.len() < sources {
            self.ids.push(self.next_id);
            self.next_id += 1;
        }
        self.ids.truncate(sources);
    }

    fn index(&self, id: u64) -> Option<usize> {
        self.ids.iter().position(|&i| i == id)
    }
}

/// `<code>: <user text>` (§1.15, Q-64).
fn error_text(e: &IoError) -> String {
    format!("{}: {}", e.code(), e.user_text())
}

/// The kind hint of a listed entry (Q-99: sniffed when `None`).
fn kind_hint(kind: EntryKind) -> Option<ModelKind> {
    match kind {
        EntryKind::Exchange => Some(ModelKind::ExchangeFile),
        EntryKind::Ply => Some(ModelKind::Ply),
        // A checkpoint travels by path (D-51); the bytes' variant decides.
        _ => None,
    }
}

/// The name of a reference: its file stem (§1.5 Instances).
fn ref_name(r: &ModelRef) -> String {
    let last = match r {
        ModelRef::File(p) => return stem(p),
        ModelRef::Http(url) => url
            .path_segments()
            .and_then(|mut s| s.rfind(|s| !s.is_empty()))
            .unwrap_or("")
            .to_owned(),
        ModelRef::Document(uri) => uri.rsplit('/').next().unwrap_or(uri).to_owned(),
    };
    stem(Path::new(&last))
}

fn stem(p: &Path) -> String {
    p.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// §1.10: a file with the model's stem and `.ogg`, `.wav`, `.flac` or `.mp3` beside a
/// local model, the first that exists in that order.
fn sidecar(r: &ModelRef) -> Option<ModelRef> {
    let ModelRef::File(path) = r else {
        return None;
    };
    let dir = path.parent()?;
    let stem = path.file_stem()?;
    ["ogg", "wav", "flac", "mp3"].iter().find_map(|ext| {
        let mut name = stem.to_os_string();
        name.push(".");
        name.push(ext);
        let p = dir.join(name);
        p.is_file().then_some(ModelRef::File(p))
    })
}

/// Lists a source on the I/O pool.
fn list(pending: &mut Pending, id: u64, source: Arc<dyn ModelSource>) {
    let task = IoTaskPool::get().spawn(async move { source.list("") });
    pending.listings.retain(|(i, _)| *i != id);
    pending.listings.push((id, task));
}

/// A dropped file is a `LoadRef` (§1.7).
fn read_drops(
    mut drops: MessageReader<FileDragAndDrop>,
    mut commands: MessageWriter<LibraryCommand>,
) {
    for drop in drops.read() {
        if let FileDragAndDrop::DroppedFile { path_buf, .. } = drop {
            commands.write(LibraryCommand::LoadRef(ModelRef::File(path_buf.clone())));
        }
    }
}

/// The reduce-to-budget count (Q-158): the budget's remaining bytes over the static
/// bytes of one primitive under the tier's layout and the governor's SH cap, floored;
/// `None` without a device or when every row fits.
// Q-158
fn reduce_count(info: &ModelInfo, gpu: Option<&SplatGpu>, tier: &SplatTierState) -> Option<usize> {
    let dev = gpu?.device();
    let remaining = dev
        .memory_budget()
        .bytes
        .saturating_sub(dev.model_bytes_in_use());
    let opts = ftgsxr_core::GpuLayoutOptions {
        sh_cap: Some(tier.sh_cap),
        ..tier.tier.layout
    };
    let per = memory_estimate(1, info.sh_degree, &opts, false)
        .static_bytes
        .max(1);
    let count = remaining / per;
    (count < info.n).then(|| usize::try_from(count).unwrap_or(usize::MAX))
}

#[expect(
    clippy::too_many_arguments,
    reason = "a Bevy system: one parameter per resource it reads"
)]
fn run_commands(
    mut reader: MessageReader<LibraryCommand>,
    mut library: ResMut<Library>,
    mut pending: ResMut<Pending>,
    client: Res<Client>,
    platform: Res<PlatformRes>,
    gpu: Option<Res<SplatGpu>>,
    tier: Res<SplatTierState>,
    instances: Query<(Entity, &LoadCtx)>,
    mut commands: Commands,
) {
    // #291: no fetch runs beside a prune. A download holds the cache lock, so a prune
    // spawned before it would wait behind it on multi-threaded pools and then read an
    // in-use set without the model just fetched, evicting its new entry. A load read
    // while a prune runs waits for it (a few milliseconds), and so does every command
    // after it, to keep their order.
    let read: Vec<LibraryCommand> = reader.read().cloned().collect();
    let mut queue = std::mem::take(&mut pending.deferred);
    queue.extend(read);
    let pruning = pending.is_pruning();
    let provider: Option<Arc<dyn DocumentProvider>> = platform.0.document_provider();
    for command in queue {
        let fetches = matches!(
            command,
            LibraryCommand::Load { .. } | LibraryCommand::LoadRef(_)
        );
        if !pending.deferred.is_empty() || (pruning && fetches) {
            pending.deferred.push(command);
            continue;
        }
        pending.sync(library.sources.len());
        match command {
            LibraryCommand::AddSource(spec) => {
                match open_source(&spec, client.0.clone(), provider.clone()) {
                    Ok(source) => {
                        library.sources.push((spec, source.clone()));
                        pending.sync(library.sources.len());
                        let id = *pending.ids.last().expect("invariant: just synced");
                        list(&mut pending, id, source);
                    }
                    Err(e) => warn!("source {}: {}", spec.location, error_text(&e)),
                }
            }
            LibraryCommand::RemoveSource(i) => {
                if i >= library.sources.len() {
                    continue;
                }
                library.sources.remove(i);
                let id = pending.ids.remove(i);
                pending.listings.retain(|(s, _)| *s != id);
                pending.infos.retain(|(s, _, _)| *s != id);
                pending.sublistings.retain(|(s, _, _)| *s != id);
                library.entries.retain(|e| e.source != i);
                for e in &mut library.entries {
                    if e.source > i {
                        e.source -= 1;
                    }
                }
            }
            LibraryCommand::Refresh(i) => {
                if let Some((_, source)) = library.sources.get(i) {
                    let source = source.clone();
                    let id = pending.ids[i];
                    list(&mut pending, id, source);
                }
            }
            LibraryCommand::Load { entry, reduce } => {
                let Some(src) = library.entries.get(entry).map(|e| e.source) else {
                    continue;
                };
                let source = library.sources[src].1.clone();
                let id = pending.ids[src];
                let e = &mut library.entries[entry];
                if matches!(e.state, EntryState::Fetching(_) | EntryState::Loading) {
                    continue;
                }
                // Q-192: loading a listed directory lists it (D-31's "on demand").
                if e.entry.kind == EntryKind::Dir {
                    e.state = EntryState::Loading;
                    let location = e.entry.location.clone();
                    let at = location.clone();
                    let task = IoTaskPool::get().spawn(async move { source.list(&at) });
                    pending
                        .sublistings
                        .retain(|(s, l, _)| (*s, l) != (id, &location));
                    pending.sublistings.push((id, location, task));
                    continue;
                }
                // §1.10: the index entry's soundtrack, downloaded with the model, else the
                // sidecar beside the file.
                let audio_url = e.entry.audio.as_deref().and_then(|a| Url::parse(a).ok());
                let http = client.0.clone();
                let reduce_to = if reduce {
                    e.info
                        .as_ref()
                        .and_then(|info| reduce_count(info, gpu.as_deref(), &tier))
                } else {
                    None
                };
                let listed = e.entry.clone();
                let cancel = Arc::new(AtomicBool::new(false));
                let progress = Arc::new(Mutex::new(None));
                let (c, p) = (cancel.clone(), progress.clone());
                let task = IoTaskPool::get().spawn(async move {
                    let (r, bytes) = source.fetch(
                        &listed,
                        &mut |now| *p.lock().unwrap_or_else(|e| e.into_inner()) = Some(now),
                        &c,
                    )?;
                    let audio = match (audio_url, http) {
                        (Some(url), Some(http)) => {
                            match http.download(&url, None, &mut |_| {}, &c) {
                                Ok(path) => Some(ModelRef::File(path)),
                                Err(e) => {
                                    warn!("the soundtrack {url}: {}", error_text(&e));
                                    None
                                }
                            }
                        }
                        _ => sidecar(&r),
                    };
                    Ok((r, bytes, audio))
                });
                e.state = EntryState::Fetching(Progress {
                    done: 0,
                    total: e.entry.size,
                });
                pending.fetches.push(Fetch {
                    ctx: LoadCtx {
                        source: ModelRef::File(PathBuf::new()),
                        name: stem(Path::new(&e.entry.name)),
                        entry: Some((id, e.entry.location.clone())),
                        file_n: e.info.as_ref().map(|i| i.n),
                        reduce_to,
                        audio: None,
                        locate: None,
                    },
                    kind_hint: kind_hint(e.entry.kind),
                    cancel,
                    progress,
                    task,
                });
            }
            LibraryCommand::LoadRef(r) => {
                // Q-225: a pick answering "Locate…" carries the identity it must have.
                let locate = pending
                    .locating
                    .iter()
                    .position(|(x, _)| *x == r)
                    .map(|i| pending.locating.remove(i).1);
                let (client, provider) = (client.0.clone(), provider.clone());
                let at = r.clone();
                let task = IoTaskPool::get().spawn(async move {
                    let bytes = open_ref(&at, client.as_deref(), provider.as_deref())?;
                    let audio = sidecar(&at);
                    Ok((at, bytes, audio))
                });
                pending.fetches.push(Fetch {
                    ctx: LoadCtx {
                        name: ref_name(&r),
                        source: r,
                        entry: None,
                        file_n: None,
                        reduce_to: None,
                        audio: None,
                        locate,
                    },
                    kind_hint: None,
                    cancel: Arc::new(AtomicBool::new(false)),
                    progress: Arc::new(Mutex::new(None)),
                    task,
                });
            }
            LibraryCommand::Cancel(i) => {
                let Some(e) = library.entries.get(i) else {
                    continue;
                };
                let key = (pending.ids[e.source], e.entry.location.clone());
                for f in &pending.fetches {
                    if f.ctx.entry.as_ref() == Some(&key) {
                        f.cancel.store(true, Ordering::Relaxed);
                    }
                }
            }
            LibraryCommand::Unload(entity) => {
                if let Ok((entity, ctx)) = instances.get(entity) {
                    unload(&mut library, &pending, entity, ctx, &mut commands);
                }
            }
            LibraryCommand::UnloadAll => {
                for (entity, ctx) in &instances {
                    unload(&mut library, &pending, entity, ctx, &mut commands);
                }
            }
            LibraryCommand::Pick(kind) => {
                let h = platform.0.pick(kind);
                pending.picks.push((kind, h));
            }
        }
    }
}

/// Despawns an instance; its entry, if any, returns to `Listed` (§1.5).
fn unload(
    library: &mut Library,
    pending: &Pending,
    entity: Entity,
    ctx: &LoadCtx,
    commands: &mut Commands,
) {
    if let Some(e) = entry_of(library, pending, ctx) {
        e.state = EntryState::Listed;
    }
    commands.entity(entity).despawn();
}

/// The library entry a load came from, while its source is configured.
fn entry_of<'a>(
    library: &'a mut Library,
    pending: &Pending,
    ctx: &LoadCtx,
) -> Option<&'a mut LibraryEntry> {
    let (id, location) = ctx.entry.as_ref()?;
    let source = pending.index(*id)?;
    library
        .entries
        .iter_mut()
        .find(|e| e.source == source && &e.entry.location == location)
}

/// A listing's entries as library entries of `source`: `old`'s state kept for a location
/// listed again, and its information, identity and thumbnail when the file did not change
/// (§1.5 Refresh); a header read started for every `Exchange` entry without information.
/// Directories stay, to list on demand (D-31, Q-192).
fn fresh_entries(
    pending: &mut Pending,
    id: u64,
    source: usize,
    model: &Arc<dyn ModelSource>,
    entries: Vec<Entry>,
    old: &[LibraryEntry],
) -> Vec<LibraryEntry> {
    let mut fresh = Vec::new();
    for entry in entries {
        if entry.kind == EntryKind::Other {
            continue;
        }
        let kept = old.iter().find(|o| o.entry.location == entry.location);
        let same = kept.is_some_and(|o| {
            o.entry.size == entry.size
                && o.entry.mtime == entry.mtime
                && o.entry.sha256 == entry.sha256
        });
        let (info, identity, thumbnail) = match kept {
            Some(o) if same => (o.info.clone(), o.identity.clone(), o.thumbnail.clone()),
            _ => (None, None, None),
        };
        let state = kept.map_or(EntryState::Listed, |o| o.state.clone());
        if info.is_none() && entry.kind == EntryKind::Exchange {
            let (m, e) = (model.clone(), entry.clone());
            let task = IoTaskPool::get().spawn(async move { m.info(&e) });
            pending.infos.push((id, entry.location.clone(), task));
        }
        fresh.push(LibraryEntry {
            source,
            entry,
            info,
            identity,
            thumbnail,
            state,
        });
    }
    // Q-203: a scene's missing model stays listed until it is located.
    for o in old {
        if o.state == EntryState::Missing
            && !fresh.iter().any(|f| f.entry.location == o.entry.location)
        {
            fresh.push(o.clone());
        }
    }
    fresh
}

/// Whether a listing failed only because `AppPaths.models`, the default `dir` source,
/// does not exist yet (Q-216); any other source's missing directory stays E-IO.
// Q-216
fn is_missing_default(spec: &SourceSpec, e: &IoError, default_models: &Path) -> bool {
    spec.kind == "dir"
        && Path::new(&spec.location) == default_models
        && matches!(e, IoError::Io(io) if io.kind() == std::io::ErrorKind::NotFound)
        && !default_models.exists()
}

/// Collects finished listings, header reads and fetches.
fn poll_tasks(
    mut library: ResMut<Library>,
    mut pending: ResMut<Pending>,
    mut failures: MessageWriter<LibraryFailure>,
    platform: Res<PlatformRes>,
    mut commands: Commands,
) {
    pending.sync(library.sources.len());
    let pending = &mut *pending;
    let default_models = platform.0.paths().models.clone();
    // Listings: entries keep their state and, when the file did not change, their
    // information (§1.5 Refresh). A source's root listing replaces every entry of it,
    // the subdirectories' included (Q-192).
    let mut listed = Vec::new();
    pending
        .listings
        .retain_mut(|(id, task)| match check_ready(task) {
            Some(result) => {
                listed.push((*id, result));
                false
            }
            None => true,
        });
    for (id, result) in listed {
        let Some(source) = pending.index(id) else {
            continue;
        };
        let entries = match result {
            Ok(entries) => entries,
            // Q-216: a first run has no default models directory yet (Q-14 creates
            // none); that source lists empty, without a toast.
            Err(e) if is_missing_default(&library.sources[source].0, &e, &default_models) => {
                Vec::new()
            }
            Err(e) => {
                // Q-193: a listing has no entry to carry its failure.
                let spec = &library.sources[source].0;
                warn!("listing {}: {}", spec.location, error_text(&e));
                failures.write(LibraryFailure::new(spec.name.clone(), None, error_text(&e)));
                continue;
            }
        };
        let mut old: Vec<LibraryEntry> = Vec::new();
        library.entries.retain(|e| {
            if e.source == source {
                old.push(e.clone());
                false
            } else {
                true
            }
        });
        let at = library
            .entries
            .iter()
            .position(|e| e.source > source)
            .unwrap_or(library.entries.len());
        let model = library.sources[source].1.clone();
        let fresh = fresh_entries(pending, id, source, &model, entries, &old);
        library.entries.splice(at..at, fresh);
    }
    // Subdirectories: their entries follow the directory's, replacing any listed before
    // (Q-192).
    let mut sublisted = Vec::new();
    pending
        .sublistings
        .retain_mut(|(id, location, task)| match check_ready(task) {
            Some(result) => {
                sublisted.push((*id, location.clone(), result));
                false
            }
            None => true,
        });
    for (id, location, result) in sublisted {
        let Some(source) = pending.index(id) else {
            continue;
        };
        let Some(dir) = library
            .entries
            .iter()
            .position(|e| e.source == source && e.entry.location == location)
        else {
            continue;
        };
        let entries = match result {
            Ok(entries) => entries,
            Err(e) => {
                library.entries[dir].state = EntryState::Failed(error_text(&e));
                continue;
            }
        };
        library.entries[dir].state = EntryState::Listed;
        let inside = |e: &LibraryEntry| {
            e.source == source
                && e.entry.location != location
                && e.entry.location.starts_with(&location)
        };
        let old: Vec<LibraryEntry> = library
            .entries
            .iter()
            .filter(|e| inside(e))
            .cloned()
            .collect();
        library.entries.retain(|e| !inside(e));
        let model = library.sources[source].1.clone();
        let fresh = fresh_entries(pending, id, source, &model, entries, &old);
        library.entries.splice(dir + 1..dir + 1, fresh);
    }
    // Header-only information, filled as it arrives.
    let mut infos = Vec::new();
    pending
        .infos
        .retain_mut(|(id, location, task)| match check_ready(task) {
            Some(result) => {
                infos.push((*id, location.clone(), result));
                false
            }
            None => true,
        });
    for (id, location, result) in infos {
        let Some(source) = pending.index(id) else {
            continue;
        };
        let Some(e) = library
            .entries
            .iter_mut()
            .find(|e| e.source == source && e.entry.location == location)
        else {
            continue;
        };
        match result {
            Ok(info) => e.info = Some(info),
            // The load reports the error with its code (Q-38): the entry stays listed.
            Err(err) => warn!("{location}: {}", error_text(&err)),
        }
    }
    // Fetches: an instance per model, hidden until its upload (Q-148).
    let fetches = std::mem::take(&mut pending.fetches);
    for mut f in fetches {
        let Some(result) = check_ready(&mut f.task) else {
            if let Some(p) = *f.progress.lock().unwrap_or_else(|e| e.into_inner())
                && let Some(e) = entry_of(&mut library, pending, &f.ctx)
            {
                e.state = EntryState::Fetching(p);
            }
            pending.fetches.push(f);
            continue;
        };
        match result {
            Ok((source, bytes, audio)) => {
                let mut ctx = f.ctx;
                ctx.source = source;
                ctx.audio = audio;
                commands.spawn((
                    SplatLoad {
                        bytes,
                        kind_hint: f.kind_hint,
                        reduce_to: ctx.reduce_to,
                    },
                    Transform::IDENTITY,
                    Visibility::Hidden,
                    ShowOnUpload(true),
                    ctx.clone(),
                ));
                if let Some(e) = entry_of(&mut library, pending, &ctx) {
                    e.state = EntryState::Loading;
                }
                pending.prune = true;
            }
            // A cancelled fetch returns its entry to the list (§1.5).
            Err(_) if f.cancel.load(Ordering::Relaxed) => {
                if let Some(e) = entry_of(&mut library, pending, &f.ctx) {
                    e.state = EntryState::Listed;
                }
            }
            Err(err) => {
                let text = error_text(&err);
                match entry_of(&mut library, pending, &f.ctx) {
                    Some(e) => e.state = EntryState::Failed(text),
                    // Q-173: a load without an entry sends its own failure.
                    None => {
                        warn!("{}: {text}", f.ctx.name);
                        let reference = Some(f.ctx.source.clone());
                        failures.write(LibraryFailure::new(f.ctx.name, reference, text));
                    }
                }
            }
        }
    }
}

/// Polls the open pickers: picked files load, a picked folder joins as a source for
/// the session (§1.5 Picked files).
fn poll_picks(
    mut pending: ResMut<Pending>,
    platform: Res<PlatformRes>,
    mut commands: MessageWriter<LibraryCommand>,
) {
    // Q-225: the first picked file answers a "Locate…".
    let locates = std::mem::take(&mut pending.locates);
    for (id, h) in locates {
        match platform.0.poll_pick(h) {
            Poll::Pending => pending.locates.push((id, h)),
            Poll::Ready(Err(e)) => warn!("the picker failed: {e}"),
            Poll::Ready(Ok(refs)) => {
                if let Some(r) = refs.into_iter().next() {
                    pending.locating.push((r.clone(), id));
                    commands.write(LibraryCommand::LoadRef(r));
                }
            }
        }
    }
    let picks = std::mem::take(&mut pending.picks);
    for (kind, h) in picks {
        match platform.0.poll_pick(h) {
            Poll::Pending => pending.picks.push((kind, h)),
            Poll::Ready(Err(e)) => warn!("the picker failed: {e}"),
            Poll::Ready(Ok(refs)) => {
                for r in refs {
                    match (kind, r) {
                        // Q-227: a picked file loads once; only folders join as sources.
                        (PickKind::Files, r) => commands.write(LibraryCommand::LoadRef(r)),
                        (PickKind::Folder, r) => {
                            let (kind, location) = match &r {
                                ModelRef::File(p) => ("dir", p.to_string_lossy().into_owned()),
                                ModelRef::Document(uri) => ("documents", uri.clone()),
                                ModelRef::Http(url) => ("http", url.to_string()),
                            };
                            commands.write(LibraryCommand::AddSource(SourceSpec {
                                kind: kind.into(),
                                name: ref_name(&r),
                                location,
                                flavor: None,
                            }))
                        }
                    };
                }
            }
        }
    }
}

/// `LoadedModel` as soon as the model is read (Q-148).
// Q-148
fn on_model(
    add: On<Add, SplatModel>,
    loads: Query<(&LoadCtx, &SplatModel)>,
    assets: Res<Assets<SplatAsset>>,
    mut library: ResMut<Library>,
    pending: Res<Pending>,
    mut failures: MessageWriter<LibraryFailure>,
    mut commands: Commands,
) {
    let Ok((ctx, model)) = loads.get(add.entity) else {
        return;
    };
    let Some(asset) = assets.get(&model.0) else {
        return;
    };
    let identity: String = asset.identity.iter().map(|b| format!("{b:02x}")).collect();
    // Q-225: a located file of another identity is refused; a load of a missing
    // entry's identity, located or not, clears the entry.
    if let Some(want) = &ctx.locate
        && *want != identity
    {
        let missing = library
            .entries
            .iter()
            .find(|e| e.state == EntryState::Missing && e.identity.as_ref() == Some(want))
            .map_or_else(|| ctx.name.clone(), |e| stem(Path::new(&e.entry.name)));
        failures.write(LibraryFailure::new(
            ctx.name.clone(),
            Some(ctx.source.clone()),
            format!(
                "E-SCENE-MISSING: Model {missing} not found; locate it ({} is another model)",
                ctx.name
            ),
        ));
        commands.entity(add.entity).despawn();
        return;
    }
    library
        .entries
        .retain(|e| !(e.state == EntryState::Missing && e.identity.as_ref() == Some(&identity)));
    // #205: reduced when the rows requested are fewer than the file's.
    let file_n = ctx.file_n.unwrap_or(asset.info.n);
    let reduced = ctx.reduce_to.is_some_and(|k| (k as u64) < file_n);
    if let Some(e) = entry_of(&mut library, &pending, ctx) {
        e.identity = Some(identity.clone());
        if e.info.is_none() {
            e.info = Some(asset.info.clone());
        }
    }
    commands.entity(add.entity).insert(LoadedModel {
        source: ctx.source.clone(),
        identity,
        info: asset.info.clone(),
        name: ctx.name.clone(),
        time: ModelTime::from_span(&asset.info.time),
        reduced,
        audio: ctx.audio.clone(),
        // Q-174
        frame_step_s: asset.frame_step_s,
    });
}

/// `Uploaded` shows the instance and marks its entry loaded; `Failed` removes the entity
/// and marks the entry failed, an E-MEM-BUDGET text followed by ` (N = n)` (Q-117, Q-148).
// Q-148, Q-117
fn on_splat_event(
    event: On<SplatEvent>,
    loads: Query<(&LoadCtx, Option<&SplatModel>, Option<&ShowOnUpload>)>,
    assets: Res<Assets<SplatAsset>>,
    mut library: ResMut<Library>,
    pending: Res<Pending>,
    mut failures: MessageWriter<LibraryFailure>,
    mut commands: Commands,
) {
    match event.event() {
        SplatEvent::Uploaded { entity, .. } => {
            let Ok((ctx, _, shown)) = loads.get(*entity) else {
                return;
            };
            // Q-202: a scene's hidden instance stays hidden.
            let show = shown.is_none_or(|s| s.0);
            let mut e = commands.entity(*entity);
            e.remove::<ShowOnUpload>();
            if show {
                e.insert(Visibility::Visible);
            }
            if let Some(e) = entry_of(&mut library, &pending, ctx) {
                e.state = EntryState::Loaded(*entity);
            }
        }
        SplatEvent::Failed { entity, error } => {
            let Ok((ctx, model, _)) = loads.get(*entity) else {
                return;
            };
            let asset = model.and_then(|m| assets.get(&m.0));
            let mut text = error.clone();
            if text.starts_with("E-MEM-BUDGET")
                && let Some(n) = ctx.file_n.or(asset.map(|a| a.info.n))
            {
                text = format!("{text} (N = {n})");
            }
            match entry_of(&mut library, &pending, ctx) {
                Some(e) => {
                    if e.info.is_none() && ctx.reduce_to.is_none() {
                        e.info = asset.map(|a| a.info.clone());
                    }
                    e.state = EntryState::Failed(text);
                }
                // Q-173: a load without an entry sends its own failure.
                None => {
                    failures.write(LibraryFailure::new(
                        ctx.name.clone(),
                        Some(ctx.source.clone()),
                        text,
                    ));
                }
            }
            commands.entity(*entity).despawn();
        }
        SplatEvent::Overflow { .. } => {}
    }
}

/// Q-4, Q-56: the HTTP client's in-use set follows every change of the loaded set: a
/// local file or checkpoint directory by its path, a downloaded file by its cache entry's
/// directory (§1.11 Cache entries, Q-42). After a fetch completes the cache is pruned on
/// the I/O pool (Q-45).
// Q-4, Q-56, Q-45
fn follow_in_use(
    added: Query<(), Added<LoadCtx>>,
    mut removed: RemovedComponents<LoadCtx>,
    all: Query<&LoadCtx>,
    client: Res<Client>,
    platform: Res<PlatformRes>,
    mut pending: ResMut<Pending>,
) {
    let changed = !added.is_empty() || removed.read().count() > 0;
    let Some(client) = client.0.as_ref() else {
        return;
    };
    if changed {
        let entries = platform.0.paths().cache.join("http");
        // A soundtrack downloaded with its model stays with it (§1.10).
        let paths: Vec<PathBuf> = all
            .iter()
            .flat_map(|c| [Some(&c.source), c.audio.as_ref()])
            .flatten()
            .filter_map(|r| match r {
                ModelRef::File(p) if !p.as_os_str().is_empty() => Some(p.clone()),
                ModelRef::Http(url) => Some(cache_entry(&entries, url)),
                ModelRef::Document(_) => None,
                ModelRef::File(_) => None,
            })
            .collect();
        client.set_in_use(&paths);
    }
    // #291: only once every fetch has landed in the in-use set above; `run_commands`
    // starts no fetch while the prune runs.
    if pending.prune && pending.fetches.is_empty() && !pending.is_pruning() {
        pending.prune = false;
        let client = client.clone();
        pending.pruning = Some(IoTaskPool::get().spawn(async move {
            if let Err(e) = client.prune() {
                warn!("pruning the download cache: {}", error_text(&e));
            }
        }));
    }
}

/// The directory of `url`'s cache entry under the cache's `http/` directory: the SHA-256
/// of the URL's serialization (§1.11 Cache entries, Q-42).
// Q-42
fn cache_entry(entries: &Path, url: &Url) -> PathBuf {
    entries.join(ftgsxr_core::hex(&ftgsxr_core::sha256(
        url.as_str().as_bytes(),
    )))
}
