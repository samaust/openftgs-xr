// SPDX-License-Identifier: Apache-2.0

//! Model sources (§1.5 Sources, D-31, D-49): the references a model is loaded from, the
//! `ModelSource` port the library lists, reads headers from and fetches through, its three
//! adapters (a local directory, an HTTP index, a picked document tree) and the
//! document-provider port the Android shim implements (§2.5).

use std::fs::File;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ftgsxr_core::{CoreError, ModelInfo, SAFETENSORS_PREFIX, read_info};
use url::Url;

use crate::http::status_error;
use crate::http::{MANIFEST, PARAMS};
use crate::listing::{as_directory, checked_header_len, header_info, percent_decode};
use crate::util::cancelled;
use crate::{
    Entry, EntryKind, HttpClient, Progress, list_dir, parse_html_listing, parse_index_manifest,
    parse_nginx_json,
};

/// §2.1 shows these two in this module's block; their homes are the crate root and
/// `listing`.
pub use crate::{IoError, listing::ListingFlavor};

/// Where a model comes from (§1.5): a local path, a URL, or an Android content URI.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ModelRef {
    /// A file (or a checkpoint directory) on a local file system.
    File(PathBuf),
    /// A model file on an HTTP server, or a checkpoint directory when the path ends in `/`
    /// (Q-60), fetched through the download cache.
    Http(Url),
    /// An Android content URI, read through the `DocumentProvider` (D-49).
    Document(String),
}

/// A fetched model (D-50, D-51): a file's bytes read into memory, what the readers of §1.2
/// take, or a checkpoint directory on disk, loaded by path with `read_checkpoint`. Where it
/// came from is the `ModelRef` beside it. It has no accessor: callers match on the two
/// variants, so a new variant fails to compile where it is not handled (Q-58).
// D-50, D-51, Q-58
#[derive(Clone, PartialEq, Eq)]
pub enum ModelBytes {
    /// The whole file, owned.
    Owned(Vec<u8>),
    /// A checkpoint directory holding `manifest.json` and `params.safetensors`.
    Checkpoint(PathBuf),
}

impl std::fmt::Debug for ModelBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelBytes::Owned(bytes) => write!(f, "ModelBytes::Owned({} bytes)", bytes.len()),
            ModelBytes::Checkpoint(dir) => write!(f, "ModelBytes::Checkpoint({})", dir.display()),
        }
    }
}

/// A place models are listed and fetched from (§1.5): the library's port to its sources.
pub trait ModelSource: Send + Sync {
    /// The source's location as text: the root path, the base URL, the tree URI (Q-47).
    fn name(&self) -> &str;
    /// The entries of the source's root (`subdir` empty) or of one of its subdirectories,
    /// listed on demand (§1.5, D-31; Q-47).
    fn list(&self, subdir: &str) -> Result<Vec<Entry>, IoError>;
    /// The header-only information of an `Exchange` entry (§1.5, Q-33), reading the
    /// 8-byte prefix and the header and nothing else; other kinds are E-IO (Q-47).
    fn info(&self, entry: &Entry) -> Result<ModelInfo, IoError>;
    /// The entry's file, read into memory, with the reference it was read from; `progress`
    /// hears as it arrives and `cancel` stops it (§1.5, Q-47). A checkpoint comes back as
    /// its directory on disk, `ModelBytes::Checkpoint`, referenced by `ModelRef::File` of it
    /// (D-51, Q-59); a document source refuses one (Q-61).
    fn fetch(
        &self,
        entry: &Entry,
        progress: &mut dyn FnMut(Progress),
        cancel: &AtomicBool,
    ) -> Result<(ModelRef, ModelBytes), IoError>;
}

/// A local directory scanned one level deep (§1.5): its listing is `list_dir`'s (Q-38).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalDir {
    /// The directory listed.
    pub root: PathBuf,
}

/// An HTTP index (§1.5, D-31): a base URL listed in one of the three listing flavors,
/// fetched through the shared `HttpClient`.
pub struct HttpIndex {
    /// The directory URL listed, ending in `/` (`new` appends it, Q-48), so `index.json`,
    /// subdirectories and relative entry locations resolve under it.
    pub base: Url,
    /// The listing read, or `Auto` to detect it on every listing (Q-48).
    pub flavor: ListingFlavor,
    client: Arc<HttpClient>,
}

/// The port Android's document provider implements (§2.5); the desktop has none.
pub trait DocumentProvider: Send + Sync {
    /// The documents of a tree, as entries with §1.5's kinds; `location` is each one's
    /// content URI.
    fn list_tree(&self, tree_uri: &str) -> Result<Vec<Entry>, IoError>;
    /// The document at `uri`, opened for reading.
    fn open(&self, uri: &str) -> Result<std::fs::File, IoError>;
}

/// A picked document tree (§1.5 Picked files, D-49), read through a `DocumentProvider`.
pub struct DocumentSource {
    /// The tree's content URI.
    pub tree_uri: String,
    provider: Arc<dyn DocumentProvider>,
}

/// A configured source as `settings.toml` and scene files record it (§1.11 `[[sources]]`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceSpec {
    /// `dir`, `http` or `documents` (§1.11).
    pub kind: String,
    /// The directory path, base URL or tree URI.
    pub location: String,
    /// The label the user gave it.
    pub name: String,
    /// The HTTP listing flavor; `None` is `Auto` (§1.11).
    pub flavor: Option<ListingFlavor>,
}

/// The player's own listing under an HTTP index's base (§1.5, §1.11).
const INDEX: &str = "index.json";
/// The `sha256sum` list a model's digest may be read from, beside the model (§1.5, Q-47).
const SUMS: &str = "SHA256SUMS";
/// The bytes a local or document file is read in between two cancel checks (Q-47).
const CHUNK: usize = 1 << 20;

impl HttpIndex {
    /// An HTTP index of `base` in `flavor`, fetched through `client` (Q-47). A base is a
    /// directory: one whose path does not end in `/` gets one (Q-48).
    // Q-47, Q-48
    pub fn new(base: Url, flavor: ListingFlavor, client: Arc<HttpClient>) -> HttpIndex {
        HttpIndex {
            base: as_directory(base),
            flavor,
            client,
        }
    }
}

impl DocumentSource {
    /// A document tree `tree_uri` read through `provider` (Q-47).
    // Q-47
    pub fn new(tree_uri: String, provider: Arc<dyn DocumentProvider>) -> DocumentSource {
        DocumentSource { tree_uri, provider }
    }
}

// Q-47
impl ModelSource for LocalDir {
    fn name(&self) -> &str {
        self.root.to_str().unwrap_or("")
    }

    fn list(&self, subdir: &str) -> Result<Vec<Entry>, IoError> {
        if subdir.is_empty() {
            return list_dir(&self.root);
        }
        let relative = Path::new(subdir);
        if !relative
            .components()
            .all(|c| matches!(c, Component::Normal(_)))
        {
            return Err(invalid(format!(
                "{subdir} is not a subdirectory of {}",
                self.root.display()
            )));
        }
        list_dir(&self.root.join(relative))
    }

    fn info(&self, entry: &Entry) -> Result<ModelInfo, IoError> {
        header_only(entry)?;
        let path = Path::new(&entry.location);
        let mut file = open_file(path)?;
        let len = file.metadata()?.len();
        header_info(&mut file, len)
    }

    fn fetch(
        &self,
        entry: &Entry,
        progress: &mut dyn FnMut(Progress),
        cancel: &AtomicBool,
    ) -> Result<(ModelRef, ModelBytes), IoError> {
        fetchable(entry)?;
        let path = PathBuf::from(&entry.location);
        if entry.kind == EntryKind::Checkpoint {
            let bytes = local_checkpoint(&path, progress, cancel)?;
            return Ok((ModelRef::File(path), bytes));
        }
        let bytes = read_local(&path, progress, cancel)?;
        Ok((ModelRef::File(path), bytes))
    }
}

// Q-47, Q-48
impl ModelSource for HttpIndex {
    fn name(&self) -> &str {
        self.base.as_str()
    }

    fn list(&self, subdir: &str) -> Result<Vec<Entry>, IoError> {
        let dir = if subdir.is_empty() {
            self.base.clone()
        } else {
            self.base
                .join(subdir)
                .ok()
                .filter(|d| d.as_str().starts_with(self.base.as_str()))
                .ok_or_else(|| invalid(format!("{subdir} is not under {}", self.base)))?
        };
        let index = || {
            dir.join(INDEX)
                .map_err(|e| IoError::Fetch(format!("{dir} ({e})")))
        };
        match self.flavor {
            ListingFlavor::Index => parse_index_manifest(&self.client.get_text(&index()?)?, &dir),
            ListingFlavor::NginxJson => parse_nginx_json(&self.client.get_text(&dir)?, &dir),
            ListingFlavor::Html => parse_html_listing(&self.client.get_text(&dir)?, &dir),
            ListingFlavor::Auto => self.detect(&dir, &index()?),
        }
    }

    fn info(&self, entry: &Entry) -> Result<ModelInfo, IoError> {
        header_only(entry)?;
        let url = self.entry_url(entry)?;
        // Q-47: the prefix, then the header: 8 + header-length bytes on the wire.
        let (prefix, size) = self
            .client
            .get_range_sized(&url, 0, SAFETENSORS_PREFIX as u64)?;
        let prefix: [u8; SAFETENSORS_PREFIX] = prefix
            .try_into()
            .map_err(|_| CoreError::Format(format!("{url}: shorter than its length prefix")))?;
        let len = size
            .or(entry.size)
            .ok_or_else(|| IoError::Fetch(format!("{url} (no file size)")))?;
        let n = checked_header_len(&prefix, len)?;
        let header = self.client.get_range(&url, SAFETENSORS_PREFIX as u64, n)?;
        if header.len() as u64 != n {
            return Err(CoreError::Format(format!("{url}: shorter than its header")).into());
        }
        Ok(read_info(&header, len)?)
    }

    fn fetch(
        &self,
        entry: &Entry,
        progress: &mut dyn FnMut(Progress),
        cancel: &AtomicBool,
    ) -> Result<(ModelRef, ModelBytes), IoError> {
        fetchable(entry)?;
        let url = self.entry_url(entry)?;
        if entry.kind == EntryKind::Checkpoint {
            let dir = self.client.download_checkpoint(
                &dir_url(url),
                entry.sha256.as_deref(),
                progress,
                cancel,
            )?;
            return Ok((ModelRef::File(dir.clone()), ModelBytes::Checkpoint(dir)));
        }
        // Q-47: an entry without the index's digest may have one beside the model.
        let expected = match &entry.sha256 {
            Some(digest) => Some(digest.clone()),
            None => self.sidecar_digest(entry, &url)?,
        };
        let path = self
            .client
            .download(&url, expected.as_deref(), progress, cancel)?;
        let bytes = read_local(&path, &mut |_| {}, &AtomicBool::new(false))?;
        Ok((ModelRef::Http(url), bytes))
    }
}

impl HttpIndex {
    /// `Auto`'s detection at the directory URL `dir` (D-31, Q-48): `index` when it
    /// answers 2xx, then `dir` once, as nginx JSON, then as HTML anchors. A `dir` answering
    /// 404 or 403 has no listing (E-NET-LISTING); any other failure status is E-NET-FETCH.
    // Q-48
    fn detect(&self, dir: &Url, index: &Url) -> Result<Vec<Entry>, IoError> {
        if let Ok(body) = self.client.get_text_or_status(index)? {
            return parse_index_manifest(&body, dir);
        }
        let body = match self.client.get_text_or_status(dir)? {
            Ok(body) => body,
            Err(404 | 403) => return Err(IoError::Listing(dir.to_string())),
            Err(status) => return Err(status_error(dir, status)),
        };
        parse_nginx_json(&body, dir)
            .or_else(|_| parse_html_listing(&body, dir))
            .map_err(|_| IoError::Listing(dir.to_string()))
    }

    /// The digest a file beside the model at `url` gives (§1.5 Download cache, Q-47): the
    /// first whitespace-separated token of `<name>.sha256`, else the line of `SHA256SUMS`
    /// for the model's file name. A 404, or no such line, is no digest there; any other
    /// answer is E-NET-FETCH. The token is returned as found, an empty `.sha256` giving an
    /// empty one, so that `download` refuses one that is not a digest with E-NET-CHECKSUM
    /// (Q-43).
    // Q-47
    fn sidecar_digest(&self, entry: &Entry, url: &Url) -> Result<Option<String>, IoError> {
        let named = beside(url, &format!("{}.sha256", entry.name));
        if let Some(body) = named.map_or(Ok(None), |named| self.sidecar(&named))? {
            return Ok(Some(
                body.split_whitespace().next().unwrap_or("").to_owned(),
            ));
        }
        let Some(body) = beside(url, SUMS).map_or(Ok(None), |sums| self.sidecar(&sums))? else {
            return Ok(None);
        };
        let file = url
            .path_segments()
            .and_then(|mut segments| segments.next_back())
            .map(percent_decode)
            .unwrap_or_default();
        Ok(sums_digest(&body, &file))
    }

    /// The body of the sidecar at `url`; `None` when it answers 404, E-NET-FETCH for any
    /// other failure (Q-47).
    fn sidecar(&self, url: &Url) -> Result<Option<String>, IoError> {
        match self.client.get_text_or_status(url)? {
            Ok(body) => Ok(Some(body)),
            Err(404) => Ok(None),
            Err(status) => Err(status_error(url, status)),
        }
    }

    /// The URL of an entry this index listed: its absolute `location` (Q-36), or one
    /// relative to the base.
    fn entry_url(&self, entry: &Entry) -> Result<Url, IoError> {
        self.base
            .join(&entry.location)
            .map_err(|e| IoError::Fetch(format!("{} (not a URL: {e})", entry.location)))
    }
}

/// `url` as a directory URL, ending in `/`, so a checkpoint's files resolve inside it and
/// its cache entry has one key (Q-36, Q-42).
fn dir_url(mut url: Url) -> Url {
    if !url.path().ends_with('/') {
        let path = format!("{}/", url.path());
        url.set_path(&path);
    }
    url
}

// Q-47
impl ModelSource for DocumentSource {
    fn name(&self) -> &str {
        &self.tree_uri
    }

    fn list(&self, subdir: &str) -> Result<Vec<Entry>, IoError> {
        let tree = if subdir.is_empty() {
            self.tree_uri.as_str()
        } else {
            subdir
        };
        self.provider.list_tree(tree)
    }

    fn info(&self, entry: &Entry) -> Result<ModelInfo, IoError> {
        header_only(entry)?;
        let mut file = self.provider.open(&entry.location)?;
        let len = file.metadata()?.len();
        header_info(&mut file, len)
    }

    fn fetch(
        &self,
        entry: &Entry,
        progress: &mut dyn FnMut(Progress),
        cancel: &AtomicBool,
    ) -> Result<(ModelRef, ModelBytes), IoError> {
        fetchable(entry)?;
        // Q-61: a picked checkpoint waits on the shim's document provider (M-5).
        if entry.kind == EntryKind::Checkpoint {
            return Err(unsupported(&format!(
                "{}: a checkpoint cannot be read from picked documents yet",
                entry.location
            )));
        }
        let bytes = read_document(self.provider.as_ref(), &entry.location, progress, cancel)?;
        Ok((ModelRef::Document(entry.location.clone()), bytes))
    }
}

/// Opens `source` as `spec` describes it (§1.11 `[[sources]]`; Q-47): `dir` a
/// `LocalDir`, `http` an `HttpIndex` through `client`, `documents` a `DocumentSource`
/// through `provider`. Nothing is read until the source is listed.
// Q-47
pub fn open_source(
    spec: &SourceSpec,
    client: Option<Arc<HttpClient>>,
    provider: Option<Arc<dyn DocumentProvider>>,
) -> Result<Arc<dyn ModelSource>, IoError> {
    match spec.kind.as_str() {
        "dir" => Ok(Arc::new(LocalDir {
            root: PathBuf::from(&spec.location),
        })),
        "http" => {
            let client = client.ok_or_else(|| unsupported("an HTTP source without a client"))?;
            // Q-47: a location that is not a URL is a setting to fix, E-IO.
            let base = Url::parse(&spec.location)
                .map_err(|e| invalid(format!("{} (not a URL: {e})", spec.location)))?;
            let flavor = spec.flavor.unwrap_or(ListingFlavor::Auto);
            Ok(Arc::new(HttpIndex::new(base, flavor, client)))
        }
        "documents" => {
            let provider =
                provider.ok_or_else(|| unsupported("a document source without a provider"))?;
            Ok(Arc::new(DocumentSource::new(
                spec.location.clone(),
                provider,
            )))
        }
        other => Err(invalid(format!("unknown source kind {other:?}"))),
    }
}

/// The model `r` names (§1.5): a local file read into memory, a local directory as the
/// checkpoint it holds, loaded by path (D-51), a download through `client`, or a document
/// through `provider` (Q-47). An `Http` URL whose path ends in `/` is a checkpoint directory,
/// downloaded into its cache entry as `HttpIndex::fetch` downloads one (its params checked
/// against the manifest's digest, Q-57), a missing manifest being E-NET-FETCH; any other
/// names a file, downloaded with no expected digest (Q-60).
// D-51, Q-47, Q-57, Q-60
pub fn open_ref(
    r: &ModelRef,
    client: Option<&HttpClient>,
    provider: Option<&dyn DocumentProvider>,
) -> Result<ModelBytes, IoError> {
    let cancel = AtomicBool::new(false);
    match r {
        ModelRef::File(path) => {
            if path.is_dir() {
                return Ok(ModelBytes::Checkpoint(path.clone()));
            }
            read_local(path, &mut |_| {}, &cancel)
        }
        ModelRef::Http(url) => {
            let client = client.ok_or_else(|| unsupported("an HTTP reference without a client"))?;
            // Q-60: the trailing `/` only chooses what to try; the manifest is the proof.
            if url.path().ends_with('/') {
                let dir = client.download_checkpoint(url, None, &mut |_| {}, &cancel)?;
                return Ok(ModelBytes::Checkpoint(dir));
            }
            let path = client.download(url, None, &mut |_| {}, &cancel)?;
            read_local(&path, &mut |_| {}, &cancel)
        }
        ModelRef::Document(uri) => {
            let provider =
                provider.ok_or_else(|| unsupported("a document reference without a provider"))?;
            read_document(provider, uri, &mut |_| {}, &cancel)
        }
    }
}

/// The file `name` in the directory of `url`, `name` percent-encoded as one path segment;
/// `None` for a URL without a path (Q-47).
fn beside(url: &Url, name: &str) -> Option<Url> {
    let mut out = url.clone();
    out.set_query(None);
    out.set_fragment(None);
    out.path_segments_mut().ok()?.pop().push(name);
    Some(out)
}

/// The digest of the `sha256sum` line of `sums` whose file name is `file`: `<64 hex>`, a
/// space, a space or `*`, the name. Other lines and malformed ones are ignored (Q-47).
fn sums_digest(sums: &str, file: &str) -> Option<String> {
    sums.lines().find_map(|line| {
        let (digest, rest) = line.split_at_checked(64)?;
        let name = rest
            .strip_prefix("  ")
            .or_else(|| rest.strip_prefix(" *"))?;
        (name == file && digest.bytes().all(|b| b.is_ascii_hexdigit())).then(|| digest.to_owned())
    })
}

/// Refuses an entry of a kind `info` has no header-only read for (Q-33, Q-47).
fn header_only(entry: &Entry) -> Result<(), IoError> {
    if entry.kind == EntryKind::Exchange {
        Ok(())
    } else {
        Err(unsupported(&format!(
            "{}: no header-only information for a {:?} entry",
            entry.location, entry.kind
        )))
    }
}

/// Refuses an entry that is not a model: a plain directory or another file.
fn fetchable(entry: &Entry) -> Result<(), IoError> {
    match entry.kind {
        EntryKind::Exchange | EntryKind::Ply | EntryKind::Checkpoint => Ok(()),
        EntryKind::Dir | EntryKind::Other => Err(invalid(format!(
            "{}: a {:?} entry is not a model file",
            entry.location, entry.kind
        ))),
    }
}

/// The local checkpoint `dir`, returned by path (D-51): nothing is read, so `progress`
/// hears once, with the two files' size (Q-38, Q-59), unless `cancel` is already set (Q-47).
// D-51, Q-59
fn local_checkpoint(
    dir: &Path,
    progress: &mut dyn FnMut(Progress),
    cancel: &AtomicBool,
) -> Result<ModelBytes, IoError> {
    if cancel.load(Ordering::Relaxed) {
        return Err(cancelled(dir.display()));
    }
    let mut size = 0;
    for name in [MANIFEST, PARAMS] {
        size += open_file(&dir.join(name))?.metadata()?.len();
    }
    progress(Progress {
        done: size,
        total: Some(size),
    });
    Ok(ModelBytes::Checkpoint(dir.to_path_buf()))
}

/// The local file `path`, read in chunks with progress and cancel (Q-47).
fn read_local(
    path: &Path,
    progress: &mut dyn FnMut(Progress),
    cancel: &AtomicBool,
) -> Result<ModelBytes, IoError> {
    let file = open_file(path)?;
    let len = file.metadata()?.len();
    read_chunked(
        file,
        Some(len),
        &path.display().to_string(),
        progress,
        cancel,
    )
}

/// The document `uri`, read through `provider` in chunks with progress and cancel.
fn read_document(
    provider: &dyn DocumentProvider,
    uri: &str,
    progress: &mut dyn FnMut(Progress),
    cancel: &AtomicBool,
) -> Result<ModelBytes, IoError> {
    let file = provider.open(uri)?;
    let len = file.metadata()?.len();
    read_chunked(file, Some(len), uri, progress, cancel)
}

/// Everything `reader` holds, read `CHUNK` bytes at a time: `cancel` is read before each
/// chunk (a cancel is E-IO `Interrupted`) and `progress` hears after each (Q-47).
fn read_chunked(
    mut reader: impl Read,
    total: Option<u64>,
    what: &str,
    progress: &mut dyn FnMut(Progress),
    cancel: &AtomicBool,
) -> Result<ModelBytes, IoError> {
    let mut bytes = Vec::with_capacity(total.map_or(0, |t| t as usize));
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(cancelled(what));
        }
        let read = (&mut reader).take(CHUNK as u64).read_to_end(&mut bytes)?;
        if read == 0 {
            break;
        }
        progress(Progress {
            done: bytes.len() as u64,
            total,
        });
    }
    if bytes.is_empty() {
        progress(Progress { done: 0, total });
    }
    Ok(ModelBytes::Owned(bytes))
}

/// Opens `path` for reading, or E-IO naming it.
fn open_file(path: &Path) -> Result<File, IoError> {
    File::open(path).map_err(|e| {
        IoError::Io(std::io::Error::new(
            e.kind(),
            format!("{}: {e}", path.display()),
        ))
    })
}

/// E-IO `Unsupported` with `message` (Q-47).
fn unsupported(message: &str) -> IoError {
    IoError::Io(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        message.to_owned(),
    ))
}

/// E-IO `InvalidInput` with `message` (Q-47).
fn invalid(message: String) -> IoError {
    IoError::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        message,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sums_digest_reads_the_line_of_the_file() {
        let a = "a".repeat(64);
        let b = "B".repeat(64);
        let sums = format!(
            "{a}  other.ply\nnot a line\n{}  m p.ply\n{b} *m p.ply\r\n{a}  m p.ply\n",
            "z".repeat(64)
        );
        // The first well-formed line naming the file, in text or binary mode, CRLF or not.
        assert_eq!(sums_digest(&sums, "m p.ply"), Some(b.clone()));
        assert_eq!(sums_digest(&sums, "other.ply"), Some(a));
        assert_eq!(sums_digest(&sums, "m"), None);
        assert_eq!(sums_digest("", "m p.ply"), None);
        assert_eq!(sums_digest(&format!("{b} m p.ply"), "m p.ply"), None);
    }

    #[test]
    fn beside_replaces_the_last_segment_as_one() {
        let url = Url::parse("http://h/d/a%20b.ply?x=1#y").unwrap();
        assert_eq!(
            beside(&url, "a b?#/.sha256").unwrap().as_str(),
            "http://h/d/a%20b%3F%23%2F.sha256"
        );
        assert_eq!(
            beside(&url, SUMS).unwrap().as_str(),
            "http://h/d/SHA256SUMS"
        );
    }
}
