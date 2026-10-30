# The download cache in `ftgsxr_io::http`

How `HttpClient` keeps the download cache: the files of a cache entry, the three states an entry can be in, how a download moves between them, and how `prune` keeps the cache to its size limit. The spec items are §1.5 (Sources, HTTP; Download cache), §1.11 (Locations, the write rule, the versioning rule, Cache entries), §1.15 (E-NET-FETCH, E-NET-CHECKSUM), D-20 and U-28; this doc explains them and never overrides them.

## The client

One `HttpClient` holds one ureq agent, shared by every request it makes, over rustls with the `ring` provider and the webpki roots (D-20). The agent treats every status as data, so each method decides what a 206, a 304 or a 416 means for it, and anything it did not ask for becomes E-NET-FETCH with the status in its text. It sends no `Accept-Encoding`: a compressed body would make byte offsets meaningless for a resume and for a header-only read.

Three reads, three contracts:

- **`get_text`** fetches a listing body (§1.5 Sources, D-31) fresh every time; listings are relisted on refresh, so they never enter the cache (Q-44).
- **`get_range`** is the header-only read of a library entry (§1.5 Library): one `Range` request, so reading a model file's length prefix and JSON header costs two small requests and never the whole file. A server that ignores `Range` is read only as far as the slice needs (Q-44).
- **`download`** goes through the download cache, below.

Text and range reads run in parallel; `download` and `prune` take one lock, so downloads run one at a time as §1.5 asks and a prune never removes a download in progress (Q-45).

## A cache entry

Each URL owns one directory, `http/<sha256 of the URL>/` under the cache directory `HttpClient::new` receives (§1.11 Locations: the `<cache dir>` root, `AppPaths.cache`). The URL hashed is its serialization by the url crate, so two spellings the url crate normalizes alike share an entry (Q-42).

| File | Holds |
| --- | --- |
| `file` | the complete model file (§1.11) |
| `meta.json` | its record: §1.11's keys, with the `format` and `version` the versioning rule asks of every player file (Q-42) |
| `file.part` | the download in progress, the temporary sibling of §1.11's write rule |
| `meta.part.json` | the download in progress's record, with its validators and no digest yet (Q-42) |

Every record is written to a temporary sibling and renamed (§1.11). A record that does not parse, names another format or major version, or disagrees with its file's size counts as absent, so the entry is fetched again rather than trusted (Q-42).

## Three states

- **Complete**: `file` with a `meta.json` that carries its digest. `download` revalidates it with `If-None-Match` and `If-Modified-Since` (§1.5). A 304 reuses it and refreshes `fetched_at`; a 200 means the server's file changed, and the new one is downloaded beside the old one, which stays usable until the new one is complete (Q-42, §1.15 E-NET-FETCH: "cached file kept if complete").
- **Partial**: `file.part` with its own record. `download` resumes it with `Range` from the bytes held and `If-Range` with the record's ETag, or its Last-Modified date when the server gave no ETag (§1.5). A 206 that starts where the partial ends appends; a 200, which is what a server answers when the validator no longer matches, restarts from the first byte; a 416 or a 206 starting elsewhere discards the partial and asks once more from scratch (Q-43). A partial without any validator cannot be resumed safely and restarts: this is §1.5's "a file whose server supports neither is downloaded whole".
- **Absent**: a plain GET.

When an entry has both, the partial wins: it was started after the complete file was found out of date.

## A download's life

The body is streamed into `file.part` chunk by chunk. Before each chunk `cancel` is read, and after each one `progress` hears the bytes held and the size the server announced, counting the bytes a resume started from. The SHA-256 is computed while streaming; a resume first hashes the bytes already held, since a digest in progress cannot survive the process that started it.

The download ends in one of four ways:

- **Complete.** The size must agree with the server's; then, when the caller gave an `expected_sha256` (from the index entry, a `SHA256SUMS` or a `<name>.sha256`, §1.5), the digest must match, compared without regard to case. `file.part` is renamed over `file` and the record over `meta.json`, in that order, and the partial's record is removed. The returned path is always the entry's `file`.
- **Checksum mismatch.** The partial and its record are deleted and the call returns E-NET-CHECKSUM (§1.5, §1.15). A complete cached file whose recorded digest disagrees with the expected one is not offered for revalidation in the first place: the expected digest wins and the file is fetched again (Q-43).
- **Failure part-way.** A body that breaks off keeps every byte received in `file.part`, and the call returns E-NET-FETCH without retrying itself; the next call resumes. That is §1.15's retry button, and what U-28's truncation exercises through the fixture server's broken chunk (Q-40).
- **Cancel.** `cancel` is read before the request and before every chunk read. The partial is kept for a later resume and the call returns E-IO, an `IoError::Io` of kind `Interrupted` with the URL in its text, as a cancelled local or document `fetch` does, so a caller recognises every cancel by its kind (Q-43, Q-47); a cancel already set before the call sends no request.

## Pruning

`prune` sees only the `http/` entries: thumbnails and logs share the cache directory but are written by others (Q-45). It measures each entry as the bytes of all its files, orders entries by `fetched_at`, which every successful `download`, a 304 included, rewrites and which therefore is the entry's last use (Q-42), and removes whole entries, oldest first, until the total is within the limit `HttpClient::new` was given. An entry without a readable record sorts first.

It never removes an entry whose `file` is in the set `set_in_use` last gave (Q-4): the application passes the paths `download` returned for the models loaded now, before it prunes, and paths are compared canonicalized. If the files in use alone exceed the limit, `prune` frees what it can and leaves them. `download` never prunes by itself, since only the application knows what is loaded.

## Q-n rows relied on

- **Q-4** `set_in_use`, the public-API addition that tells `prune` which files are loaded.
- **Q-42** the entry's files and records: the `<cache dir>` root, the URL's serialization as the key, `format` and `version` in the record, `fetched_at` as the last use, the partial and its own record.
- **Q-43** what `download` does where §1.5 is silent: failures part-way, cancel, a resume answered with 200 or 416, the expected digest against the cache's record, a malformed digest, timeouts.
- **Q-44** `get_range` on a short file or a server without ranges; `get_text` uncached.
- **Q-45** what `prune` counts and removes, and the one lock it shares with `download`.
- **Q-39** and **Q-40** the fixture server U-28 runs against: its validators, its ranges and its truncation.
