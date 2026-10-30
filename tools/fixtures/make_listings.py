# SPDX-License-Identifier: Apache-2.0
"""The three listing bodies of U-27 and U-31 (§1.5 Sources, §1.11 `index.json`, D-31).

The bodies are written by hand from the documented formats (spec Appendix A row 35: nginx's
`autoindex_format json`, Apache's `mod_autoindex` HTML; §1.11 for the player's own
`ftgsxr-index`), not captured from a running server; this script only fills in the sizes and
digests of the files they list. All three describe one directory, tests/fixtures/listing/models/,
which holds the listed files:

- `probe p1.safetensors`: a copy of probes/P1.safetensors (a name that needs percent-encoding);
- `snapshot.ply`: a copy of the probe PLY snapshot;
- `tiny/`: a checkpoint directory holding copies of checkpoint/tiny/manifest.json and
  params.safetensors (the two files the player fetches);
- `notes.txt`: a stray file that is not a model;
- `probe p1.sha256`: the probe's digest, a `<name>.sha256` sidecar (§1.5 Download cache; Q-47);
- `SHA256SUMS`: a `sha256sum` sidecar with a wrong digest for `snapshot.ply`, a line for
  `notes.txt`, a malformed line and no line for the probe (Q-47).

The nginx and Apache bodies also list `.hidden.safetensors`, a dot-name a listing client skips
(no such file exists), and the Apache body has its parent-directory link and column-sort links.
The bodies are listing/index.json, listing/nginx.json and listing/apache.html. They do not list
the two sidecars, so the listings U-27 reads are unchanged by them.
"""

import hashlib
import html
import json
import shutil
from pathlib import Path
from urllib.parse import quote

from fixture_common import FIXTURES

LISTING = FIXTURES / "listing"
MODELS = LISTING / "models"
MTIME_NGINX = "Thu, 29 Oct 2026 12:00:00 GMT"  # nginx's RFC 1123 date
MTIME_APACHE = "2026-10-29 12:00"  # mod_autoindex's default date format
PARENT = "/listing/"  # the parent link of the Apache listing, as an absolute path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_files() -> None:
    shutil.rmtree(MODELS, ignore_errors=True)
    (MODELS / "tiny").mkdir(parents=True)
    shutil.copyfile(FIXTURES / "probes" / "P1.safetensors", MODELS / "probe p1.safetensors")
    shutil.copyfile(FIXTURES / "probes" / "R2_overlap_deg3.tau1.3.ply", MODELS / "snapshot.ply")
    for name in ("manifest.json", "params.safetensors"):
        shutil.copyfile(FIXTURES / "checkpoint" / "tiny" / name, MODELS / "tiny" / name)
    (MODELS / "notes.txt").write_text(
        "Not a model: a stray file of the listing fixture.\n", encoding="utf-8"
    )
    write_sidecars()


def write_sidecars() -> None:
    """The digest files beside the models (§1.5 Download cache; Q-47): the probe's right digest
    in `probe p1.sha256`, and in `SHA256SUMS` the probe's digest given for `snapshot.ply` (a
    wrong one, in binary mode), with a line for another file and a malformed line."""
    probe = sha256(MODELS / "probe p1.safetensors")
    (MODELS / "probe p1.sha256").write_text(f"{probe}  probe p1.safetensors\n", encoding="utf-8")
    lines = [
        f"{sha256(MODELS / 'notes.txt')}  notes.txt",
        "not a sha256sum line",
        f"{probe} *snapshot.ply",
    ]
    (MODELS / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def size(name: str) -> int:
    path = MODELS / name
    if path.is_dir():
        return sum(p.stat().st_size for p in sorted(path.iterdir()))
    return path.stat().st_size


def index_json() -> str:
    """§1.11 `ftgsxr-index` v1: paths are URL references relative to the index's URL; a
    checkpoint's size is the sum of the two files the player fetches."""
    entries = [
        {
            "name": "probe p1",
            "path": quote("probe p1.safetensors"),
            "kind": "ftgs-if",
            "size": size("probe p1.safetensors"),
            "sha256": sha256(MODELS / "probe p1.safetensors"),
        },
        {
            "name": "snapshot",
            "path": "snapshot.ply",
            "kind": "ply",
            "size": size("snapshot.ply"),
            "sha256": sha256(MODELS / "snapshot.ply"),
        },
        {"name": "tiny", "path": "tiny/", "kind": "ftgs-checkpoint", "size": size("tiny")},
    ]
    body = {"format": "ftgsxr-index", "version": 1, "entries": entries}
    return json.dumps(body, indent=2) + "\n"


# The directory as a server lists it: (name, is_directory); nginx lists directories first.
LISTED = [
    ("tiny", True),
    (".hidden.safetensors", False),
    ("notes.txt", False),
    ("probe p1.safetensors", False),
    ("snapshot.ply", False),
]


def listed_size(name: str) -> int:
    return 1092 if name == ".hidden.safetensors" else size(name)  # the dot-name has no file


def nginx_json() -> str:
    """nginx's `autoindex_format json`: an array of {name, type, mtime, size}, size for files
    only, names unescaped."""
    rows = []
    for name, is_dir in LISTED:
        row = {"name": name, "type": "directory" if is_dir else "file", "mtime": MTIME_NGINX}
        if not is_dir:
            row["size"] = listed_size(name)
        rows.append(
            "{ " + ", ".join(f"{json.dumps(k)}:{json.dumps(v)}" for k, v in row.items()) + " }"
        )
    return "[\n" + ",\n".join(rows) + "\n]\n"


def human(n: int) -> str:
    """mod_autoindex's abbreviated size."""
    if n < 973:
        return f"{n}"
    for unit in ("K", "M", "G"):
        n /= 1024.0
        if n < 9.95:
            return f"{n:.1f}{unit}"
        if n < 973:
            return f"{n:.0f}{unit}"
    return f"{n:.0f}T"


def apache_html() -> str:
    """Apache's `mod_autoindex` with FancyIndexing and HTMLTable: a table of anchors, the
    column-sort links, the parent-directory link, then one row per entry, hrefs percent-encoded."""
    head = [
        '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">',
        "<html>",
        " <head>",
        "  <title>Index of /listing/models</title>",
        " </head>",
        " <body>",
        "<h1>Index of /listing/models</h1>",
        "  <table>",
        '   <tr><th valign="top"><img src="/icons/blank.gif" alt="[ICO]"></th>'
        '<th><a href="?C=N;O=D">Name</a></th><th><a href="?C=M;O=A">Last modified</a></th>'
        '<th><a href="?C=S;O=A">Size</a></th><th><a href="?C=D;O=A">Description</a></th></tr>',
        '   <tr><th colspan="5"><hr></th></tr>',
        f'<tr><td valign="top"><img src="/icons/back.gif" alt="[PARENTDIR]"></td>'
        f'<td><a href="{PARENT}">Parent Directory</a></td><td>&nbsp;</td>'
        f'<td align="right">  - </td><td>&nbsp;</td></tr>',
    ]
    rows = []
    for name, is_dir in sorted(LISTED, key=lambda e: e[0].lower()):
        href = quote(name) + ("/" if is_dir else "")
        label = html.escape(name + ("/" if is_dir else ""))
        icon, alt = ("folder.gif", "[DIR]") if is_dir else ("unknown.gif", "[   ]")
        shown = "  - " if is_dir else human(listed_size(name))
        rows.append(
            f'<tr><td valign="top"><img src="/icons/{icon}" alt="{alt}"></td>'
            f'<td><a href="{href}">{label}</a></td><td align="right">{MTIME_APACHE}  </td>'
            f'<td align="right">{shown}</td><td>&nbsp;</td></tr>'
        )
    tail = [
        '   <tr><th colspan="5"><hr></th></tr>',
        "</table>",
        "<address>Apache/2.4 Server at localhost Port 80</address>",
        "</body></html>",
    ]
    return "\n".join(head + rows + tail) + "\n"


def main() -> None:
    copy_files()
    (LISTING / "index.json").write_text(index_json(), encoding="utf-8")
    (LISTING / "nginx.json").write_text(nginx_json(), encoding="utf-8")
    (LISTING / "apache.html").write_text(apache_html(), encoding="utf-8")


if __name__ == "__main__":
    main()
