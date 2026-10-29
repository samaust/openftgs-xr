# SPDX-License-Identifier: Apache-2.0
"""The crates a branch's changes can reach, so that a run tests those and not the workspace.

    affected_tests.py [-C DIR] [--base REF] [--files PATH...]
        prints `cargo test` package flags on stdout, `-p <crate>` per affected crate, one per
        line, or `--workspace` alone when everything is affected; on stderr, what brought each
        crate in

The changes are the files that differ between REF (default `main`) and the working tree: those
`git diff REF...HEAD` lists, uncommitted and untracked ones included, a rename as both its
paths. `--files` names them instead, to ask what a change to those files would reach.

A workspace member (from `cargo metadata`) is affected when:

- a file under its directory changed: its sources, shaders, tests, fixtures and `Cargo.toml`;
- it depends on an affected crate, at any depth, through normal, dev or build dependencies
  (its tests compile against them);
- a file under the shared fixtures directory, `tests/fixtures/`, changed: every crate's tests
  may read it;
- it holds the skeleton test (ALWAYS_CRATES): it walks every source file, so it always runs.

The whole workspace is affected when the root `Cargo.toml`, `Cargo.lock`,
`rust-toolchain.toml`, `.cargo/config.toml` or this script changed. A changed file outside
every crate (docs, `android/`, `tools/`) reaches no crate and is reported on stderr; the
`device` lane and the fixture tooling check those by hand.

Every choice errs towards running a test. What it can't see is a file a test finds by a name
built at run time outside its crate and outside `tests/fixtures/`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import deque
from pathlib import Path, PurePosixPath

ALWAYS_CRATES: tuple[str, ...] = ()  # the crate holding the skeleton test, once spec §2.7 names it
FIXTURES = "tests/fixtures"
WHOLE_WORKSPACE = ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "rust-toolchain", ".cargo/config.toml")
SELF = ".claude/skills/implement-without-subagents/scripts/affected_tests.py"
WORKSPACE = "--workspace"


def git(cwd: str | Path, *args: str) -> list[str]:
    out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line]


def changed_files(root: Path, base: str) -> set[str]:
    committed = git(root, "diff", "--name-only", "--no-renames", f"{base}...HEAD")
    uncommitted = git(root, "diff", "--name-only", "--no-renames", "HEAD")
    untracked = git(root, "ls-files", "--others", "--exclude-standard")
    return {*committed, *uncommitted, *untracked}


class Workspace:
    """The members of the cargo workspace at `root`: their directories and dependency edges."""

    def __init__(self, root: Path) -> None:
        out = subprocess.run(
            ["cargo", "metadata", "--format-version", "1"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        meta = json.loads(out)
        members = set(meta["workspace_members"])
        packages = [p for p in meta["packages"] if p["id"] in members]
        self.names = sorted(p["name"] for p in packages)
        self.dirs: dict[str, str] = {}  # crate -> its directory, relative to root, posix
        for p in packages:
            manifest = Path(p["manifest_path"]).resolve()
            self.dirs[p["name"]] = PurePosixPath(manifest.parent.relative_to(root.resolve())).as_posix()
        self.dependents: dict[str, set[str]] = {name: set() for name in self.names}
        for p in packages:
            for dep in p["dependencies"]:  # normal, dev and build dependencies alike
                if dep["name"] in self.dependents:
                    self.dependents[dep["name"]].add(p["name"])

    def crate_of(self, path: str) -> str | None:
        """The member whose directory holds `path`; the deepest one when crates nest."""
        best = None
        for name, directory in self.dirs.items():
            if directory in (".", "") or path == directory or path.startswith(f"{directory}/"):
                if best is None or len(directory) > len(self.dirs[best]):
                    best = name
        return best


def affected(root: Path, changed: set[str]) -> tuple[list[str], list[str]]:
    """The package flags to pass to `cargo test`, or [WORKSPACE], and the lines that say why."""
    for path in sorted(changed):
        if path in WHOLE_WORKSPACE or path == SELF:
            return [WORKSPACE], [f"the whole workspace: {path} changed"]
    try:
        ws = Workspace(root)
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError) as error:
        return [], [f"no cargo workspace at {root}: {error}"]
    via: dict[str, str] = {}  # crate -> the file or crate that brought it in
    queue: deque[str] = deque()
    for path in sorted(changed):
        p = PurePosixPath(path)
        if p.suffix == ".md":
            continue
        if path == FIXTURES or path.startswith(f"{FIXTURES}/"):
            for name in ws.names:
                via.setdefault(name, f"{path} (shared fixtures)")
                queue.append(name)
            continue
        crate = ws.crate_of(path)
        if crate is None:
            print(f"{path}: outside every crate, reaches no test", file=sys.stderr)
            continue
        if crate not in via:
            via[crate] = path
            queue.append(crate)
    while queue:
        crate = queue.popleft()
        for user in sorted(ws.dependents.get(crate, ())):
            if user not in via:
                via[user] = crate
                queue.append(user)
    for name in ALWAYS_CRATES:
        if name in ws.dependents and name not in via:
            via[name] = "always"
    chosen = [name for name in ws.names if name in via]
    why = []
    for name in chosen:
        hops = [name]
        while hops[-1] in via and via[hops[-1]] in ws.dependents and via[hops[-1]] != hops[-1]:
            hops.append(via[hops[-1]])
        why.append(" <- ".join(hops) + f" <- {via[hops[-1]]}")
    flags = [f"-p {name}" for name in chosen]
    return flags, why


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="affected_tests.py", description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-C", metavar="DIR", default=".", help="the checkout (a worktree)")
    parser.add_argument("--base", default="main", help="the branch the changes count from")
    parser.add_argument("--files", nargs="+", metavar="PATH", help="these changes instead")
    args = parser.parse_args()
    root = Path(git(args.C, "rev-parse", "--show-toplevel")[0])
    changed = set(args.files) if args.files else changed_files(root, args.base)
    flags, why = affected(root, changed)
    print("\n".join(why), file=sys.stderr)
    print("\n".join(flags))
    return 0


if __name__ == "__main__":
    sys.exit(main())
