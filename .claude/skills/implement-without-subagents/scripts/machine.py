# SPDX-License-Identifier: Apache-2.0
"""The machine's CPU, GPU and headset budget, shared by every session on this checkout.

Sessions that run `implement-without-subagents` in parallel have no coordinator, so they keep
their reservations in a ledger: one JSON file per run, holding or waiting for the machine, in
`.scratch/machine/` at the root of the main checkout, which every worktree shares. A run's entry
lives as long as the process that holds it: an entry whose process has died is ignored.

    machine.py status [-C DIR] [-- COMMAND...]
        the physical cores, the GPUs, the load, every run holding or waiting for the machine,
        what is free, who holds the headset and, with COMMAND, that command's past runs on
        this machine
    machine.py run (--threads N | --exclusive) [--gpu] [--headset] [--wait]
                   [--estimate SECONDS] [-C DIR] -- COMMAND...
        runs COMMAND with N threads (CARGO_BUILD_JOBS, RUST_TEST_THREADS and RAYON_NUM_THREADS
        for cargo and the tests; OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS and
        TORCHINDUCTOR_COMPILE_THREADS for the Python tooling that drives openftgs), every GPU
        with --gpu, the headset with --headset, or the whole machine at the default thread
        count with --exclusive, and exits with COMMAND's exit code. When that budget isn't free
        it exits 75 and prints the ledger, or with --wait takes its turn.

The headset (the `device` lane's Quest, reached through `adb`) is held by one run at a time:
every `dv_` test, and anything else that installs, launches or stops the app on it, runs with
--headset, since a second run's install replaces the first's APK under it. --exclusive leaves
the headset alone: it is the host's timings it protects. A `dv_` test found running outside
the ledger (a process whose arguments hold `--ignored` and a `dv_` filter, not started by a
run that holds the headset) counts as holding the headset, so a --headset run waits for it
rather than install over it, and `status` names it.

Waiters start in arrival order: a run that fits starts at once unless a waiter ahead of it is
short of a resource it would take. Each finished run goes into `history.jsonl` beside the
entries, its command written with its directory as `<cwd>`, so that the same command run in
another worktree finds it.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

BUSY = 75  # EX_TEMPFAIL: the budget isn't free; cargo's own codes are 0, 1 and 101
POLL_S = 2.0
THREAD_VARIABLES = (
    "CARGO_BUILD_JOBS",  # rustc jobs cargo runs at once
    "RUST_TEST_THREADS",  # the test harness's threads
    "RAYON_NUM_THREADS",  # rayon pools in tests and tools
    "OMP_NUM_THREADS",  # the Python tooling that drives openftgs: PyTorch's OpenMP
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",  # NumPy's and SciPy's BLAS
    "TORCHINDUCTOR_COMPILE_THREADS",  # torch.compile's compile workers
)


def ledger_dir() -> Path:
    """`.scratch/machine/` at the root of the main checkout, whichever worktree runs this."""
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    path = Path(common).parent / ".scratch" / "machine"
    (path / "runs").mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def locked(ledger: Path) -> Iterator[None]:
    with open(ledger / "lock", "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def physical_cores() -> int:
    """Distinct (package, core) pairs in Linux's CPU topology; the logical count otherwise."""
    cores = set()
    for cpu in Path("/sys/devices/system/cpu").glob("cpu[0-9]*"):
        try:
            package = (cpu / "topology" / "physical_package_id").read_text().strip()
            core = (cpu / "topology" / "core_id").read_text().strip()
        except OSError:
            continue
        cores.add((package, core))
    return len(cores) or os.cpu_count() or 1


def gpus() -> list[str]:
    try:
        listing = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in listing.splitlines() if line.strip()]


def start_ticks(pid: int) -> int | None:
    """The process's start time, which tells a live process from a later one with its PID."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    return int(stat.rsplit(")", 1)[1].split()[19])  # field 22, after the command's name


def entries(ledger: Path) -> list[dict[str, Any]]:
    """The live entries in arrival order; the files of dead processes are removed."""
    live = []
    for path in (ledger / "runs").glob("*.json"):
        try:
            entry = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if start_ticks(entry["pid"]) == entry["start_ticks"]:
            live.append(entry)
        else:
            path.unlink(missing_ok=True)
    return sorted(live, key=lambda e: (e["arrived"], e["pid"]))


def write_entry(ledger: Path, entry: dict[str, Any]) -> None:
    path = ledger / "runs" / f"{entry['pid']}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(entry))
    tmp.replace(path)


def is_dv_run(argv: list[str]) -> bool:
    """Whether the arguments run `dv_` tests: `--ignored` and a filter naming them."""
    return "--ignored" in argv and any(a.startswith("dv_") or "::dv_" in a for a in argv)


def processes() -> dict[int, tuple[int, list[str]]]:
    """Each live process's parent and arguments, from /proc."""
    out = {}
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
            stat = (proc / "stat").read_text()
        except OSError:
            continue
        ppid = int(stat.rsplit(")", 1)[1].split()[1])
        out[int(proc.name)] = (ppid, [a.decode(errors="replace") for a in argv if a])
    return out


def foreign_headset_runs(live: list[dict[str, Any]]) -> list[tuple[int, str]]:
    """The `dv_` test processes outside the ledger's headset (pid, arguments), the outermost of
    a chain only: not a ledger entry's own process (whose arguments hold its command, running or
    waiting) nor a descendant of a run holding the headset. A `dv_` test that a run without
    --headset started is outside it."""
    entry_pids = {e["pid"] for e in live}
    holders = {e["pid"] for e in live if e["state"] == "running" and e.get("headset")}
    procs = processes()

    def ancestors(pid: int) -> Iterator[int]:
        seen = set()
        while pid in procs and pid not in seen and pid > 1:
            seen.add(pid)
            pid = procs[pid][0]
            yield pid

    found = {
        pid
        for pid, (_, argv) in procs.items()
        if pid != os.getpid()
        and pid not in entry_pids
        and is_dv_run(argv)
        and not holders & set(ancestors(pid))
    }
    return sorted(
        (pid, shlex.join(procs[pid][1]))
        for pid in found
        if not found & set(ancestors(pid))
    )


def fits(
    entry: dict[str, Any],
    live: list[dict[str, Any]],
    cores: int,
    n_gpus: int,
    headset_foreign: bool = False,
) -> bool:
    """Whether `entry` may start now: the running entries' budgets subtracted, then each waiter
    ahead of it either taking its budget (it fits, and starts at its next poll) or keeping what
    it is short of (the free cores up to its own count, the GPUs, the headset, or the whole
    machine). A `dv_` run outside the ledger (`headset_foreign`) holds the headset."""
    free, gpu_free, headset_free = cores, n_gpus > 0, not headset_foreign
    for other in live:
        if other["state"] == "running":
            free -= other["threads"]
            gpu_free = gpu_free and not other["gpu"]
            headset_free = headset_free and not other.get("headset")
    for other in live:
        if other["state"] != "waiting" or other["pid"] == entry["pid"]:
            continue
        if (other["arrived"], other["pid"]) > (entry["arrived"], entry["pid"]):
            break
        takes_headset = other.get("headset", False)
        if (
            other["threads"] <= free
            and (gpu_free or not other["gpu"])
            and (headset_free or not takes_headset)
        ):
            free -= other["threads"]
        else:
            free -= min(free, other["threads"])
        gpu_free = gpu_free and not other["gpu"]
        headset_free = headset_free and not takes_headset
    return (
        entry["threads"] <= free
        and (gpu_free or not entry["gpu"])
        and (headset_free or not entry.get("headset"))
    )


def duration(seconds: float) -> str:
    minutes, rest = divmod(round(seconds), 60)
    return f"{minutes} m {rest:02d} s" if minutes else f"{rest} s"


def budget(entry: dict[str, Any]) -> str:
    if entry["exclusive"]:
        return "the whole machine"
    n = entry["threads"]
    gpu = " + GPU" if entry["gpu"] else ""
    headset = " + headset" if entry.get("headset") else ""
    return f"{n} thread{'s' if n != 1 else ''}{gpu}{headset}"


def history(ledger: Path, command: str) -> list[dict[str, Any]]:
    try:
        lines = (ledger / "history.jsonl").read_text().splitlines()
    except OSError:
        return []
    runs = [json.loads(line) for line in lines if line.strip()]
    return [run for run in runs if run["command"] == command]


def report(ledger: Path, command: str | None = None) -> str:
    cores, devices = physical_cores(), gpus()
    live = entries(ledger)
    load = os.getloadavg()
    out = [
        f"cores: {cores} physical ({os.cpu_count()} logical)",
        f"GPUs: {'; '.join(devices) if devices else 'none'}",
        f"load: {load[0]:.1f} / {load[1]:.1f} / {load[2]:.1f} (1, 5, 15 min)",
    ]
    now = time.time()
    for state in ("running", "waiting"):
        rows = [e for e in live if e["state"] == state]
        out.append(f"{state}: {len(rows) or 'none'}")
        for e in rows:
            since = e["started"] if state == "running" else e["arrived"]
            past = history(ledger, e["command"])
            estimate = e["estimate"] or (past[-1]["seconds"] if past else None)
            expected = f", estimate {duration(estimate)}" if estimate else ""
            out.append(
                f"  {e['label']}: {budget(e)}, {duration(now - since)} so far{expected}: "
                f"{e['command']}"
            )
    held = sum(e["threads"] for e in live if e["state"] == "running")
    gpu_held = any(e["gpu"] for e in live if e["state"] == "running")
    gpu_state = "none" if not devices else ("held" if gpu_held else "free")
    waiting = any(e["state"] == "waiting" for e in live)
    claims = ", before what the waiting runs ahead will take" if waiting else ""
    out.append(f"free: {max(cores - held, 0)} cores, GPU {gpu_state}{claims}")
    holder = next(
        (e for e in live if e["state"] == "running" and e.get("headset")), None
    )
    foreign = foreign_headset_runs(live)
    if holder is not None:
        out.append(f"headset: held by {holder['label']}")
    elif not foreign:
        out.append("headset: free")
    for pid, argv in foreign:
        out.append(f"headset: held outside the ledger by pid {pid}: {argv}")
    if load[0] > held + 2:
        out.append(
            f"note: the 1-min load is {load[0]:.1f} against {held} threads in the ledger: "
            "work outside it is running"
        )
    if command is not None:
        past = history(ledger, command)[-5:]
        out.append(f"past runs of {command}: {len(past) or 'none'}")
        for run in past:
            out.append(f"  {budget(run)}: {duration(run['seconds'])}, exit {run['exit']}")
    return "\n".join(out)


class Stopped(Exception):
    pass


def run(args: argparse.Namespace) -> int:
    ledger = ledger_dir()
    cores, n_gpus = physical_cores(), len(gpus())
    if args.gpu and n_gpus == 0:
        print("machine.py: --gpu on a machine without a GPU", file=sys.stderr)
        return 2
    threads = cores if args.exclusive else args.threads
    if not 1 <= threads <= cores:
        print(f"machine.py: --threads must be from 1 to {cores}, the cores", file=sys.stderr)
        return 2
    cwd = Path(args.C).resolve() if args.C else Path.cwd()
    command = shlex.join(args.command).replace(str(cwd), "<cwd>")
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=cwd, capture_output=True, text=True, check=False
    ).stdout.strip()
    pid = os.getpid()
    entry: dict[str, Any] = {
        "pid": pid,
        "start_ticks": start_ticks(pid),
        "label": branch or str(cwd),
        "state": "waiting",
        "threads": threads,
        "gpu": args.gpu or (args.exclusive and n_gpus > 0),
        "headset": args.headset,
        "exclusive": args.exclusive,
        "command": command,
        "cwd": str(cwd),
        "arrived": time.time(),
        "started": None,
        "estimate": args.estimate,
    }
    path = ledger / "runs" / f"{pid}.json"
    what = budget(entry)
    child: subprocess.Popen[bytes] | None = None

    def stop(signum: int, _frame: object) -> None:
        if child is None:
            raise Stopped(signum)
        child.send_signal(signum)  # the child exits; `wait` below returns and the entry goes

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, stop)
    try:
        while True:
            with locked(ledger):
                live = entries(ledger)
                foreign = bool(args.headset and foreign_headset_runs(live))
                if fits(entry, live, cores, n_gpus, foreign):
                    entry |= {"state": "running", "started": time.time()}
                    write_entry(ledger, entry)
                    break
                if not args.wait:
                    print(f"machine.py: {what} isn't free", file=sys.stderr)
                    print(report(ledger, command), file=sys.stderr)
                    return BUSY
                if not path.exists():
                    write_entry(ledger, entry)  # queued, in arrival order
            time.sleep(POLL_S)
        env = dict(os.environ)
        for name in THREAD_VARIABLES:
            if args.exclusive:
                env.pop(name, None)  # the default thread count
            else:
                env[name] = str(threads)
        print(f"machine.py: running with {what}", file=sys.stderr)
        child = subprocess.Popen(args.command, cwd=cwd, env=env)
        code = child.wait()
        if code < 0:
            code = 128 - code  # killed by a signal, as a shell reports it
        seconds = time.time() - entry["started"]
        with locked(ledger), open(ledger / "history.jsonl", "a") as log:
            keys = ("command", "label", "threads", "gpu", "headset", "exclusive")
            done = {key: entry[key] for key in keys}
            done |= {"seconds": round(seconds, 1), "exit": code, "ended": time.time()}
            log.write(json.dumps(done) + "\n")
        print(f"machine.py: done in {duration(seconds)}, exit {code}", file=sys.stderr)
        return code
    except Stopped as stopped:
        return 128 + int(stopped.args[0])
    finally:
        with locked(ledger):
            path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="machine.py", description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="action", required=True)
    status = sub.add_parser("status", help="print the ledger")
    status.add_argument("-C", metavar="DIR", help="the directory COMMAND would run in")
    status.add_argument("command", nargs="*", help="a command whose past runs to list")
    go = sub.add_parser("run", help="run a command within a budget")
    budget = go.add_mutually_exclusive_group(required=True)
    budget.add_argument("--threads", type=int, help="the run's CPU threads")
    budget.add_argument(
        "--exclusive", action="store_true", help="the whole machine, at the default threads"
    )
    go.add_argument("--gpu", action="store_true", help="every GPU, for the gpu lane")
    go.add_argument(
        "--headset", action="store_true", help="the headset, for the device lane's dv_ tests"
    )
    go.add_argument("--wait", action="store_true", help="queue until the budget is free")
    go.add_argument("--estimate", type=float, help="the run's expected seconds, for others")
    go.add_argument("-C", metavar="DIR", help="run COMMAND in DIR (a worktree)")
    go.add_argument("command", nargs="+", help="the command, after --")
    args = parser.parse_args()
    if args.action == "status":
        cwd = str(Path(args.C).resolve() if args.C else Path.cwd())
        command = shlex.join(args.command).replace(cwd, "<cwd>") if args.command else None
        print(report(ledger_dir(), command))
        return 0
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
