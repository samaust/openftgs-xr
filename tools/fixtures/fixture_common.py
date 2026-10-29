# SPDX-License-Identifier: Apache-2.0
"""Shared helpers of the fixture scripts (§2.7 Fixtures, D-29).

Dev tooling: it imports openftgs and its harness (Apache-2.0, the clone at `$OPENFTGS_REPO`) and
nothing from this workspace. Every openftgs command runs on the CPU (the reference backend) with
a fixed thread count, so a second run reproduces the same bytes.
"""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

THREADS = 6  # the fixed intra-op thread count of every run, so reruns match bit for bit

ROOT = Path(__file__).resolve().parents[2]  # the workspace root
FIXTURES = ROOT / "tests" / "fixtures"
# Intermediates (training runs, ground-truth renders), never committed: $FIXTURES_WORK, else the
# gitignored .scratch/fixtures-work of the workspace.
WORK = Path(os.environ.get("FIXTURES_WORK", ROOT / ".scratch" / "fixtures-work"))

PRODUCER = "openftgs-xr tools/fixtures"  # the `producer` of the models these scripts write

# The commands the scripts ran, in order, as the README records them (cwd: tests/fixtures).
COMMANDS: list[str] = []


def environment() -> dict[str, str]:
    """The environment of every openftgs command: no CUDA device (so the CPU and the reference
    backend), a fixed thread count."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = str(THREADS)
    env["MKL_NUM_THREADS"] = str(THREADS)
    return env


def pin_threads() -> None:
    """The same settings for this process, which imports openftgs directly."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(THREADS)


def program(name: str) -> str:
    """An openftgs console script of the interpreter's environment (`ftgs`, `ftgs-adapter`)."""
    path = Path(sys.executable).parent / name
    if not path.exists():
        raise SystemExit(f"{path} not found: run the fixture scripts with openftgs's Python")
    return str(path)


def run(name: str, *args: str, cwd: Path = FIXTURES, record: bool = True) -> str:
    """Run an openftgs program with `args` in `cwd`, record the command and return its stdout."""
    if record:
        shown = [name, *args]
        if cwd != FIXTURES:
            shown = ["cd", os.path.relpath(cwd, FIXTURES), "&&", *shown]
        text = shlex.join(shown).replace("'&&'", "&&")
        COMMANDS.append(text.replace(str(WORK), "$FIXTURES_WORK"))
    done = subprocess.run(
        [program(name), *args], cwd=cwd, env=environment(), capture_output=True, text=True
    )
    if done.returncode != 0:
        raise SystemExit(f"{name} {' '.join(args)} failed ({done.returncode}):\n{done.stderr}")
    return done.stdout


def rel(path: Path) -> str:
    """`path` relative to tests/fixtures, POSIX form; absolute when it lies outside them (the
    intermediates under WORK)."""
    if not Path(path).resolve().is_relative_to(FIXTURES):
        return str(path)
    return Path(os.path.relpath(path, FIXTURES)).as_posix()


def exchange_model(
    arrays: dict[str, np.ndarray],
    sh_degree: int,
    tau0: float,
    tau1: float,
    *,
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
):  # -> openftgs.types.ExchangeModel
    """An openftgs ExchangeModel from numpy arrays in the `ftgs-if` dtypes."""
    from openftgs.types import ExchangeModel

    return ExchangeModel(
        arrays={k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in arrays.items()},
        sh_degree=sh_degree,
        tau0=float(tau0),
        tau1=float(tau1),
        time_unit="s",
        producer=PRODUCER,
        background=background,
    )


def write_model(path: Path, model) -> None:  # model: openftgs.types.ExchangeModel
    """Write `model` with openftgs's own writer (`io.write_exchange`: `render_conventions`
    included, canonical unit quaternions checked)."""
    from openftgs.io import write_exchange

    path.parent.mkdir(parents=True, exist_ok=True)
    write_exchange(model, path)


def render_reference(model: Path, request: Path, out: Path) -> None:
    """`ftgs render MODEL --request REQUEST --out OUT --backend reference`, paths relative to
    tests/fixtures when they lie under it."""
    run(
        "ftgs",
        "render",
        rel(model),
        "--request",
        rel(request),
        "--out",
        rel(out),
        "--backend",
        "reference",
    )


def save_commands(path: Path) -> None:
    """Append this script's commands to the log the entry script turns into the README."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for command in COMMANDS:
            f.write(json.dumps(command) + "\n")
    COMMANDS.clear()
