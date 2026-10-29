# SPDX-License-Identifier: Apache-2.0
"""The pre-posed merged models of R-7 and R-8 (§2.7, §3.2; Q-12).

A merged model holds the rows of every instance of a multi-instance scene with each instance's
pose P = (t, q, s) baked in (E5): position′ = t + s·R(q)·position, velocity′ = s·R(q)·velocity,
scale′ = s·scale, rotation′ = q ⊗ rotation (canonical sign), time centers, durations and
opacities unchanged, and the SH coefficients rotated so that the colour seen along a world
direction d is the instance's colour along R(q)⁻¹·d (E4's eye in model space). openftgs's
reference backend renders the merged model; the player renders the instances with their poses,
and R-7 and R-8 compare the two.

Writes under tests/fixtures/:
- merged/R7_instance_a.safetensors, merged/R7_instance_b.safetensors: P7's two rows, one model
  each; merged/R7_poses.json: their poses;
- merged/R8_poses.json: the pose of R-8's one instance, probes/R2_overlap_deg3.safetensors
  (two degree-3 primitives, written by make_probes.py, which runs first);
- probes/R7_merged.* and probes/R8_merged.*: the merged models, their requests (R-7: eight
  times; R-8: the 32 items of R2_overlap_deg3's ring) and openftgs's reference renders.

The pose files hold, per instance, `model` (relative to the pose file), `translation`,
`rotation` as (x, y, z, w) like the player's `poses.json` (§1.11, §1.14) and `scale`.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
from fixture_common import (
    FIXTURES,
    PRODUCER,
    WORK,
    exchange_model,
    pin_threads,
    render_reference,
    save_commands,
    write_model,
)

PROBES_DIR = FIXTURES / "probes"
MERGED_DIR = FIXTURES / "merged"
TAU0, TAU1 = 0.0, 1.9  # §3.2


# Q-12: the poses. Each is (translation, unit quaternion (w, x, y, z), scale).
def axis_angle(axis: tuple[float, float, float], degrees: float) -> np.ndarray:
    a = np.asarray(axis, np.float64)
    a = a / np.linalg.norm(a)
    half = math.radians(degrees) / 2.0
    return np.concatenate([[math.cos(half)], math.sin(half) * a])


R7_POSES = {
    "R7_instance_a": (np.array([0.04, 0.0, -0.03]), axis_angle((1, 0, 0), 30.0), 1.0),
    "R7_instance_b": (np.array([-0.03, 0.0, 0.02]), axis_angle((0, 0, 1), 20.0), 1.2),
}
R7_TIMES = np.linspace(TAU0, TAU1, 8)  # R-7: eight times, none within 0.02 of P7's depth tie
# R-8: rotated 90° about y and scaled 2, no translation.
R8_POSE = (np.zeros(3), axis_angle((0, 1, 0), 90.0), 2.0)
R8_SOURCE = "R2_overlap_deg3"


def rotation_matrix(q: np.ndarray) -> np.ndarray:
    """R(q) of a unit quaternion (w, x, y, z), acting on column vectors."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The Hamilton product a ⊗ b of (w, x, y, z) quaternions, row-wise over b."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b.T
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=1,
    )


def canonical(q: np.ndarray) -> np.ndarray:
    """Unit quaternions with w ≥ 0, and where w = 0 the first nonzero component positive
    (openftgs Q-24, §1.2)."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    first = np.argmax(q != 0, axis=1)
    sign = np.where(q[np.arange(len(q)), first] < 0, -1.0, 1.0)
    return q * sign[:, None]


def sh_rotation(rotation: np.ndarray, degree: int) -> np.ndarray:
    """M with Σ_m (M c)_m Y_m(d) = Σ_m c_m Y_m(R⁻¹ d) for every direction d, in openftgs's
    basis order and signs (`openftgs.sh.sh_basis`): fitted by least squares over 512 random
    directions in float64, exact up to rounding since every band is closed under rotation."""
    from openftgs.sh import sh_basis

    rng = np.random.default_rng(7)
    d = rng.normal(size=(512, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    basis = sh_basis(torch.from_numpy(d), degree).numpy()
    rotated = sh_basis(torch.from_numpy(d @ rotation), degree).numpy()  # rows R⁻¹ d = dᵀ R
    m, *_ = np.linalg.lstsq(basis, rotated, rcond=None)
    assert np.abs(basis @ m - rotated).max() < 1e-12
    return m


def posed(arrays: dict[str, np.ndarray], degree: int, pose) -> dict[str, np.ndarray]:
    """An instance's rows with its pose baked in (E5, E4's eye)."""
    t, q, s = pose
    r = rotation_matrix(q)
    m = sh_rotation(r, degree)
    sh = arrays["sh"].astype(np.float64)
    return {
        "position": (t + s * arrays["position"].astype(np.float64) @ r.T).astype(np.float32),
        "time_center": arrays["time_center"].copy(),
        "duration": arrays["duration"].copy(),
        "velocity": (s * arrays["velocity"].astype(np.float64) @ r.T).astype(np.float32),
        "scale": (s * arrays["scale"].astype(np.float64)).astype(np.float32),
        "rotation": canonical(quat_multiply(q, arrays["rotation"].astype(np.float64))).astype(
            np.float32
        ),
        "opacity": arrays["opacity"].copy(),
        "sh": np.einsum("ij,njc->nic", m, sh).astype(np.float32),
    }


def read_arrays(path: Path):
    from openftgs.io import read_exchange

    model = read_exchange(path)
    return {k: v.numpy() for k, v in model.arrays.items()}, model


def pose_entry(model: str, pose) -> dict:
    t, q, s = pose
    w, x, y, z = (float(c) for c in q)
    return {
        "model": model,
        "translation": [float(c) for c in t],
        "rotation": [x, y, z, w],
        "scale": float(s),
    }


def write_poses(path: Path, merged: str, entries: list[dict]) -> None:
    body = {
        "format": "openftgs-xr-fixture-poses",
        "version": 1,
        "producer": PRODUCER,
        "convention": (
            "x_merged = translation + scale * R(rotation) * x_instance; rotation is a unit "
            "quaternion (x, y, z, w); paths are relative to this file"
        ),
        "merged": merged,
        "instances": entries,
    }
    path.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")


def write_merged(name: str, model, items: list) -> None:
    from ftgs_harness.formats import write_request

    path = PROBES_DIR / f"{name}.safetensors"
    request = PROBES_DIR / f"{name}.request.json"
    write_model(path, model)
    write_request(request, items, time_unit=model.time_unit)
    render_reference(path, request, PROBES_DIR / f"{name}.renders.safetensors")


def r7() -> None:
    """R-7: P7's two primitives as two posed instances, at eight times."""
    from make_probes import time_items

    arrays, p7 = read_arrays(PROBES_DIR / "P7.safetensors")
    rows = []
    entries = []
    for k, (name, pose) in enumerate(R7_POSES.items()):
        one = {key: value[k : k + 1].copy() for key, value in arrays.items()}
        write_model(MERGED_DIR / f"{name}.safetensors", exchange_model(one, 0, TAU0, TAU1))
        rows.append(posed(one, p7.sh_degree, pose))
        entries.append(pose_entry(f"{name}.safetensors", pose))
    merged = {key: np.concatenate([r[key] for r in rows]) for key in rows[0]}
    write_merged("R7_merged", exchange_model(merged, 0, TAU0, TAU1), time_items(R7_TIMES))
    write_poses(MERGED_DIR / "R7_poses.json", "../probes/R7_merged.safetensors", entries)


def r8() -> None:
    """R-8: one degree-3 instance rotated 90° about y and scaled 2, on R2_overlap_deg3's ring."""
    from make_probes import r2_overlap_deg3

    arrays, source = read_arrays(PROBES_DIR / f"{R8_SOURCE}.safetensors")
    _, items = r2_overlap_deg3()
    merged = posed(arrays, source.sh_degree, R8_POSE)
    write_merged("R8_merged", exchange_model(merged, source.sh_degree, TAU0, TAU1), items)
    entry = pose_entry(f"../probes/{R8_SOURCE}.safetensors", R8_POSE)
    write_poses(MERGED_DIR / "R8_poses.json", "../probes/R8_merged.safetensors", [entry])


def main() -> None:
    pin_threads()
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    r7()
    r8()
    save_commands(WORK / "commands.jsonl")


if __name__ == "__main__":
    main()
