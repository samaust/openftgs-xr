# SPDX-License-Identifier: Apache-2.0
"""The probe models, their render requests and openftgs's reference renders (§2.7, §3.2).

Writes, under tests/fixtures/probes/, for every probe NAME: NAME.safetensors (the model, through
openftgs's `io.write_exchange`), NAME.request.json (its `ftgs-render-request`) and
NAME.renders.safetensors (`ftgs render NAME.safetensors --request NAME.request.json --out
NAME.renders.safetensors --backend reference`). Also the PLY snapshot of one probe at a τ inside
its span (U-6), through openftgs's `write_ply`.

The probes:
- P1–P10, openftgs's L1 probes (§3.2 R-1; spec Appendix A row 8): each built by the harness's own
  `PROBES[k].build(32)`, so the sweeps and their margins are openftgs's, not restated here;
- R2_overlap_deg3, R2_white_background, R2_single_timestamp, R2_epoch_time, R2_random: R-2's
  additional probes (Q-10, Q-11);
- R19_transmittance: D-7's bound (R-19, Q-10).
"""

import argparse
import math

import numpy as np
import torch
from fixture_common import (
    FIXTURES,
    WORK,
    exchange_model,
    pin_threads,
    render_reference,
    save_commands,
    write_model,
)

# Adapted from openftgs (https://github.com/samaust/openftgs), commit 9362202,
# harness/src/ftgs_harness/probes.py, Apache-2.0: the probes' constants, the camera and item
# helpers (`_camera`, `_item`, `_time_items`), P5's 360° ring of eyes and the pixel-centre
# offset of `_opacity_items`, used below by the additional probes.
# §3.2: the probes' camera and time span (openftgs's L1 probes, spec Appendix A row 8).
WIDTH, HEIGHT, FOCAL = 64, 48, 60.0
TAU0, TAU1, TC = 0.0, 1.9, 0.95
EYE = np.array([0.0, -4.0, 0.0])  # looking along +y, world z up
ITEMS = 32  # §3.2: 32 items per probe

PROBES_DIR = FIXTURES / "probes"

# U-6: the probe and the τ of the PLY snapshot (inside the span, away from its time centers).
PLY_PROBE, PLY_TAU = "R2_overlap_deg3", 1.3


def camera(
    eye: np.ndarray = EYE,
    *,
    width: int = WIDTH,
    height: int = HEIGHT,
    focal: float = FOCAL,
    dx: float = 0.0,
    dy: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """K and world-to-camera of a camera at `eye` looking at the origin, world z up, with the
    principal point moved by (dx, dy) pixels from the image centre."""
    from ftgs_harness.cameras import intrinsics, look_at

    K = intrinsics(width, height, focal)
    K[0, 2] += dx
    K[1, 2] += dy
    return K, look_at(eye)


def item(i: int, tau: float, cam: tuple[np.ndarray, np.ndarray], width=WIDTH, height=HEIGHT):
    from ftgs_harness.formats import RequestItem

    return RequestItem(f"r{i:04d}", float(tau), cam[0], cam[1], width, height)


def time_items(times: np.ndarray) -> list:
    cam = camera()
    return [item(i, t, cam) for i, t in enumerate(times)]


def single(**kwargs) -> dict[str, np.ndarray]:
    """The arrays of the harness's one-row probe model."""
    from ftgs_harness.models import single_primitive

    return single_primitive(**kwargs).arrays


def stack(*rows: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}


# Q-10: R-2's additional probes and R-19's, each with 32 items.


def r2_overlap_deg3():
    """Two overlapping degree-3 primitives, anisotropic and rotated, one of them moving and
    short-lived, seen from P5's 360° ring while τ runs over the span."""
    from ftgs_harness.models import unit_quaternions

    rng = np.random.default_rng(2)
    sh = np.zeros((2, 16, 3), np.float32)
    sh[0, 0] = (0.6, 0.1, -0.4)
    sh[1, 0] = (-0.3, 0.5, 0.2)
    sh[:, 1:] = (rng.normal(size=(2, 15, 3)) * 0.12).astype(np.float32)
    arrays = {
        "position": np.array([[-0.05, 0.0, 0.02], [0.06, 0.1, -0.03]], np.float32),
        "time_center": np.array([TC, 0.8]),
        "duration": np.array([10.0, 0.9]),
        "velocity": np.array([[0.0, 0.0, 0.0], [0.08, 0.0, 0.05]], np.float32),
        "scale": np.array([[0.15, 0.08, 0.05], [0.06, 0.12, 0.10]], np.float32),
        "rotation": unit_quaternions(rng, 2),
        "opacity": np.array([0.7, 0.6], np.float32),
        "sh": sh,
    }
    items = []
    times = np.linspace(TAU0, TAU1, ITEMS)
    for i, deg in enumerate(np.linspace(0.0, 360.0, ITEMS, endpoint=False)):
        rad = math.radians(deg)
        eye = 4.0 * np.array([math.sin(rad), -math.cos(rad), 0.3])
        items.append(item(i, times[i], camera(eye)))
    return exchange_model(arrays, 3, TAU0, TAU1), items


def r2_white_background():
    """A primitive of a model trained against white, rendered with the request's black
    background (the request's background wins, openftgs D-84), over τ."""
    arrays = single(duration=0.5, scale=(0.1, 0.1, 0.1), opacity=0.8)
    model = exchange_model(arrays, 0, TAU0, TAU1, background=(1.0, 1.0, 1.0))
    return model, time_items(np.linspace(TAU0, TAU1, ITEMS))


def r2_single_timestamp():
    """A single-timestamp model (τ₀ = τ₁ = 0.95, durations in the unit 1, openftgs D-70, D-71)
    rendered at τ ≠ τ₀, before, after and far outside it."""
    from ftgs_harness.models import unit_quaternions

    rng = np.random.default_rng(3)
    sh = (rng.normal(size=(3, 4, 3)) * 0.3).astype(np.float32)
    arrays = {
        "position": np.array([[-0.3, 0.0, 0.0], [0.0, 0.2, 0.1], [0.25, -0.1, -0.15]], np.float32),
        "time_center": np.full(3, TC),
        "duration": np.array([1.0, 0.3, 2.0]),
        "velocity": np.zeros((3, 3), np.float32),
        "scale": np.array([[0.08, 0.08, 0.08], [0.12, 0.05, 0.07], [0.06, 0.1, 0.09]], np.float32),
        "rotation": unit_quaternions(rng, 3),
        "opacity": np.array([0.8, 0.5, 0.9], np.float32),
        "sh": sh,
    }
    times = np.linspace(-1.0, 3.0, ITEMS)
    assert not np.any(times == TC)
    return exchange_model(arrays, 1, TC, TC), time_items(times)


def r2_epoch_time():
    """P2's moving primitive, with a duration of 0.5, in a span starting at τ₀ = 10⁹ s."""
    tau0 = 1e9
    arrays = single(
        velocity=(0.6, 0.0, 0.0),
        scale=(0.07, 0.07, 0.07),
        duration=0.5,
        time_center=tau0 + TC,
        tau0=tau0,
        tau1=tau0 + TAU1,
    )
    times = tau0 + np.linspace(TAU0, TAU1, ITEMS)
    return exchange_model(arrays, 0, tau0, tau0 + TAU1), time_items(times)


# Q-11: R-2's random probe, made smaller than R-2's N = 50,000 at 640 × 480 so the committed set
# stays small (ticket #3): N, the degree and the image size below.
RANDOM_N, RANDOM_DEGREE = 10_000, 0
RANDOM_WIDTH, RANDOM_HEIGHT, RANDOM_FOCAL = 320, 240, 300.0
RANDOM_TIMES = (0.3, TC, 1.6)


def r2_random():
    """openftgs's harness random model (`random_exchange`, seed 50) at three times."""
    from ftgs_harness.models import random_exchange

    rng = np.random.default_rng(50)
    model = random_exchange(rng, RANDOM_N, sh_degree=RANDOM_DEGREE, tau0=TAU0, tau1=TAU1)
    cam = camera(width=RANDOM_WIDTH, height=RANDOM_HEIGHT, focal=RANDOM_FOCAL)
    items = [item(i, t, cam, RANDOM_WIDTH, RANDOM_HEIGHT) for i, t in enumerate(RANDOM_TIMES)]
    return exchange_model(model.arrays, RANDOM_DEGREE, TAU0, TAU1), items


def r19_transmittance():
    """A near primitive of α = 0.95 in front of a far one of α = α_max = 0.999, both centred on
    a pixel centre at the first item, so openftgs's transmittance stop (10⁻⁴) excludes the far
    one there (T = 0.05 · 0.001 = 5·10⁻⁵); the principal point then moves by sub-pixel steps."""
    near = single(
        position=(0.0, -0.5, 0.0), scale=(0.12, 0.12, 0.12), opacity=0.95,
        color_sh=(1.0, -0.5, -0.5),
    )  # fmt: skip
    far = single(scale=(0.1, 0.1, 0.1), opacity=1.0 - 1e-7, color_sh=(-0.5, -0.5, 1.0))
    shifts = np.linspace(0.0, 1.0, ITEMS)
    items = [item(i, TC, camera(dx=0.5 + s, dy=0.5)) for i, s in enumerate(shifts)]
    return exchange_model(stack(near, far), 0, TAU0, TAU1), items


ADDITIONAL = {
    "R2_overlap_deg3": r2_overlap_deg3,
    "R2_white_background": r2_white_background,
    "R2_single_timestamp": r2_single_timestamp,
    "R2_epoch_time": r2_epoch_time,
    "R2_random": r2_random,
    "R19_transmittance": r19_transmittance,
}


def l1_probes():
    """openftgs's L1 probes P1–P10, built by the harness itself with 32 items."""
    from ftgs_harness.probes import PROBES

    for probe in PROBES:
        model, entries = probe.build(ITEMS)
        yield (
            probe.name,
            (
                exchange_model(model.arrays, model.sh_degree, model.tau0, model.tau1),
                [entry.item for entry in entries],
            ),
        )


def write_probe(name: str, model, items: list) -> None:
    """The probe's model, request and reference renders."""
    from ftgs_harness.formats import write_request

    path = PROBES_DIR / f"{name}.safetensors"
    request = PROBES_DIR / f"{name}.request.json"
    renders = PROBES_DIR / f"{name}.renders.safetensors"
    write_model(path, model)
    write_request(request, items, time_unit=model.time_unit)
    render_reference(path, request, renders)


def write_probe_ply() -> None:
    """U-6: the PLY snapshot of PLY_PROBE at PLY_TAU, from openftgs's per-time arrays."""
    from openftgs.io import read_exchange, write_ply
    from openftgs.model import FreeTimeGaussians

    exchange = read_exchange(PROBES_DIR / f"{PLY_PROBE}.safetensors")
    model = FreeTimeGaussians.from_exchange(exchange, device="cpu")
    t = (PLY_TAU - exchange.tau0) / (exchange.tau1 - exchange.tau0)
    with torch.no_grad():
        g = model.at_time(t)
        write_ply(
            PROBES_DIR / f"{PLY_PROBE}.tau{PLY_TAU}.ply",
            means=g.means_t,
            scales=g.scales,
            quats=g.quats_n,
            opacities=g.opacities_t,
            sh=g.sh,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="*", help="probe names to (re)make; default: all")
    args = parser.parse_args()
    pin_threads()
    PROBES_DIR.mkdir(parents=True, exist_ok=True)
    probes = dict(l1_probes())
    probes.update({name: build() for name, build in ADDITIONAL.items()})
    for name, (model, items) in probes.items():
        if args.only and name not in args.only:
            continue
        print(f"probe {name}: N = {model.arrays['position'].shape[0]}, {len(items)} items")
        write_probe(name, model, items)
    if not args.only or PLY_PROBE in args.only:
        write_probe_ply()
    save_commands(WORK / "commands.jsonl")


if __name__ == "__main__":
    main()
