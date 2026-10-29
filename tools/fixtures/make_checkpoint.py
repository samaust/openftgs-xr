# SPDX-License-Identifier: Apache-2.0
"""The tiny checkpoint and its scene (§2.7 Fixtures, D-29; U-5, U-6, U-14, U-18, U-36).

1. A synthetic two-camera, two-frame scene: a 64-primitive ground-truth model (openftgs's harness
   `random_exchange`, seed 64) rendered by `ftgs render --backend reference` from two cameras at
   frames 0 and 1 (30 fps) into 8-bit PNG images; its `ftgs-scene` manifest written with
   openftgs's `write_scene_manifest`. The second camera moves between the frames (a per-image
   `world_to_camera` override) and its frame-1 image is the test split. World z up, metres.
2. 64 initial points (`write_init_points`: `ftgs init` needs CUDA), the ground truth's positions
   jittered, trained by `ftgs train` for 20 iterations on the reference backend (no CUDA
   device); growth and relocation start at iteration 500, so the checkpoint keeps 64 primitives.
3. `ftgs export` of the checkpoint, and its PLY snapshot at τ₀ (`write_ply` of `at_time(0)`).
4. A single-timestamp checkpoint (U-5's unit 1): the same training on the frame-0 images alone
   (τ₀ = τ₁), with its `ftgs export`.

Writes under tests/fixtures/checkpoint/: scene/scene.json, scene/single.json, scene/images/,
tiny/ (the whole checkpoint directory), tiny.safetensors, tiny.tau0.ply, single/,
single.safetensors. Intermediates (the ground truth, its renders, the init points, the run
directories) stay under $FIXTURES_WORK.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import torch
from fixture_common import (
    FIXTURES,
    PRODUCER,
    WORK,
    exchange_model,
    pin_threads,
    rel,
    render_reference,
    run,
    save_commands,
    write_model,
)

OUT = FIXTURES / "checkpoint"
SCENE = OUT / "scene"
FPS = 30.0
WIDTH, HEIGHT, FOCAL = 80, 60, 80.0
N = 64  # §2.7: 64 primitives
ITERATIONS = 20  # §2.7
# The run's TrainConfig (openftgs `config_from_dict`): 20 iterations; SH bands added every 5
# iterations so the higher coefficients are trained, not left at zero; deterministic kernels.
CONFIG = {
    "sh_degree": 3,
    "optim": {"iterations": ITERATIONS, "sh_interval": 5},
    "perf": {"deterministic": True},
}
# Camera eyes looking at the origin, world z up; camera 1 moves by 5 cm between the frames.
EYES = {
    ("cam0", 0): (0.0, -3.0, 0.6),
    ("cam1", 0): (2.2, -2.0, 0.9),
    ("cam0", 1): (0.0, -3.0, 0.6),
    ("cam1", 1): (2.25, -2.0, 0.9),
}
TEST = {("cam1", 1)}  # the test split


def times() -> dict[int, float]:
    return {0: 0.0, 1: 1.0 / FPS}


def cameras() -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]:
    from ftgs_harness.cameras import intrinsics, look_at

    K = intrinsics(WIDTH, HEIGHT, FOCAL)
    return {key: (K, look_at(np.array(eye))) for key, eye in EYES.items()}


def ground_truth():
    from ftgs_harness.models import random_exchange

    rng = np.random.default_rng(64)
    model = random_exchange(
        rng, N, sh_degree=1, tau0=0.0, tau1=1.0 / FPS, spread=0.4,
        log_scale=(-3.2, -2.2), opacity=(0.5, 0.95),
    )  # fmt: skip
    return model.arrays


def render_images(gt: dict[str, np.ndarray]) -> dict[tuple[str, int], Path]:
    """The scene's images: the ground truth rendered by openftgs, as 8-bit sRGB-coded PNGs."""
    from ftgs_harness.formats import RequestItem, write_request
    from openftgs.io import read_renders
    from PIL import Image

    gt_path = WORK / "gt.safetensors"
    write_model(gt_path, exchange_model(gt, 1, 0.0, 1.0 / FPS))
    items = [
        RequestItem(f"{cam}_{frame}", times()[frame], K, w2c, WIDTH, HEIGHT)
        for (cam, frame), (K, w2c) in cameras().items()
    ]
    request = WORK / "gt.request.json"
    write_request(request, items)
    renders_path = WORK / "gt.renders.safetensors"
    render_reference(gt_path, request, renders_path)
    renders = read_renders(renders_path)
    paths = {}
    for cam, frame in cameras():
        rgb = renders.rgb[f"{cam}_{frame}"].numpy()
        pixels = np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        path = SCENE / "images" / cam / f"{frame:06d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels, "RGB").save(path)
        paths[(cam, frame)] = path
    return paths


def write_manifest(path: Path, images: dict, frames: tuple[int, ...]) -> None:
    from openftgs.io import write_scene_manifest
    from openftgs.types import Camera, ImageRecord

    records = []
    for (cam, frame), (K, w2c) in cameras().items():
        if frame not in frames:
            continue
        camera = Camera(
            torch.tensor(K, dtype=torch.float32),
            torch.tensor(w2c, dtype=torch.float32),
            WIDTH,
            HEIGHT,
            cam,
        )
        split = "test" if (cam, frame) in TEST else "train"
        records.append(ImageRecord(camera, times()[frame], images[(cam, frame)], split, frame))
    write_scene_manifest(path, records, time_unit="s", metadata={"producer": PRODUCER})


def write_init(path: Path, gt: dict[str, np.ndarray], frames: tuple[int, ...]) -> None:
    """64 initial points: the ground truth's positions jittered by 2 cm, its DC colours, each
    at one of the scene's frame times."""
    from openftgs.io import write_init_points
    from openftgs.sh import C0
    from openftgs.types import InitPoints

    rng = np.random.default_rng(65)
    positions = gt["position"].astype(np.float64) + rng.normal(scale=0.02, size=(N, 3))
    colors = np.clip(C0 * gt["sh"][:, 0, :].astype(np.float64) + 0.5, 0.0, 1.0)
    stamps = np.array([times()[f] for f in rng.choice(frames, size=N)])
    points = InitPoints(
        positions=torch.from_numpy(positions.astype(np.float32)),
        colors=torch.from_numpy(colors.astype(np.float32)),
        timestamps=torch.from_numpy(stamps),
    )
    write_init_points(points, path, time_unit="s", producer=PRODUCER)


def train(name: str, manifest: Path, init: Path) -> Path:
    """`ftgs train` into $FIXTURES_WORK/run_<name>; the checkpoint copied to checkpoint/<name>."""
    config = WORK / "config.json"
    config.write_text(json.dumps(CONFIG) + "\n", encoding="utf-8")
    run_dir = WORK / f"run_{name}"
    shutil.rmtree(run_dir, ignore_errors=True)
    run(
        "ftgs", "train", rel(manifest), "--init", str(init), "--out", str(run_dir),
        "--config", str(config), "--seed", "0",
    )  # fmt: skip
    # ftgs train caches the decoded frames beside the manifest; not a fixture.
    shutil.rmtree(manifest.parent / ".ftgs-cache", ignore_errors=True)
    target = OUT / name
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(run_dir / "checkpoint", target)
    run(
        "ftgs",
        "export",
        "--format",
        "ftgs-if",
        "--out",
        rel(OUT / f"{name}.safetensors"),
        rel(target),
    )
    return target


def write_snapshot(checkpoint: Path, path: Path) -> None:
    """The checkpoint's PLY snapshot at τ₀ (normalized time 0)."""
    from openftgs.io import write_ply
    from openftgs.model import FreeTimeGaussians

    model = FreeTimeGaussians.load(checkpoint, device="cpu")
    with torch.no_grad():
        g = model.at_time(0.0)
        write_ply(
            path, means=g.means_t, scales=g.scales, quats=g.quats_n, opacities=g.opacities_t,
            sh=g.sh,
        )  # fmt: skip


def main() -> None:
    pin_threads()
    WORK.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(SCENE, ignore_errors=True)
    SCENE.mkdir(parents=True)
    gt = ground_truth()
    images = render_images(gt)
    write_manifest(SCENE / "scene.json", images, (0, 1))
    write_manifest(SCENE / "single.json", images, (0,))

    write_init(WORK / "init.safetensors", gt, (0, 1))
    tiny = train("tiny", SCENE / "scene.json", WORK / "init.safetensors")
    write_snapshot(tiny, OUT / "tiny.tau0.ply")

    write_init(WORK / "init_single.safetensors", gt, (0,))
    train("single", SCENE / "single.json", WORK / "init_single.safetensors")
    save_commands(WORK / "commands.jsonl")


if __name__ == "__main__":
    main()
