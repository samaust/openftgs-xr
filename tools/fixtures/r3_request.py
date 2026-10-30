# SPDX-License-Identifier: Apache-2.0
"""R-3's render request of one trained model (§3.2 R-3, §2.7 Fixtures; Q-T110.1).

Usage, with openftgs's Python (`train_tiny.sh` runs it):

    python tools/fixtures/r3_request.py SCENE.json MODEL.safetensors OUT.request.json

SCENE.json is the `ftgs-scene` manifest the model was trained on, MODEL.safetensors its
`ftgs export`. The request holds R-3's 20 items, all at the training cameras' size, in the
scene's time unit, black background, the model's full SH degree:

- 6 training cameras on their own frames inside the span (frames 2 and 9, 5 and 6, 8 and 3 of
  the first, second and third training camera);
- 3 training cameras between two of their own frames (the midpoints of frames 4–5, 7–8, 0–1);
- 4 at τ₀ and τ₁ exactly: the first training camera at τ₀, the third at τ₁, and the cameras
  halfway between the first and second at τ₀ and between the second and third at τ₁;
- 3 interpolated cameras inside the span: halfway between the second and third training camera
  on the second's frame 4, a quarter of the way from the first to the second between the first's
  frames 6 and 7, and halfway between the first and the third at the span's midpoint;
- 4 outside the span: the second training camera two frame intervals before τ₀ and after τ₁,
  and the halfway cameras of the first pair and the second pair half the span before τ₀ and
  after τ₁.

A training camera is a camera with at least one `train` image, in name order; its frames are its
own images' timestamps in order, so an unsynchronized scene's frames are per camera. An
interpolated camera moves its centre linearly and its camera-to-world rotation along the
shortest arc (slerp), with the first camera's K and size. τ₀ and τ₁ are the model's.
"""

# Q-T110.1: the 20 items above are this script's choice; R-3 names their kinds only.

import argparse
import json
from pathlib import Path

import numpy as np

ITEMS = 20  # §3.2 R-3: 20 items per model


def quaternion(R: np.ndarray) -> np.ndarray:
    """The unit quaternion (w, x, y, z) of a rotation matrix, w ≥ 0."""
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x = np.copysign(x, R[2, 1] - R[1, 2])
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def slerp(a: np.ndarray, b: np.ndarray, s: float) -> np.ndarray:
    if np.dot(a, b) < 0:
        b = -b
    angle = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
    if angle < 1e-9:
        return a
    return (np.sin((1 - s) * angle) * a + np.sin(s * angle) * b) / np.sin(angle)


class Camera:
    def __init__(self, entry: dict):
        self.width, self.height = int(entry["width"]), int(entry["height"])
        self.K = np.asarray(entry["K"], dtype=np.float64)
        w2c = np.asarray(entry["world_to_camera"], dtype=np.float64)
        self.R, self.t = w2c[:3, :3], w2c[:3, 3]
        self.centre = -self.R.T @ self.t

    def world_to_camera(self) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = self.R, self.t
        return m


def between(a: Camera, b: Camera, s: float) -> Camera:
    """The camera a fraction `s` of the way from `a` to `b`."""
    c = Camera.__new__(Camera)
    c.width, c.height, c.K = a.width, a.height, a.K
    c2w = matrix(slerp(quaternion(a.R.T), quaternion(b.R.T), s))
    c.centre = (1 - s) * a.centre + s * b.centre
    c.R = c2w.T
    c.t = -c.R @ c.centre
    return c


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scene", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    from ftgs_harness.formats import RequestItem, write_request
    from openftgs.io import read_exchange

    scene = json.loads(args.scene.read_text(encoding="utf-8"))
    model = read_exchange(args.model)
    if model.time_unit != scene["time_unit"]:
        raise SystemExit(f"time units differ: model {model.time_unit}, scene {scene['time_unit']}")
    tau0, tau1 = float(model.tau0), float(model.tau1)
    span = tau1 - tau0

    names = sorted({im["camera"] for im in scene["images"] if im["split"] == "train"})
    if len(names) < 3:
        raise SystemExit(f"R-3 needs three training cameras, the scene has {names}")
    cams = [Camera(scene["cameras"][n]) for n in names[:3]]
    frames = [
        sorted(im["timestamp"] for im in scene["images"] if im["camera"] == n) for n in names[:3]
    ]
    if min(len(f) for f in frames) < 10:
        raise SystemExit("R-3's items need ten frames per training camera")
    interval = float(np.median(np.diff(frames[0])))

    def mid(k: int, f: int) -> float:
        return (frames[k][f] + frames[k][f + 1]) / 2

    half01, half12 = between(cams[0], cams[1], 0.5), between(cams[1], cams[2], 0.5)
    items = [
        # training cameras on their own frames
        (cams[0], frames[0][2]),
        (cams[0], frames[0][9]),
        (cams[1], frames[1][5]),
        (cams[1], frames[1][6]),
        (cams[2], frames[2][8]),
        (cams[2], frames[2][3]),
        # training cameras between their frames
        (cams[0], mid(0, 4)),
        (cams[1], mid(1, 7)),
        (cams[2], mid(2, 0)),
        # τ₀ and τ₁
        (cams[0], tau0),
        (cams[2], tau1),
        (half01, tau0),
        (half12, tau1),
        # interpolated cameras inside the span
        (half12, frames[1][4]),
        (between(cams[0], cams[1], 0.25), mid(0, 6)),
        (between(cams[0], cams[2], 0.5), tau0 + span / 2),
        # outside the span
        (cams[1], tau0 - 2 * interval),
        (cams[1], tau1 + 2 * interval),
        (half01, tau0 - span / 2),
        (half12, tau1 + span / 2),
    ]
    assert len(items) == ITEMS
    request = [
        RequestItem(f"r{i:04d}", float(t), c.K, c.world_to_camera(), c.width, c.height)
        for i, (c, t) in enumerate(items)
    ]
    digest = write_request(args.out, request, time_unit=scene["time_unit"])
    print(f"{args.out}: {len(request)} items, sha256 {digest}")


if __name__ == "__main__":
    main()
