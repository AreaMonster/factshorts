"""Turn illustrations and motion clips into per-scene background clips of exact length."""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
from PIL import Image

from factshorts.render.easing import ease_in_out_sine

CAMERA_MOVES = ["push_in", "pan_right", "pull_out", "pan_left", "rise"]


def cover(img: Image.Image, width: int, height: int) -> Image.Image:
    """Scale to cover width x height, then center-crop (no letterboxing)."""
    scale = max(width / img.width, height / img.height)
    w, h = math.ceil(img.width * scale), math.ceil(img.height * scale)
    img = img.convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
    x, y = (w - width) // 2, (h - height) // 2
    return img.crop((x, y, x + width, y + height))


def _camera(move: str, p: float) -> tuple[float, float, float]:
    """(zoom, pan_x, pan_y) for progress p in [0, 1]; pans are in normalized units."""
    e = ease_in_out_sine(p)
    if move == "push_in":
        return 1.0 + 0.12 * e, 0.0, -0.01 * e
    if move == "pull_out":
        return 1.12 - 0.12 * e, 0.0, 0.0
    if move == "pan_right":
        return 1.08, -0.05 + 0.10 * e, 0.0
    if move == "pan_left":
        return 1.08, 0.05 - 0.10 * e, 0.0
    return 1.08 + 0.03 * e, 0.0, 0.04 - 0.08 * e  # rise


def parallax_frames(
    img: Image.Image, depth: np.ndarray, n: int, move: str, device: str = "cuda"
) -> Iterator[np.ndarray]:
    """3D-looking camera move: near pixels (depth≈1) shift more than far ones."""
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    w, h = img.size
    src = torch.from_numpy(np.array(img)).to(device).permute(2, 0, 1)[None].float() / 255
    d = torch.from_numpy(depth).to(device)
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device),
        torch.linspace(-1, 1, w, device=device),
        indexing="ij",
    )
    for i in range(n):
        p = i / max(1, n - 1)
        zoom, px, py = _camera(move, p)
        parallax = 0.03 * (ease_in_out_sine(p) - 0.5)
        gx = xs / zoom + px + parallax * (d - 0.5)
        gy = ys / zoom + py + 0.5 * parallax * (d - 0.5)
        grid = torch.stack([gx, gy], dim=-1)[None]
        out = torch.nn.functional.grid_sample(
            src, grid, mode="bilinear", padding_mode="reflection", align_corners=True
        )
        yield (out[0].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()


def kenburns_frames(img: Image.Image, n: int, move: str) -> Iterator[np.ndarray]:
    """Simple pan/zoom without depth (CPU only)."""
    w, h = img.size
    for i in range(n):
        zoom, px, py = _camera(move, i / max(1, n - 1))
        cw, ch = w / zoom, h / zoom
        cx = w / 2 + px * w / 2
        cy = h / 2 + py * h / 2
        box = (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)
        box = (max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3]))
        yield np.asarray(img.resize((w, h), Image.Resampling.BILINEAR, box=box))


def motion_frames(
    clip: list[np.ndarray],
    n: int,
    fps: int,
    clip_fps: int,
    width: int,
    height: int,
    min_speed: float,
) -> Iterator[np.ndarray]:
    """Fit an AI motion clip to `n` output frames: slow it down (not below `min_speed`),
    then hold the last frame while a slow push-in keeps the picture alive."""
    clip_len = len(clip) / clip_fps
    seg_len = n / fps
    speed = max(min_speed, min(1.0, clip_len / seg_len))
    frames = [Image.fromarray(f) for f in clip]
    cache: dict[int, Image.Image] = {}
    for i in range(n):
        t = i / fps
        src_i = min(len(frames) - 1, int(t * speed * clip_fps))
        if src_i not in cache:
            cache = {src_i: cover(frames[src_i], width, height)}
        base = cache[src_i]
        zoom = 1.0 + 0.05 * ease_in_out_sine(i / max(1, n - 1))
        if zoom > 1.001:
            cw, ch = width / zoom, height / zoom
            box = ((width - cw) / 2, (height - ch) / 2, (width + cw) / 2, (height + ch) / 2)
            yield np.asarray(base.resize((width, height), Image.Resampling.BILINEAR, box=box))
        else:
            yield np.asarray(base)


def crossfade(
    frames: Iterator[np.ndarray], previous_last: np.ndarray | None, k: int
) -> Iterator[np.ndarray]:
    """Blend the first k frames from the previous scene's last frame."""
    for i, f in enumerate(frames):
        if previous_last is not None and i < k:
            a = (i + 1) / (k + 1)
            f = f.astype(np.float32) * a + previous_last.astype(np.float32) * (1 - a)
            f = f.astype(np.uint8)
        yield f
