"""Visuals step: art direction → illustrations → depth/motion → per-scene background clips."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

from factshorts.config import Config
from factshorts.core.art import plan_shots
from factshorts.core.llm import LLM
from factshorts.models import AudioPlan, SceneVisual, Script, VisualsPlan
from factshorts.render import ffmpeg
from factshorts.render.backgrounds import (
    CAMERA_MOVES,
    cover,
    crossfade,
    kenburns_frames,
    motion_frames,
    parallax_frames,
)
from factshorts.render.themes import Theme
from factshorts.render.timeline import segment_frame_ranges

log = logging.getLogger(__name__)

Notify = Callable[[str], None]


def seed_for(job_id: str, segment_id: str) -> int:
    return int(hashlib.sha256(f"{job_id}:{segment_id}".encode()).hexdigest()[:8], 16)


def choose_kinds(script: Script, mode: str, wan_scenes: int, include_hook: bool) -> dict[str, str]:
    ids = [s.id for s in script.segments]
    if mode in ("parallax", "kenburns"):
        return dict.fromkeys(ids, mode)
    if mode == "wan":
        return dict.fromkeys(ids, "wan")
    # hybrid: hook (optionally) + the most visually interesting scenes get real motion
    chosen: list[str] = [script.hook.id] if include_hook and wan_scenes > 0 else []
    ranked = sorted(script.scenes, key=lambda s: (-s.visual_interest, ids.index(s.id)))
    for seg in ranked:
        if len(chosen) >= wan_scenes:
            break
        chosen.append(seg.id)
    return {i: ("wan" if i in chosen else "parallax") for i in ids}


def _key(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:24]


class VisualBuilder:
    """Builds background clips. Model classes are injectable so tests can use fakes."""

    def __init__(self, cfg: Config, theme: Theme, image_gen=None, depth=None, motion=None):
        from factshorts.render import aigen

        self.cfg = cfg
        self.v = cfg.visuals
        self.theme = theme
        self.image_gen = image_gen or aigen.ImageGenerator(self.v)
        self.depth = depth or aigen.DepthEstimator(self.v)
        self.motion = motion or aigen.MotionGenerator(self.v)
        self.cache = cfg.cache_dir
        (self.cache / "images").mkdir(parents=True, exist_ok=True)
        (self.cache / "clips").mkdir(parents=True, exist_ok=True)

    def build(
        self,
        job_id: str,
        script: Script,
        audio: AudioPlan,
        out_dir: Path,
        llm: LLM | None,
        unload_llm: Callable[[], None],
        notify: Notify,
    ) -> VisualsPlan:
        v, fps = self.v, self.cfg.render.fps
        if v.mode == "off":
            return VisualsPlan(mode="off")
        shots = script.model_copy(deep=True)
        if llm is not None:
            notify("planning illustrations")
            plan_shots(shots, llm, self.cfg.ollama.writer_model, self.theme.art.style_name)
        (out_dir / "shots.json").write_text(
            json.dumps(
                [
                    {
                        "segment_id": s.id,
                        "image_prompt": s.image_prompt,
                        "motion": s.motion,
                        "visual_interest": s.visual_interest,
                    }
                    for s in shots.segments
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        unload_llm()  # the LLM must leave the GPU before the image/video models load

        ranges = segment_frame_ranges(audio, fps)
        kinds = choose_kinds(shots, v.mode, v.wan_scenes, v.wan_include_hook)
        scenes = [
            SceneVisual(
                segment_id=s.id,
                kind=kinds[s.id],  # type: ignore[arg-type]
                frames=ranges[s.id][1] - ranges[s.id][0],
                prompt=f"{s.image_prompt}. {self.theme.art.style}",
                motion=s.motion,
                seed=seed_for(job_id, s.id),
            )
            for s in shots.segments
        ]

        images = self._illustrations(scenes, out_dir, notify)
        clips = self._motion_clips(scenes, images, notify)
        depths = self._depths(scenes, images, notify)

        notify("compositing scene backgrounds")
        codec = ffmpeg.pick_encoder(self.cfg.render.encoder)
        w, h = self.cfg.render.width, self.cfg.render.height
        previous_last: np.ndarray | None = None
        for i, sc in enumerate(scenes):
            img = cover(images[sc.segment_id], w, h)
            move = CAMERA_MOVES[i % len(CAMERA_MOVES)]
            if sc.kind == "wan" and sc.segment_id in clips:
                frames = motion_frames(
                    clips[sc.segment_id], sc.frames, fps, v.wan_fps, w, h, v.min_playback_speed
                )
            elif sc.kind in ("parallax", "wan") and sc.segment_id in depths:
                depth = np.asarray(Image.fromarray(depths[sc.segment_id]).resize((w, h)))
                frames = parallax_frames(
                    img, depth.astype(np.float32) / 255, sc.frames, move, v.device
                )
            else:
                frames = kenburns_frames(img, sc.frames, move)
            clip_path = out_dir / f"{i:02d}_{sc.segment_id}_bg.mp4"
            writer = ffmpeg.FrameWriter(clip_path, w, h, fps, codec, quality=14)
            last = None
            try:
                for f in crossfade(frames, previous_last, v.crossfade_frames):
                    writer.write(f.tobytes())
                    last = f
            finally:
                writer.close()
            previous_last = last
            sc.clip = clip_path.name
        return VisualsPlan(
            mode=v.mode,
            style=self.theme.art.style,
            models={"image": v.image_model, "depth": v.depth_model, "video": v.video_model},
            scenes=scenes,
        )

    # --- stages -----------------------------------------------------------------------

    def _illustrations(
        self, scenes: list[SceneVisual], out_dir: Path, notify: Notify
    ) -> dict[str, Image.Image]:
        v = self.v
        keys = {
            s.segment_id: _key(
                v.image_model, s.prompt, s.seed, v.image_width, v.image_height, v.image_steps
            )
            for s in scenes
        }
        missing = [
            s for s in scenes if not (self.cache / "images" / f"{keys[s.segment_id]}.png").is_file()
        ]
        if missing:
            notify(f"drawing {len(missing)} illustrations")
            generated = self.image_gen.generate(
                [s.prompt for s in missing], [s.seed for s in missing]
            )
            for s, img in zip(missing, generated, strict=True):
                img.save(self.cache / "images" / f"{keys[s.segment_id]}.png")
        images = {}
        for s in scenes:
            src = self.cache / "images" / f"{keys[s.segment_id]}.png"
            dst = out_dir / f"{s.segment_id}.png"
            shutil.copy2(src, dst)
            s.image = dst.name
            images[s.segment_id] = Image.open(dst).convert("RGB")
        return images

    def _motion_clips(
        self, scenes: list[SceneVisual], images: dict[str, Image.Image], notify: Notify
    ) -> dict[str, list[np.ndarray]]:
        v = self.v
        todo = [s for s in scenes if s.kind == "wan"]
        clips: dict[str, list[np.ndarray]] = {}
        for n, s in enumerate(todo, start=1):
            img_key = _key(s.prompt, s.seed)
            key = _key(
                v.video_model,
                img_key,
                s.motion,
                v.wan_steps,
                v.wan_width,
                v.wan_height,
                v.wan_frames,
            )
            path = self.cache / "clips" / f"{key}.npz"
            if path.is_file():
                clips[s.segment_id] = list(np.load(path)["frames"])
                continue
            notify(f"animating scene {n}/{len(todo)} ({s.segment_id}), ~3-4 min each")
            frames = self.motion.animate(images[s.segment_id], s.motion, s.seed)
            np.savez_compressed(path, frames=np.stack(frames))
            clips[s.segment_id] = frames
        self.motion.close()
        return clips

    def _depths(
        self, scenes: list[SceneVisual], images: dict[str, Image.Image], notify: Notify
    ) -> dict[str, np.ndarray]:
        todo = [s for s in scenes if s.kind == "parallax"]
        if not todo:
            return {}
        notify(f"estimating depth for {len(todo)} scenes")
        depths = {}
        for s in todo:
            d = self.depth.estimate(images[s.segment_id])
            depths[s.segment_id] = (d * 255).astype(np.uint8)
        self.depth.close()
        return depths
