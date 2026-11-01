"""Local AI models for visuals: illustrations (Z-Image), depth (Depth Anything), motion (Wan).

Each wrapper loads its model on first use and frees GPU memory on close(), so only one
large model occupies the GPU at a time.
"""

from __future__ import annotations

import gc
import logging
import time

import numpy as np
from PIL import Image

from factshorts.config import VisualsConfig

log = logging.getLogger(__name__)


def free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class ImageGenerator:
    """Z-Image-Turbo. Prompts are encoded first, then the text encoder is dropped so the
    diffusion transformer fits on a 16 GB GPU without per-step offloading."""

    def __init__(self, cfg: VisualsConfig) -> None:
        self.cfg = cfg

    def generate(self, prompts: list[str], seeds: list[int]) -> list[Image.Image]:
        import torch
        from diffusers import ZImagePipeline

        t0 = time.time()
        pipe = ZImagePipeline.from_pretrained(self.cfg.image_model, torch_dtype=torch.bfloat16)
        device = self.cfg.device
        try:
            pipe.text_encoder.to(device)
            with torch.no_grad():
                embeds = [
                    pipe.encode_prompt(p, device=device, do_classifier_free_guidance=False)[0]
                    for p in prompts
                ]
            pipe.text_encoder.to("cpu")
            pipe.text_encoder = None
            free_gpu()
            pipe.to(device)
            fast = True
        except torch.OutOfMemoryError:
            log.warning("not enough VRAM for the fast path; using CPU offload (slower)")
            pipe = ZImagePipeline.from_pretrained(self.cfg.image_model, torch_dtype=torch.bfloat16)
            pipe.enable_model_cpu_offload()
            embeds, fast = [None] * len(prompts), False
        log.info("image model ready in %.0fs (fast=%s)", time.time() - t0, fast)
        images = []
        for prompt, emb, seed in zip(prompts, embeds, seeds, strict=True):
            t = time.time()
            kwargs = {"prompt_embeds": emb} if emb is not None else {"prompt": prompt}
            img = pipe(
                **kwargs,
                height=self.cfg.image_height,
                width=self.cfg.image_width,
                num_inference_steps=self.cfg.image_steps,
                guidance_scale=0.0,
                generator=torch.Generator(device).manual_seed(seed),
            ).images[0]
            images.append(img)
            log.info("illustration %d/%d in %.1fs", len(images), len(prompts), time.time() - t)
        del pipe
        free_gpu()
        return images


class DepthEstimator:
    def __init__(self, cfg: VisualsConfig) -> None:
        self.cfg = cfg
        self._pipe = None

    def estimate(self, img: Image.Image) -> np.ndarray:
        """Normalized depth map, 1 = near, same size as `img`."""
        if self._pipe is None:
            from transformers import pipeline

            self._pipe = pipeline(
                "depth-estimation",
                model=self.cfg.depth_model,
                device=0 if self.cfg.device == "cuda" else -1,
            )
        d = np.asarray(self._pipe(img)["depth"].resize(img.size), dtype=np.float32)
        return (d - d.min()) / (d.max() - d.min() + 1e-6)

    def close(self) -> None:
        self._pipe = None
        free_gpu()


class MotionGenerator:
    """Wan 2.2 TI2V-5B image-to-video. Stays at 480x832 on 16 GB GPUs."""

    def __init__(self, cfg: VisualsConfig) -> None:
        self.cfg = cfg
        self._pipe = None

    def _load(self):
        import torch
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

        vae = AutoencoderKLWan.from_pretrained(
            self.cfg.video_model, subfolder="vae", torch_dtype=torch.float32
        )
        pipe = WanImageToVideoPipeline.from_pretrained(
            self.cfg.video_model, vae=vae, torch_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload()
        return pipe

    def animate(self, image: Image.Image, prompt: str, seed: int) -> list[np.ndarray]:
        import torch

        if self._pipe is None:
            self._pipe = self._load()
        c = self.cfg
        t = time.time()
        frames = self._pipe(
            image=image.convert("RGB").resize(
                (c.wan_width, c.wan_height), Image.Resampling.LANCZOS
            ),
            prompt=f"{prompt}, smooth 2D animation, consistent style",
            negative_prompt="photorealistic, text, letters, watermark, blurry, distorted, "
            "morphing, extra limbs, flicker",
            height=c.wan_height,
            width=c.wan_width,
            num_frames=c.wan_frames,
            guidance_scale=5.0,
            num_inference_steps=c.wan_steps,
            generator=torch.Generator("cuda").manual_seed(seed),
            output_type="np",
        ).frames[0]
        log.info("motion clip (%d frames) in %.0fs", len(frames), time.time() - t)
        return [(np.clip(f, 0, 1) * 255).astype(np.uint8) for f in frames]

    def close(self) -> None:
        self._pipe = None
        free_gpu()
