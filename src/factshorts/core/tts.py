"""Piper TTS: voice download, per-sentence synthesis with cache, pauses and timing plan."""

from __future__ import annotations

import hashlib
import logging
import os
import wave
from pathlib import Path
from typing import Protocol

import numpy as np

from factshorts.config import TTSConfig
from factshorts.core.captions import proportional_word_timings
from factshorts.core.textnorm import to_speakable
from factshorts.models import AudioPlan, Script, SegmentAudio, SentenceAudio
from factshorts.verify.normalize import split_sentences

log = logging.getLogger(__name__)

VOICES_REPO = "rhasspy/piper-voices"


class Synthesizer(Protocol):
    name: str
    sample_rate: int

    def synthesize(self, text: str) -> np.ndarray:
        """Return mono int16 samples for `text`."""
        ...


def voice_files(voice: str) -> tuple[str, str]:
    """'en_US-ryan-high' → repo paths of the .onnx model and its .json config."""
    try:
        lang_code, name, quality = voice.split("-", 2)
    except ValueError as exc:
        raise ValueError(
            f"invalid Piper voice name '{voice}' (expected e.g. en_US-ryan-high)"
        ) from exc
    family = lang_code.split("_")[0]
    base = f"{family}/{lang_code}/{name}/{quality}/{voice}"
    return f"{base}.onnx", f"{base}.onnx.json"


def ensure_voice(voice: str, models_dir: Path) -> Path:
    """Download the voice from Hugging Face on first use; return the .onnx path."""
    target_dir = models_dir / "piper"
    model_rel, config_rel = voice_files(voice)
    model_path = target_dir / model_rel
    if model_path.is_file() and (target_dir / config_rel).is_file():
        return model_path
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    from huggingface_hub import hf_hub_download

    log.info("downloading Piper voice %s (one-time)", voice)
    for rel in (model_rel, config_rel):
        hf_hub_download(repo_id=VOICES_REPO, filename=rel, local_dir=target_dir)
    return model_path


class PiperSynth:
    def __init__(self, voice: str, models_dir: Path, length_scale: float = 1.0) -> None:
        from piper import PiperVoice, SynthesisConfig

        self.name = voice
        self.voice = PiperVoice.load(ensure_voice(voice, models_dir))
        self.sample_rate = self.voice.config.sample_rate
        self.syn_config = SynthesisConfig(length_scale=length_scale)
        self.length_scale = length_scale

    def synthesize(self, text: str) -> np.ndarray:
        chunks = [c.audio_int16_array for c in self.voice.synthesize(text, self.syn_config)]
        if not chunks:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(chunks).astype(np.int16)


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    return data, rate


def write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.astype(np.int16).tobytes())


def _trim_silence(samples: np.ndarray, rate: int, threshold: int = 300) -> np.ndarray:
    """Cut leading/trailing near-silence, keeping 40 ms so words aren't clipped."""
    loud = np.flatnonzero(np.abs(samples) > threshold)
    if loud.size == 0:
        return samples
    pad = int(0.04 * rate)
    return samples[max(0, loud[0] - pad) : min(len(samples), loud[-1] + pad)]


def synthesize_cached(synth: Synthesizer, spoken: str, cache_dir: Path, settings: str) -> Path:
    key = hashlib.sha256(f"{synth.name}|{settings}|{spoken}".encode()).hexdigest()
    path = cache_dir / "tts" / f"{key}.wav"
    if not path.is_file():
        samples = _trim_silence(synth.synthesize(spoken), synth.sample_rate)
        tmp = path.with_suffix(".tmp.wav")
        write_wav(tmp, samples, synth.sample_rate)
        tmp.replace(path)
    return path


def build_audio(
    script: Script, synth: Synthesizer, cfg: TTSConfig, audio_dir: Path, cache_dir: Path
) -> AudioPlan:
    """Synthesize each sentence, join with pauses, write narration.wav and the timing plan."""
    rate = synth.sample_rate
    settings = f"ls={cfg.length_scale}"
    pieces: list[np.ndarray] = [np.zeros(int(cfg.lead_in_ms * rate / 1000), dtype=np.int16)]
    cursor = len(pieces[0])
    segments: list[SegmentAudio] = []
    sentence_gap = np.zeros(int(cfg.sentence_pause_ms * rate / 1000), dtype=np.int16)
    scene_gap = np.zeros(int(cfg.scene_pause_ms * rate / 1000), dtype=np.int16)
    n = 0
    for si, seg in enumerate(script.segments):
        seg_start = cursor
        sentences: list[SentenceAudio] = []
        for ti, sentence in enumerate(split_sentences(seg.narration)):
            spoken = to_speakable(sentence, cfg.pronunciations)
            cached = synthesize_cached(synth, spoken, cache_dir, settings)
            samples, _ = read_wav(cached)
            n += 1
            wav_path = audio_dir / f"{n:03d}_{seg.id}.wav"
            write_wav(wav_path, samples, rate)
            start = cursor / rate
            duration = len(samples) / rate
            sentences.append(
                SentenceAudio(
                    text=sentence,
                    spoken=spoken,
                    wav=wav_path.name,
                    start=start,
                    duration=duration,
                    words=proportional_word_timings(sentence, start, duration, cfg.pronunciations),
                )
            )
            pieces.append(samples)
            cursor += len(samples)
            if ti < len(split_sentences(seg.narration)) - 1:
                pieces.append(sentence_gap)
                cursor += len(sentence_gap)
        seg_end = cursor
        if si < len(script.segments) - 1:
            pieces.append(scene_gap)
            cursor += len(scene_gap)
            # The pause belongs to the scene that just finished (its visuals stay up).
            seg_end = cursor
        segments.append(
            SegmentAudio(
                segment_id=seg.id, start=seg_start / rate, end=seg_end / rate, sentences=sentences
            )
        )
    tail = np.zeros(int(cfg.tail_ms * rate / 1000), dtype=np.int16)
    pieces.append(tail)
    cursor += len(tail)
    # Lead-in belongs to the first segment, the tail to the last one.
    segments[0].start = 0.0
    segments[-1].end = cursor / rate
    narration = np.concatenate(pieces)
    out = audio_dir / "narration.wav"
    write_wav(out, narration, rate)
    return AudioPlan(
        voice=synth.name,
        sample_rate=rate,
        wav=out.name,
        duration=len(narration) / rate,
        segments=segments,
    )
