"""TTS engines (Kokoro, Piper), per-sentence synthesis with cache, pauses and timing plan."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from factshorts.config import TTSConfig
from factshorts.core.captions import (
    align_by_text,
    align_word_timings,
    proportional_word_timings,
)
from factshorts.core.textnorm import to_speakable
from factshorts.models import AudioPlan, Script, SegmentAudio, SentenceAudio
from factshorts.verify.normalize import split_sentences

log = logging.getLogger(__name__)

PIPER_VOICES_REPO = "rhasspy/piper-voices"
KOKORO_REPO = "hexgrad/Kokoro-82M"
# Kokoro v1.0 English voices: a = American, b = British; f = female, m = male.
KOKORO_VOICES = [
    *(
        f"af_{n}"
        for n in (
            "heart",
            "alloy",
            "aoede",
            "bella",
            "jessica",
            "kore",
            "nicole",
            "nova",
            "river",
            "sarah",
            "sky",
        )
    ),
    *(f"am_{n}" for n in ("adam", "echo", "eric", "fenrir", "liam", "michael", "onyx", "puck")),
    *(f"bf_{n}" for n in ("alice", "emma", "isabella", "lily")),
    *(f"bm_{n}" for n in ("daniel", "fable", "george", "lewis")),
]


@dataclass
class SpokenWord:
    """A word as spoken by the engine, with times in seconds relative to the clip start."""

    text: str
    start: float
    end: float


@dataclass
class SynthResult:
    samples: np.ndarray  # mono int16
    words: list[SpokenWord] | None = None  # None if the engine has no timestamps


class Synthesizer(Protocol):
    name: str
    sample_rate: int
    settings: str

    def synthesize(self, text: str) -> SynthResult: ...


# --- Kokoro -------------------------------------------------------------------------


class KokoroSynth:
    """Kokoro-82M (Apache 2.0). Voices: af_*/am_* American, bf_*/bm_* British."""

    sample_rate = 24000

    def __init__(self, voice: str, speed: float = 1.0, device: str | None = None) -> None:
        os.environ.setdefault("HF_HUB_VERBOSITY", "error")
        import torch
        from kokoro import KPipeline

        lang = voice[0] if voice[:1] in ("a", "b") else "a"
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = KPipeline(lang_code=lang, repo_id=KOKORO_REPO, device=device)
        self.voice = voice
        self.speed = speed
        self.name = f"kokoro:{voice}"
        self.settings = f"speed={speed}"

    def synthesize(self, text: str) -> SynthResult:
        chunks: list[np.ndarray] = []
        words: list[SpokenWord] = []
        offset = 0.0
        for result in self.pipeline(text, voice=self.voice, speed=self.speed):
            if result.audio is None:
                continue
            audio = result.audio.detach().cpu().numpy()
            for tok in result.tokens or []:
                if tok.start_ts is None or tok.end_ts is None:
                    continue
                words.append(SpokenWord(tok.text, offset + tok.start_ts, offset + tok.end_ts))
            chunks.append(audio)
            offset += len(audio) / self.sample_rate
        if not chunks:
            return SynthResult(np.zeros(0, dtype=np.int16), [])
        audio = np.concatenate(chunks)
        samples = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        return SynthResult(samples, words)


# --- Piper --------------------------------------------------------------------------


def piper_voice_files(voice: str) -> tuple[str, str]:
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


def ensure_piper_voice(voice: str, models_dir: Path) -> Path:
    """Download the voice from Hugging Face on first use; return the .onnx path."""
    target_dir = models_dir / "piper"
    model_rel, config_rel = piper_voice_files(voice)
    model_path = target_dir / model_rel
    if model_path.is_file() and (target_dir / config_rel).is_file():
        return model_path
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    from huggingface_hub import hf_hub_download

    log.info("downloading Piper voice %s (one-time)", voice)
    for rel in (model_rel, config_rel):
        hf_hub_download(repo_id=PIPER_VOICES_REPO, filename=rel, local_dir=target_dir)
    return model_path


class PiperSynth:
    def __init__(self, voice: str, models_dir: Path, length_scale: float = 1.0) -> None:
        from piper import PiperVoice, SynthesisConfig

        self.name = f"piper:{voice}"
        self.voice = PiperVoice.load(ensure_piper_voice(voice, models_dir))
        self.sample_rate = self.voice.config.sample_rate
        self.syn_config = SynthesisConfig(length_scale=length_scale)
        self.settings = f"ls={length_scale}"

    def synthesize(self, text: str) -> SynthResult:
        chunks = [c.audio_int16_array for c in self.voice.synthesize(text, self.syn_config)]
        if not chunks:
            return SynthResult(np.zeros(0, dtype=np.int16))
        return SynthResult(np.concatenate(chunks).astype(np.int16))


def make_synth(cfg: TTSConfig, models_dir: Path) -> Synthesizer:
    if cfg.engine == "kokoro":
        return KokoroSynth(cfg.voice, cfg.speed)
    return PiperSynth(cfg.piper_voice, models_dir, cfg.length_scale)


# --- wav helpers and cache ------------------------------------------------------------


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


def _trim_silence(samples: np.ndarray, rate: int, threshold: int = 300) -> tuple[np.ndarray, float]:
    """Cut leading/trailing near-silence, keeping 40 ms. Returns (samples, seconds cut at start)."""
    loud = np.flatnonzero(np.abs(samples) > threshold)
    if loud.size == 0:
        return samples, 0.0
    pad = int(0.04 * rate)
    start = max(0, loud[0] - pad)
    return samples[start : min(len(samples), loud[-1] + pad)], start / rate


def synthesize_cached(
    synth: Synthesizer, spoken: str, cache_dir: Path
) -> tuple[Path, list[SpokenWord] | None]:
    key = hashlib.sha256(f"{synth.name}|{synth.settings}|{spoken}".encode()).hexdigest()
    path = cache_dir / "tts" / f"{key}.wav"
    words_path = path.with_suffix(".words.json")
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        result = synth.synthesize(spoken)
        samples, cut = _trim_silence(result.samples, synth.sample_rate)
        duration = len(samples) / synth.sample_rate
        words = None
        if result.words is not None:
            words = [
                SpokenWord(
                    w.text,
                    min(max(0.0, w.start - cut), duration),
                    min(max(0.0, w.end - cut), duration),
                )
                for w in result.words
            ]
            words_path.write_text(json.dumps([w.__dict__ for w in words]), encoding="utf-8")
        tmp = path.with_suffix(".tmp.wav")
        write_wav(tmp, samples, synth.sample_rate)
        tmp.replace(path)
    words = None
    if words_path.is_file():
        words = [SpokenWord(**w) for w in json.loads(words_path.read_text(encoding="utf-8"))]
    return path, words


def whisper_words(aligner: Any, wav: Path) -> list[SpokenWord] | None:
    """Whisper word timings for a cached sentence wav (cached next to it)."""
    path = wav.with_suffix(".whisper.json")
    if path.is_file():
        return [SpokenWord(**w) for w in json.loads(path.read_text(encoding="utf-8"))]
    try:
        words = [SpokenWord(t, a, b) for t, a, b in aligner.words(wav)]
    except Exception as exc:  # timing falls back to an estimate; the audio is fine
        log.warning("Whisper alignment failed for %s: %s", wav.name, exc)
        return None
    path.write_text(json.dumps([w.__dict__ for w in words]), encoding="utf-8")
    return words


def build_audio(
    script: Script,
    synth: Synthesizer,
    cfg: TTSConfig,
    audio_dir: Path,
    cache_dir: Path,
    aligner: Any = None,
) -> AudioPlan:
    """Synthesize each sentence, join with pauses, write narration.wav and the timing plan.

    Word timings come from the voice engine; if it has none, from `aligner` (Whisper) run on
    the sentence audio; if that fails too, they are estimated from word lengths."""
    rate = synth.sample_rate
    pieces: list[np.ndarray] = [np.zeros(int(cfg.lead_in_ms * rate / 1000), dtype=np.int16)]
    cursor = len(pieces[0])
    segments: list[SegmentAudio] = []
    sentence_gap = np.zeros(int(cfg.sentence_pause_ms * rate / 1000), dtype=np.int16)
    scene_gap = np.zeros(int(cfg.scene_pause_ms * rate / 1000), dtype=np.int16)
    n = 0
    for si, seg in enumerate(script.segments):
        seg_start = cursor
        sentences: list[SentenceAudio] = []
        seg_sentences = split_sentences(seg.narration)
        for ti, sentence in enumerate(seg_sentences):
            spoken = to_speakable(sentence, cfg.pronunciations)
            cached, spoken_words = synthesize_cached(synth, spoken, cache_dir)
            samples, _ = read_wav(cached)
            n += 1
            wav_path = audio_dir / f"{n:03d}_{seg.id}.wav"
            write_wav(wav_path, samples, rate)
            start = cursor / rate
            duration = len(samples) / rate
            words = None
            heard = None
            if not spoken_words and aligner is not None:
                heard = whisper_words(aligner, cached)
            if heard:
                words = align_by_text(
                    sentence, [(w.text, w.start, w.end) for w in heard], start, duration
                )
            elif spoken_words:
                words = align_word_timings(
                    sentence,
                    [(w.text, w.start, w.end) for w in spoken_words],
                    start,
                    duration,
                    cfg.pronunciations,
                )
            if words is None:
                words = proportional_word_timings(sentence, start, duration, cfg.pronunciations)
            sentences.append(
                SentenceAudio(
                    text=sentence,
                    spoken=spoken,
                    wav=wav_path.name,
                    start=start,
                    duration=duration,
                    words=words,
                )
            )
            pieces.append(samples)
            cursor += len(samples)
            if ti < len(seg_sentences) - 1:
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
