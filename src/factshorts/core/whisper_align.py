"""Word timestamps from faster-whisper (local, CPU) for voices that report none (Piper)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


class WhisperAligner:
    def __init__(self, model: str = "small.en", device: str = "cpu") -> None:
        self.model_name = model
        self.device = device
        self._model = None

    def words(self, wav: Path) -> list[tuple[str, float, float]]:
        """(word, start, end) in seconds for everything spoken in `wav`."""
        if self._model is None:
            os.environ.setdefault("HF_HUB_VERBOSITY", "error")
            from faster_whisper import WhisperModel

            self._model = WhisperModel(self.model_name, device=self.device, compute_type="int8")
        segments, _ = self._model.transcribe(
            str(wav), word_timestamps=True, beam_size=1, vad_filter=False
        )
        return [(w.word.strip(), w.start, w.end) for seg in segments for w in seg.words or []]
