"""
Whisper STT via mlx-whisper (Apple Silicon optimised).

Loads the model once at init. transcribe() is synchronous — the service
calls it from the audio thread after VAD gates an utterance.
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TranscriptResult:
    text: str
    confidence: float          # 0.0–1.0; below threshold → flag for flywheel
    language: str
    no_speech_prob: float      # raw Whisper signal, useful for flywheel filtering
    avg_logprob: float         # raw Whisper signal


class WhisperSTT:
    def __init__(self, model: str = "mlx-community/whisper-small.en-mlx", language: str = "en") -> None:
        self._model_path = model
        self._language = language
        self._mlx_whisper = None   # lazy import after init log
        logger.info("Loading Whisper model: %s", model)
        import mlx_whisper
        self._mlx_whisper = mlx_whisper
        # Warm up: transcribe silence to pre-compile MLX graph
        self._warmup()
        logger.info("Whisper ready")

    def transcribe(self, audio: np.ndarray) -> TranscriptResult:
        """
        Transcribe a 1-D float32 numpy array at 16kHz.
        Returns TranscriptResult with text and a confidence score.
        """
        result = self._mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._model_path,
            language=self._language,
            verbose=False,
        )

        text = result.get("text", "").strip()
        segments = result.get("segments", [])

        if segments:
            avg_logprob   = float(np.mean([s.get("avg_logprob", -1.0) for s in segments]))
            no_speech_prob = float(np.mean([s.get("no_speech_prob", 0.0) for s in segments]))
        else:
            avg_logprob    = -1.0
            no_speech_prob = 1.0   # no segments → treat as silence

        confidence = _compute_confidence(avg_logprob, no_speech_prob)

        return TranscriptResult(
            text=text,
            confidence=confidence,
            language=result.get("language", self._language),
            no_speech_prob=no_speech_prob,
            avg_logprob=avg_logprob,
        )

    def _warmup(self) -> None:
        silence = np.zeros(16000, dtype=np.float32)   # 1 second of silence
        self._mlx_whisper.transcribe(
            silence,
            path_or_hf_repo=self._model_path,
            language=self._language,
            verbose=False,
        )
        logger.debug("Whisper warmup complete")


def _compute_confidence(avg_logprob: float, no_speech_prob: float) -> float:
    """
    Map Whisper's internal scores to a [0, 1] confidence value.

    avg_logprob is typically in [-2, 0]:
      -0.5 → high confidence, -1.5 → low confidence
    We map [-1.5, -0.3] → [0.0, 1.0] linearly, then scale by (1 - no_speech_prob).
    """
    LOW, HIGH = -1.5, -0.3
    log_conf = (avg_logprob - LOW) / (HIGH - LOW)
    log_conf = max(0.0, min(1.0, log_conf))
    return log_conf * (1.0 - no_speech_prob)
