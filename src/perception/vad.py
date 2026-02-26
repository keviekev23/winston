"""
Silero VAD wrapper.

Processes fixed-size audio chunks (512 samples @ 16kHz = 32ms) and returns
a speech probability [0.0, 1.0] per chunk. The service layer handles the
state machine (WAITING → SPEAKING → SILENCE → transcribe).

Model is loaded once at init and held in memory for the process lifetime.
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
_CHUNK_SAMPLES = 512   # Silero requires exactly 512 for 16kHz


class SileroVAD:
    def __init__(self) -> None:
        logger.info("Loading Silero VAD...")
        from silero_vad import load_silero_vad
        self._model = load_silero_vad()
        self._model.eval()
        logger.info("Silero VAD loaded")

    def speech_probability(self, chunk: np.ndarray) -> float:
        """
        Return speech probability for a single 512-sample float32 chunk.
        chunk must be 1-D, float32, normalized to [-1, 1].
        """
        if len(chunk) != _CHUNK_SAMPLES:
            raise ValueError(f"VAD expects {_CHUNK_SAMPLES} samples, got {len(chunk)}")

        tensor = torch.from_numpy(chunk).float()
        with torch.no_grad():
            prob = self._model(tensor, _SAMPLE_RATE).item()
        return float(prob)

    def reset_states(self) -> None:
        """Call between utterances to clear Silero's internal hidden state."""
        self._model.reset_states()
