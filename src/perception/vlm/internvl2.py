"""
InternVL2.5-1B VLM adapter via mlx-vlm — activated because Moondream2 failed Phase A.

Why InternVL2.5-1B over Moondream2:
  Moondream2 via transformers/PyTorch runs ~20s/frame on M2 Pro MPS — well below
  the ≥1fps target. InternVL2.5-1B via mlx-vlm runs ~820ms-1400ms/frame warm
  (borderline 1fps), which is the best we've found on this hardware with a VLM.
  The MLX path avoids the per-token MPS sync overhead that kills PyTorch VLMs here.

Model: mlx-community/InternVL2_5-1B-4bit (4-bit quantized InternVL2.5 1B)
RAM:   ~1.75-2.0 GB peak
FPS:   ~0.7-1.2 fps warm on M2 Pro (measure via evaluate_vlm.py; JIT first run ~6s)

Latency breakdown (512x512 image, max_tokens=10):
  First run (JIT cold): ~6000ms
  Warm run 2+: 820-1400ms
  Target: <1000ms → measure real-camera frames; varies with image complexity.
"""

import logging
import re
import time

from PIL import Image

from src.perception.vlm.base import DetectionResult, VLMAdapter

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID   = "mlx-community/InternVL2_5-1B-4bit"
# Keep max_tokens small: InternVL2 generates at ~3-5 tok/s, so 20 tokens = 4-7s
# latency. Labels are 5-8 tokens (e.g. "NONE_OF_ABOVE", "CUTTING_VEGETABLES").
# 10 tokens captures the full label with a tiny buffer.
_DEFAULT_MAX_TOKENS = 10


class InternVL2Adapter(VLMAdapter):
    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL_ID,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> None:
        self._model_id    = model_id
        self._max_tokens  = max_tokens
        self._model        = None
        self._processor    = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Download (first time) and load InternVL2.5-1B into MLX memory."""
        from mlx_vlm import load as mlx_load

        logger.info("Loading InternVL2.5-1B via mlx-vlm: %s", self._model_id)
        t0 = time.monotonic()
        self._model, self._processor = mlx_load(self._model_id)
        load_ms = (time.monotonic() - t0) * 1000
        logger.info("InternVL2.5-1B loaded in %.0fms", load_ms)

    def unload(self) -> None:
        """Release MLX model weights and clear Metal cache."""
        import mlx.core as mx

        del self._model
        del self._processor
        self._model     = None
        self._processor = None
        mx.clear_cache()
        logger.info("InternVL2.5-1B unloaded")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect(self, image: Image.Image, prompt: str) -> DetectionResult:
        """
        Classify the image using the given prompt via mlx-vlm.

        Expects a structured classification prompt from detect_event.py's
        build_prompt() — "Classify the current activity... Respond with EXACTLY
        one of these labels: ..."

        Returns DetectionResult with latency_ms for Phase A evaluation.
        """
        if self._model is None or self._processor is None:
            raise RuntimeError("InternVL2Adapter not loaded — call load() first")

        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        t0 = time.monotonic()

        formatted = apply_chat_template(
            self._processor, self._model.config, prompt, num_images=1
        )
        result = generate(
            self._model,
            self._processor,
            formatted,
            [image],
            max_tokens=self._max_tokens,
            verbose=False,
        )
        raw_answer: str = result.text.strip()

        latency_ms = (time.monotonic() - t0) * 1000.0

        detected_label, confidence = _parse_label(raw_answer)

        logger.info(
            "InternVL2.5-1B: label=%s  conf=%.1f  latency=%.0fms",
            detected_label, confidence, latency_ms,
        )
        logger.debug("InternVL2.5-1B raw response: %s", raw_answer)

        return DetectionResult(
            prompt=prompt,
            detected_label=detected_label,
            description=raw_answer,
            confidence=confidence,
            latency_ms=round(latency_ms, 1),
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_label(text: str) -> tuple[str, float]:
    """
    Extract the first all-caps token from VLM response.

    We instruct the model to lead with the label in ALL_CAPS. Grab
    the first all-caps word ≥ 3 chars as the label; confidence=1.0
    if found, 0.5 if only a best-effort match.
    """
    match = re.search(r'\b([A-Z_]{3,})\b', text)
    if match:
        return match.group(1), 1.0
    first_word = text.split()[0].upper() if text else "UNKNOWN"
    return first_word, 0.5
