"""
Moondream2 VLM adapter — Phase A primary evaluation candidate.

Why Moondream2 over SmolVLM2:
  SmolVLM2-500M runs at ~32s/frame on M2 Pro MPS, making it incompatible
  with the ≥1 fps target for event detection. Moondream2 (1.86B) is purpose-built
  for edge inference and achieves ~1-3 fps on Apple Silicon. SmolVLM2 was
  disqualified for both latency and scene understanding quality (see phase_a_perception.md
  Evaluated Assumptions).

Model: vikhyatk/moondream2 (loaded via transformers + trust_remote_code)
RAM:   ~1.8 GB on MPS
FPS:   ~1-3 fps on M2 Pro (estimated; measure with evaluate_vlm.py)

Prompt design: Moondream2 responds well to direct question format. For event
detection, we use structured classification prompts that request a specific
label from a known list — easier to parse reliably than free-form description.
"""

import logging
import time

from PIL import Image

from src.perception.vlm.base import DetectionResult, VLMAdapter

logger = logging.getLogger(__name__)

# "2025-01-09" is the known-stable revision for transformers 4.x.
# transformers 5.x broke the trust_remote_code path (all_tied_weights_keys API change);
# downgrade to transformers<5.0 (pyproject.toml) to stay on this working revision.
_DEFAULT_MODEL_ID = "vikhyatk/moondream2"
_DEFAULT_REVISION  = "2025-01-09"


class MoondreamAdapter(VLMAdapter):
    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL_ID,
        revision: str = _DEFAULT_REVISION,
    ) -> None:
        self._model_id = model_id
        self._revision = revision
        self._model     = None
        self._tokenizer = None
        self._device    = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load Moondream2 onto MPS (Apple Silicon) or CPU fallback."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch.backends.mps.is_available():
            self._device = "mps"
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"
            logger.warning("No GPU available — Moondream2 will run on CPU (expect ~5-10s/frame)")

        logger.info("Loading Moondream2 on %s: %s @ %s", self._device, self._model_id, self._revision)

        kwargs = dict(revision=self._revision) if self._revision else {}

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id,
            trust_remote_code=True,
            **kwargs,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            trust_remote_code=True,
            dtype="auto",
            attn_implementation="eager",   # Moondream2 does not support SDPA
            **kwargs,
        ).to(self._device)
        self._model.eval()

        logger.info("Moondream2 loaded — device=%s", self._device)

    def unload(self) -> None:
        import torch
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        if self._device == "mps":
            torch.mps.empty_cache()
        elif self._device == "cuda":
            torch.cuda.empty_cache()
        logger.info("Moondream2 unloaded")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect(self, image: Image.Image, prompt: str) -> DetectionResult:
        """
        Run Moondream2 on image with the given classification prompt.

        The prompt should be a direct question — the model responds best to
        concise, instruction-following questions that end with a label list.
        Example: "Classify the activity. Respond with EXACTLY one label: ..."

        Returns DetectionResult with latency_ms recorded.
        confidence=1.0 if response begins with a known all-caps label,
        0.5 otherwise (response was valid but label didn't match).
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("MoondreamAdapter not loaded — call load() first")

        t0 = time.monotonic()

        enc_image = self._model.encode_image(image)
        raw_answer: str = self._model.query(enc_image, prompt)["answer"].strip()

        latency_ms = (time.monotonic() - t0) * 1000.0

        detected_label, confidence = _parse_label(raw_answer)

        logger.info(
            "Moondream2: label=%s  conf=%.1f  latency=%.0fms",
            detected_label, confidence, latency_ms,
        )
        logger.debug("Moondream2 raw response: %s", raw_answer)

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

    Moondream2 is instructed to respond with the label first. We extract the
    first word that is entirely uppercase (and at least 3 chars long) as the label.
    confidence=1.0 if found, 0.5 if only a best-effort match.
    """
    import re
    # Look for first all-caps word >= 3 chars (excludes "A", "I", "OK", etc.)
    match = re.search(r'\b([A-Z_]{3,})\b', text)
    if match:
        return match.group(1), 1.0

    # Fallback: return first word uppercased as the label, low confidence
    first_word = text.split()[0].upper() if text else "UNKNOWN"
    return first_word, 0.5
