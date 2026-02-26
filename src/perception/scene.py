"""
SmolVLM2-500M scene understanding.

Runs periodic snapshot inference on kitchen frames. Designed around the
constraint that edge VLM answers "what is happening?" not "what should we do?"
— complex reasoning is the cerebrum's job.

Key design choices:
  - Prompt is intentionally simple; SmolVLM2-500M is brittle on complex instructions
  - Activity and objects are extracted from free-text output via keyword matching,
    not JSON parsing — more robust for a 500M model
  - Change detection uses pixel-level MAD before spending a VLM inference budget
  - Confidence is a heuristic from generation token scores, not calibrated probability
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Activity vocabulary — order matters (more specific first)
_ACTIVITY_KEYWORDS: list[tuple[str, list[str]]] = [
    ("cooking",  ["chop", "cut", "stir", "boil", "fry", "saute", "sauté", "cook",
                  "prepare", "peel", "mince", "dice", "slice", "grate", "mix",
                  "pour", "bake", "roast", "simmer", "knead", "whisk"]),
    ("eating",   ["eat", "dining", "meal", "food", "bite", "drink"]),
    ("cleaning", ["wash", "clean", "wipe", "scrub", "rinse", "tidy"]),
    ("idle",     ["empty", "no one", "nobody", "nothing", "quiet", "unoccupied"]),
]

# Object vocabulary for keyword matching
_KITCHEN_OBJECTS = [
    "knife", "cutting board", "pot", "pan", "bowl", "plate", "cup", "mug",
    "spoon", "fork", "spatula", "whisk", "ladle", "tongs", "oven", "stove",
    "microwave", "refrigerator", "fridge", "counter", "sink", "faucet",
    "vegetables", "fruit", "meat", "chicken", "fish", "onion", "garlic",
    "carrot", "tomato", "pepper", "potato", "herbs", "spices", "flour",
    "water", "oil", "butter", "eggs", "pasta", "rice",
    "person", "hand", "hands",
]

# Pixel-level change detection
_CHANGE_DETECTION_SIZE = (64, 64)    # resize for fast comparison
_CHANGE_MAD_THRESHOLD  = 25.0 / 255  # mean absolute difference threshold (0–1 range)


@dataclass
class SceneResult:
    description: str
    objects: list[str]
    activity: str                  # cooking | eating | cleaning | idle | unknown
    confidence: float              # 0.0–1.0 heuristic, NOT calibrated
    change_detected: bool = False
    change_magnitude: float = 0.0  # 0.0–1.0


class SmolVLM2Scene:
    PROMPT = (
        "You are observing a kitchen. In 1-2 sentences, describe what activity "
        "is happening and what main objects or people you can see."
    )

    def __init__(self, model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct") -> None:
        self._model_id  = model_id
        self._processor = None
        self._model     = None
        self._device    = None
        self._last_frame_gray: Optional[np.ndarray] = None   # for change detection

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if torch.backends.mps.is_available():
            self._device = "mps"
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"
            logger.warning("No GPU available — SmolVLM2 will run on CPU (slow)")

        logger.info("Loading SmolVLM2 on %s: %s", self._device, self._model_id)

        self._processor = AutoProcessor.from_pretrained(self._model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self._model_id,
            dtype=torch.bfloat16,
            _attn_implementation="sdpa",    # ~2x faster than eager on MPS
        )
        self._model = self._model.to(self._device)
        self._model.eval()
        logger.info("SmolVLM2 loaded — device=%s", self._device)

    def unload(self) -> None:
        import torch
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        if self._device == "mps":
            torch.mps.empty_cache()
        elif self._device == "cuda":
            torch.cuda.empty_cache()
        logger.info("SmolVLM2 unloaded")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def describe(self, image: Image.Image) -> SceneResult:
        """
        Run VLM inference on a single frame. Detects scene change vs. last frame.
        Returns SceneResult with description, objects, activity, and confidence.
        """
        import torch

        frame_gray = _to_gray_array(image)
        change_mag, change_detected = self._detect_change(frame_gray)
        self._last_frame_gray = frame_gray

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.PROMPT},
                ],
            }
        ]

        prompt_text = self._processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self._processor(
            text=prompt_text,
            images=[image],
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

        generated_ids = output.sequences[:, inputs["input_ids"].shape[1]:]
        description   = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        confidence = _scores_to_confidence(output.scores)
        activity   = _parse_activity(description)
        objects    = _parse_objects(description)

        logger.info(
            "Scene: activity=%s  objects=%s  conf=%.2f  change=%.2f",
            activity, objects[:3], confidence, change_mag,
        )
        logger.debug("Scene description: %s", description)

        return SceneResult(
            description=description,
            objects=objects,
            activity=activity,
            confidence=confidence,
            change_detected=change_detected,
            change_magnitude=round(change_mag, 3),
        )

    # ------------------------------------------------------------------
    # Change detection
    # ------------------------------------------------------------------

    def _detect_change(self, current_gray: np.ndarray) -> tuple[float, bool]:
        """
        Compare current frame to last stored frame.
        Returns (magnitude 0–1, exceeded_threshold).
        """
        if self._last_frame_gray is None:
            return 0.0, False
        mad = float(np.mean(np.abs(current_gray.astype(float) - self._last_frame_gray.astype(float))))
        exceeded = mad > _CHANGE_MAD_THRESHOLD
        return mad, exceeded


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _to_gray_array(image: Image.Image) -> np.ndarray:
    """Resize to small square and convert to grayscale float32 array."""
    small = image.resize(_CHANGE_DETECTION_SIZE).convert("L")
    return np.array(small, dtype=np.float32) / 255.0


def _scores_to_confidence(scores: tuple) -> float:
    """
    Derive a heuristic confidence from generation token log-probabilities.
    Mean softmax-max over output tokens — higher = model was more decisive.
    Not calibrated; use only as a relative ordering signal.
    """
    if not scores:
        return 0.5
    import torch
    probs = [torch.softmax(s, dim=-1).max().item() for s in scores]
    return round(float(np.mean(probs)), 3)


def _parse_activity(text: str) -> str:
    lower = text.lower()
    for activity, keywords in _ACTIVITY_KEYWORDS:
        if any(kw in lower for kw in keywords):
            return activity
    return "unknown"


def _parse_objects(text: str) -> list[str]:
    lower = text.lower()
    found = [obj for obj in _KITCHEN_OBJECTS if obj in lower]
    # Deduplicate preserving order (e.g. don't list "hand" and "hands")
    seen: set[str] = set()
    unique = []
    for obj in found:
        root = obj.rstrip("s")
        if root not in seen:
            seen.add(root)
            unique.append(obj)
    return unique[:6]   # cap at 6
