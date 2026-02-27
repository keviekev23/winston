"""
Abstract base class for VLM adapters.

All adapters expose the same interface: load(), unload(), detect().
The caller (detect_event.py, scene_service.py) is decoupled from the model.

Adding a new model = implement VLMAdapter, add to ADAPTER_REGISTRY.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from PIL import Image


@dataclass
class DetectionResult:
    """
    Output of a single VLM inference call.

    detected_label is the raw parsed label from the response — callers are
    responsible for matching against their scenario's known event IDs.
    latency_ms is the primary Phase A metric; track it across runs.
    """
    prompt: str
    detected_label: str      # parsed uppercase label, e.g. "CUTTING_VEGETABLES"
    description: str         # full raw VLM response text
    confidence: float        # 1.0 if label matched known labels, 0.5 if unknown, 0.0 if error
    latency_ms: float
    image_path: str | None = None


class VLMAdapter(ABC):
    """
    Abstract VLM adapter. Inject prompt at detect() call time — no static prompts.
    """

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory. Call once before any detect() calls."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release model weights and free GPU/MPS memory."""
        ...

    @abstractmethod
    def detect(self, image: Image.Image, prompt: str) -> DetectionResult:
        """
        Run a single VLM inference on image with the given prompt.
        Returns DetectionResult including latency_ms for performance tracking.
        """
        ...
