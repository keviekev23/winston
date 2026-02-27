"""
VLM adapter package — abstract interface for on-device visual language models.

Adapters are interchangeable implementations of VLMAdapter. The config key
`scene.adapter` selects which one to load at runtime:

  moondream2   → MoondreamAdapter (Phase A primary candidate)
  internvl2_1b → InternVL2Adapter (stub — activate if Moondream2 fails)

Usage:
  from src.perception.vlm import VLMAdapter, DetectionResult
  from src.perception.vlm.moondream import MoondreamAdapter
"""

from src.perception.vlm.base import DetectionResult, VLMAdapter

__all__ = ["VLMAdapter", "DetectionResult"]
