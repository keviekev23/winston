"""
Camera frame capture for scene understanding.

Thin wrapper around OpenCV's VideoCapture. Provides a single-frame grab
and handles device open/release lifecycle.

MacBook built-in camera = device index 0.
"""

import logging
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class Camera:
    def __init__(self, device_index: int = 0, width: int = 1280, height: int = 720) -> None:
        self._device_index = device_index
        self._width  = width
        self._height = height
        self._cap    = None

    def open(self) -> None:
        import cv2
        self._cap = cv2.VideoCapture(self._device_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera (device_index={self._device_index}). "
                "Check that no other process is using the camera."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        # Let the camera warm up — first few frames are often underexposed
        for _ in range(3):
            self._cap.read()
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("Camera opened: device=%d  resolution=%dx%d", self._device_index, actual_w, actual_h)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera closed")

    def capture(self) -> Optional[Image.Image]:
        """
        Capture one frame. Returns a PIL Image (RGB) or None on failure.
        """
        if self._cap is None:
            raise RuntimeError("Camera not open. Call open() first.")

        import cv2
        ok, frame = self._cap.read()
        if not ok or frame is None:
            logger.warning("Camera: failed to read frame")
            return None

        # OpenCV gives BGR — convert to RGB for PIL / HF processors
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def capture_as_numpy(self) -> Optional[np.ndarray]:
        """Capture one frame as a uint8 RGB numpy array (H, W, 3)."""
        img = self.capture()
        return np.array(img) if img is not None else None

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.close()
