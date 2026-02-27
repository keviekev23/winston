"""
Scene understanding service — standalone runner for passive VLM snapshots.

NOTE (2026-02): The primary Phase A VLM evaluation tool is now detect_event.py
(cerebrum-directed, targeted event detection). This service provides passive
background snapshots for the legacy flywheel workflow and future Phase C
ambient monitoring. See docs/phases/phase_a_perception.md for current status.

Pipeline:
  Camera frame (every N seconds)
    → VLMAdapter.detect(frame, passive_prompt)
    → pixel MAD change detection
    → MQTT publish (perception/scene/snapshot)
    → [if change detected] MQTT publish (perception/scene/change)
    → data collection (save JPEG + JSON sidecar)

Run with:
  python -m src.perception.scene_service

Adapter is selected via config scene.adapter (default: moondream2).
"""

import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from src.debug.memory_monitor import MemoryMonitor
from src.perception.camera import Camera
from src.transport.client import MQTTClient
from src.transport.topics import Perception, System

logger = logging.getLogger(__name__)

# Pixel-level change detection (moved from SmolVLM2Scene to service level)
_CHANGE_DETECTION_SIZE = (64, 64)
_CHANGE_MAD_THRESHOLD  = 25.0 / 255   # mean absolute difference threshold (0–1 range)

# Default prompt for passive (non-targeted) operation.
# Cerebrum-directed targeted detection uses detect_event.py + scenario YAMLs instead.
_PASSIVE_PROMPT = (
    "Classify the current kitchen activity. "
    "Respond with EXACTLY one label: COOKING, EATING, CLEANING, IDLE, or NONE_OF_ABOVE. "
    "Respond with the label only, followed by a dash and one sentence description."
)


def load_config(path: str = "config/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_adapter(scene_cfg: dict):
    """Instantiate VLM adapter based on config scene.adapter."""
    adapter_name = scene_cfg.get("adapter", "moondream2")
    model_id     = scene_cfg.get("model", "vikhyatk/moondream2")
    revision     = scene_cfg.get("model_revision", "2025-01-09")

    if adapter_name == "moondream2":
        from src.perception.vlm.moondream import MoondreamAdapter
        return MoondreamAdapter(model_id=model_id, revision=revision)
    elif adapter_name == "internvl2_1b":
        from src.perception.vlm.internvl2 import InternVL2Adapter
        return InternVL2Adapter()
    else:
        raise ValueError(
            f"Unknown scene.adapter '{adapter_name}'. "
            "Valid options: moondream2, internvl2_1b"
        )


def _to_gray_array(image) -> np.ndarray:
    """Resize to small square + grayscale for fast change detection."""
    small = image.resize(_CHANGE_DETECTION_SIZE).convert("L")
    return np.array(small, dtype=np.float32) / 255.0


def _detect_change(
    current_gray: np.ndarray,
    last_gray: np.ndarray | None,
) -> tuple[float, bool]:
    if last_gray is None:
        return 0.0, False
    mad = float(np.mean(np.abs(current_gray.astype(float) - last_gray.astype(float))))
    return mad, mad > _CHANGE_MAD_THRESHOLD


class SceneService:
    def __init__(self, config: dict) -> None:
        self._cfg       = config
        self._scene_cfg = config["scene"]
        self._mqtt_cfg  = config["mqtt"]
        self._dc_cfg    = config["data_collection"]

        self._running = False
        self._snapshot_count = 0
        self._last_frame_gray: np.ndarray | None = None

        self._mqtt = MQTTClient(
            host=self._mqtt_cfg["host"],
            port=self._mqtt_cfg["port"],
            client_id="souschef-scene",
            keepalive=self._mqtt_cfg["keepalive"],
        )
        self._camera  = Camera()
        self._vlm     = _load_adapter(self._scene_cfg)
        self._monitor = MemoryMonitor(
            mqtt=self._mqtt,
            interval_seconds=config["memory_monitor"]["interval_seconds"],
            alert_headroom_mb=config["memory_monitor"]["alert_headroom_mb"],
        )
        self._monitor.register("scene")

        Path(self._dc_cfg["images_dir"]).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._running = True
        self._mqtt.connect()
        self._monitor.start()
        self._mqtt.publish(System.HEALTH, {"subsystem": "scene", "status": "starting"})

        self._vlm.load()
        self._camera.open()

        self._mqtt.publish(System.HEALTH, {"subsystem": "scene", "status": "ok"})
        logger.info(
            "Scene service running — snapshot every %ds  adapter=%s",
            self._scene_cfg["snapshot_interval_seconds"],
            self._scene_cfg.get("adapter", "moondream2"),
        )

        self._loop()

    def stop(self) -> None:
        self._running = False
        try:
            self._camera.close()
        except Exception:
            pass
        try:
            self._vlm.unload()
        except Exception:
            pass
        self._monitor.stop()
        self._mqtt.disconnect()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        interval = self._scene_cfg["snapshot_interval_seconds"]

        while self._running:
            t_start = time.monotonic()

            frame = self._camera.capture()
            if frame is None:
                logger.warning("Failed to capture frame — skipping this interval")
                time.sleep(interval)
                continue

            # Pixel MAD change detection (fast, no VLM budget)
            current_gray = _to_gray_array(frame)
            change_mag, change_detected = _detect_change(current_gray, self._last_frame_gray)
            self._last_frame_gray = current_gray

            result = self._vlm.detect(frame, _PASSIVE_PROMPT)
            self._snapshot_count += 1

            low_confidence = result.confidence < self._scene_cfg["confidence_threshold"]

            snapshot_payload = {
                "description":     result.description,
                "activity":        result.detected_label,   # COOKING, IDLE, etc.
                "confidence":      result.confidence,
                "low_confidence":  low_confidence,
                "latency_ms":      result.latency_ms,
                "snapshot_number": self._snapshot_count,
            }
            self._mqtt.publish(Perception.SCENE_SNAPSHOT, snapshot_payload)

            if change_detected:
                change_payload = {
                    "change_type":      "scene_change",
                    "description":      result.description,
                    "change_magnitude": round(change_mag, 3),
                    "confidence":       result.confidence,
                }
                self._mqtt.publish(Perception.SCENE_CHANGE, change_payload)
                logger.info("Scene change detected (magnitude=%.3f)", change_mag)

            jpeg_path = None
            if self._dc_cfg["enabled"]:
                jpeg_path = self._save_snapshot(frame, snapshot_payload)

            # Publish context for cerebrum: text summary + JPEG path
            context_payload = {
                "activity":        result.detected_label,
                "description":     result.description,
                "confidence":      result.confidence,
                "change_detected": change_detected,
                "image_path":      str(jpeg_path) if jpeg_path else None,
                "snapshot_number": self._snapshot_count,
            }
            self._mqtt.publish(Perception.SCENE_CONTEXT, context_payload)

            elapsed   = time.monotonic() - t_start
            remaining = max(0.0, interval - elapsed)
            logger.debug(
                "Snapshot #%d: %.1fs inference, sleeping %.1fs",
                self._snapshot_count, elapsed, remaining,
            )
            if remaining > 0:
                time.sleep(remaining)

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def _save_snapshot(self, frame, payload: dict) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        base = Path(self._dc_cfg["images_dir"]) / ts

        jpeg_path = base.with_suffix(".jpg")
        frame.save(str(jpeg_path), "JPEG", quality=85)

        meta_path = base.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(payload, f, indent=2)

        logger.debug("Saved snapshot: %s", jpeg_path)
        return jpeg_path


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    config = load_config()

    if not config["scene"]["enabled"]:
        logger.warning(
            "scene.enabled is false in config/default.yaml. "
            "Set it to true to run the scene service."
        )
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--force", action="store_true", help="Run even if scene.enabled=false")
        args = p.parse_args()
        if not args.force:
            sys.exit(1)

    service = SceneService(config)

    def _shutdown(sig, frame):
        logger.info("Shutting down scene service...")
        service.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    service.start()


if __name__ == "__main__":
    main()
