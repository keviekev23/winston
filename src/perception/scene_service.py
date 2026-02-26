"""
Scene understanding service — standalone runner for Phase A VLM testing.

Pipeline:
  Camera frame (every N seconds)
    → SmolVLM2 describe()
    → MQTT publish (perception/scene/snapshot)
    → [if change detected] MQTT publish (perception/scene/change)
    → data collection (save JPEG + JSON sidecar)

Run with:
  python -m src.perception.scene_service

Can run independently alongside the audio perception service,
or as the only service when testing VLM in isolation.
"""

import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.debug.memory_monitor import MemoryMonitor
from src.perception.camera import Camera
from src.perception.scene import SmolVLM2Scene
from src.transport.client import MQTTClient
from src.transport.topics import Perception, System

logger = logging.getLogger(__name__)


def load_config(path: str = "config/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class SceneService:
    def __init__(self, config: dict) -> None:
        self._cfg       = config
        self._scene_cfg = config["scene"]
        self._mqtt_cfg  = config["mqtt"]
        self._dc_cfg    = config["data_collection"]

        self._running = False
        self._snapshot_count = 0

        self._mqtt = MQTTClient(
            host=self._mqtt_cfg["host"],
            port=self._mqtt_cfg["port"],
            client_id="souschef-scene",
            keepalive=self._mqtt_cfg["keepalive"],
        )
        self._camera = Camera()
        self._vlm    = SmolVLM2Scene(model_id=self._scene_cfg["model"])
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
            "Scene service running — snapshot every %ds",
            self._scene_cfg["snapshot_interval_seconds"],
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

            result = self._vlm.describe(frame)
            self._snapshot_count += 1

            low_confidence = result.confidence < self._scene_cfg["confidence_threshold"]

            snapshot_payload = {
                "description":     result.description,
                "objects":         result.objects,
                "activity":        result.activity,
                "confidence":      result.confidence,
                "low_confidence":  low_confidence,
                "snapshot_number": self._snapshot_count,
            }
            self._mqtt.publish(Perception.SCENE_SNAPSHOT, snapshot_payload)

            if result.change_detected:
                change_payload = {
                    "change_type":      "scene_change",
                    "description":      result.description,
                    "affected_objects": result.objects,
                    "change_magnitude": result.change_magnitude,
                    "confidence":       result.confidence,
                }
                self._mqtt.publish(Perception.SCENE_CHANGE, change_payload)
                logger.info("Scene change detected (magnitude=%.3f)", result.change_magnitude)

            jpeg_path = None
            if self._dc_cfg["enabled"]:
                jpeg_path = self._save_snapshot(frame, snapshot_payload)

            # Publish context for cerebrum: text summary + JPEG path for on-demand
            # vision queries. Cerebrum uses text by default; loads image when needed.
            context_payload = {
                "activity":        result.activity,
                "objects":         result.objects,
                "description":     result.description,
                "confidence":      result.confidence,
                "change_detected": result.change_detected,
                "image_path":      str(jpeg_path) if jpeg_path else None,
                "snapshot_number": self._snapshot_count,
            }
            self._mqtt.publish(Perception.SCENE_CONTEXT, context_payload)

            elapsed = time.monotonic() - t_start
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
        # Allow override at CLI for testing without editing config
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
