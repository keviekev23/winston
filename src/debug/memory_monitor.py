"""
Memory monitor — publishes system RAM stats every N seconds to:
  system/debug/memory_monitor  — per-process and system-wide view
  system/health                — simple subsystem heartbeat

Alerts to stderr when free memory drops below the configured threshold.
Start this early; it's how you catch leaks under sustained load.
"""

import logging
import threading
import time

import psutil

from src.transport.client import MQTTClient
from src.transport.topics import System

logger = logging.getLogger(__name__)

BYTES_PER_MB = 1024 * 1024


class MemoryMonitor:
    def __init__(
        self,
        mqtt: MQTTClient,
        interval_seconds: int = 30,
        alert_headroom_mb: int = 2048,
    ) -> None:
        self._mqtt = mqtt
        self._interval = interval_seconds
        self._alert_threshold_mb = alert_headroom_mb
        self._running = False
        self._thread: threading.Thread | None = None

        # Subsystems register themselves here so we can include them in health pings.
        # key: subsystem name, value: dict of extra fields
        self._registered: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Subsystem registration
    # ------------------------------------------------------------------

    def register(self, subsystem: str, **metadata) -> None:
        """Register a subsystem so it appears in health broadcasts."""
        self._registered[subsystem] = metadata
        logger.debug("Memory monitor: registered subsystem '%s'", subsystem)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="memory-monitor")
        self._thread.start()
        logger.info("Memory monitor started (interval=%ds, alert_threshold=%dMB)",
                    self._interval, self._alert_threshold_mb)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while self._running:
            self._publish_memory()
            self._publish_health()
            time.sleep(self._interval)

    def _publish_memory(self) -> None:
        vm = psutil.virtual_memory()
        total_mb = vm.total // BYTES_PER_MB
        used_mb  = vm.used  // BYTES_PER_MB
        free_mb  = vm.available // BYTES_PER_MB

        process = psutil.Process()
        process_mb = process.memory_info().rss // BYTES_PER_MB

        payload = {
            "total_mb":   total_mb,
            "used_mb":    used_mb,
            "free_mb":    free_mb,
            "process_mb": process_mb,
            "headroom_mb": free_mb,
        }

        self._mqtt.publish(System.MEMORY_MONITOR, payload)

        if free_mb < self._alert_threshold_mb:
            logger.warning(
                "LOW MEMORY: %dMB free (threshold %dMB) — process using %dMB",
                free_mb, self._alert_threshold_mb, process_mb,
            )
        else:
            logger.debug("Memory: %dMB free / %dMB total (process: %dMB)", free_mb, total_mb, process_mb)

    def _publish_health(self) -> None:
        for subsystem, metadata in self._registered.items():
            payload = {"subsystem": subsystem, "status": "ok", **metadata}
            self._mqtt.publish(System.HEALTH, payload)
