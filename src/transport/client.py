"""
MQTT client wrapper around paho-mqtt.

All publish calls auto-inject a UTC timestamp. Payloads are always JSON.
Reconnect is handled automatically by paho's built-in loop.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable

import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)


class MQTTClient:
    def __init__(self, host: str, port: int, client_id: str, keepalive: int = 60) -> None:
        self._host = host
        self._port = port
        self._keepalive = keepalive

        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
        )
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message_raw

        # topic → list of handlers
        self._handlers: dict[str, list[Callable[[dict], None]]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect and start background network loop."""
        self._client.connect(self._host, self._port, self._keepalive)
        self._client.loop_start()
        logger.info("MQTT connecting to %s:%s", self._host, self._port)

    def disconnect(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()
        logger.info("MQTT disconnected")

    def publish(self, topic: str, payload: dict[str, Any], qos: int = 0) -> None:
        """Publish a dict payload as JSON. Timestamp is injected automatically."""
        if "timestamp" not in payload:
            payload["timestamp"] = _utc_now()
        self._client.publish(topic, json.dumps(payload), qos=qos)

    def subscribe(self, topic: str, handler: Callable[[dict], None]) -> None:
        """Subscribe to a topic and register a handler for decoded JSON payloads."""
        if topic not in self._handlers:
            self._handlers[topic] = []
            self._client.subscribe(topic)
        self._handlers[topic].append(handler)

    # ------------------------------------------------------------------
    # Internal callbacks
    # ------------------------------------------------------------------

    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        if reason_code == 0:
            logger.info("MQTT connected")
            # Re-subscribe after reconnect
            for topic in self._handlers:
                client.subscribe(topic)
        else:
            logger.error("MQTT connection failed: reason_code=%s", reason_code)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties) -> None:
        if reason_code != 0:
            logger.warning("MQTT unexpected disconnect (reason_code=%s) — will reconnect", reason_code)

    def _on_message_raw(self, client, userdata, message) -> None:
        topic = message.topic
        try:
            payload = json.loads(message.payload.decode())
        except json.JSONDecodeError:
            logger.warning("Non-JSON message on %s — ignoring", topic)
            return

        handlers = self._handlers.get(topic, [])
        for handler in handlers:
            try:
                handler(payload)
            except Exception:
                logger.exception("Handler error on topic %s", topic)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
