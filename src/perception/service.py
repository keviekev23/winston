"""
Perception service — the main runnable for Phase A audio perception.

Pipeline:
  sounddevice stream
    → SileroVAD (per 512-sample chunk)
    → VAD state machine (WAITING / SPEAKING / SILENCE)
    → WhisperSTT (on utterance end)
    → MQTT publish (perception/speech/transcript)
    → data collection (save WAV if enabled)

Run with:
  python -m src.perception.service
"""

import json
import logging
import queue
import signal
import sys
import threading
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sounddevice as sd
import yaml

from src.debug.memory_monitor import MemoryMonitor
from src.perception.stt import WhisperSTT
from src.perception.vad import SileroVAD
from src.transport.client import MQTTClient
from src.transport.topics import Perception, System

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# VAD state machine states
# ------------------------------------------------------------------
_WAITING  = "WAITING"
_SPEAKING = "SPEAKING"
_SILENCE  = "SILENCE"


def load_config(path: str = "config/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class PerceptionService:
    def __init__(self, config: dict) -> None:
        self._cfg = config
        self._audio_cfg = config["audio"]
        self._vad_cfg   = config["vad"]
        self._stt_cfg   = config["stt"]
        self._mqtt_cfg  = config["mqtt"]
        self._dc_cfg    = config["data_collection"]

        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._running = False

        # Initialise subsystems
        self._mqtt = MQTTClient(
            host=self._mqtt_cfg["host"],
            port=self._mqtt_cfg["port"],
            client_id=self._mqtt_cfg["client_id"],
            keepalive=self._mqtt_cfg["keepalive"],
        )
        self._vad = SileroVAD()
        self._stt = WhisperSTT(
            model=self._stt_cfg["model"],
            language=self._stt_cfg["language"],
        )
        self._monitor = MemoryMonitor(
            mqtt=self._mqtt,
            interval_seconds=config["memory_monitor"]["interval_seconds"],
            alert_headroom_mb=config["memory_monitor"]["alert_headroom_mb"],
        )
        self._monitor.register("perception")

        # Ensure data collection dir exists
        if self._dc_cfg["enabled"]:
            Path(self._dc_cfg["audio_dir"]).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._running = True
        self._mqtt.connect()
        self._monitor.start()

        self._mqtt.publish(System.HEALTH, {"subsystem": "perception", "status": "starting"})
        logger.info("Perception service starting — listening for speech...")

        sample_rate  = self._audio_cfg["sample_rate"]
        chunk_size   = self._audio_cfg["chunk_size"]

        def audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            if status:
                logger.warning("sounddevice status: %s", status)
            # indata is (chunk_size, channels); take channel 0, squeeze to 1-D
            self._audio_queue.put(indata[:, 0].copy())

        with sd.InputStream(
            samplerate=sample_rate,
            channels=self._audio_cfg["channels"],
            dtype=self._audio_cfg["dtype"],
            blocksize=chunk_size,
            callback=audio_callback,
        ):
            self._mqtt.publish(System.HEALTH, {"subsystem": "perception", "status": "ok"})
            self._process_loop()

    def stop(self) -> None:
        self._running = False
        self._monitor.stop()
        self._mqtt.disconnect()

    # ------------------------------------------------------------------
    # Main processing loop
    # ------------------------------------------------------------------

    def _process_loop(self) -> None:
        sample_rate   = self._audio_cfg["sample_rate"]
        chunk_size    = self._audio_cfg["chunk_size"]
        max_samples   = int(self._vad_cfg["max_speech_seconds"] * sample_rate)

        onset_threshold  = self._vad_cfg["threshold_onset"]
        offset_threshold = self._vad_cfg["threshold_offset"]
        min_speech_frames = self._vad_cfg["min_speech_frames"]
        silence_frames_needed = self._vad_cfg["silence_frames"]

        state = _WAITING
        speech_buffer: list[np.ndarray] = []
        onset_counter  = 0
        silence_counter = 0

        while self._running:
            try:
                chunk = self._audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            prob = self._vad.speech_probability(chunk)

            if state == _WAITING:
                if prob >= onset_threshold:
                    onset_counter += 1
                    if onset_counter >= min_speech_frames:
                        state = _SPEAKING
                        speech_buffer = []
                        silence_counter = 0
                        onset_counter = 0
                        self._publish_vad(is_speaking=True)
                        logger.debug("VAD: speech started")
                else:
                    onset_counter = 0

            elif state == _SPEAKING:
                speech_buffer.append(chunk)

                if prob < offset_threshold:
                    silence_counter += 1
                    if silence_counter >= silence_frames_needed:
                        # End of utterance
                        state = _WAITING
                        self._publish_vad(is_speaking=False)
                        self._vad.reset_states()
                        logger.debug("VAD: speech ended (%d chunks buffered)", len(speech_buffer))
                        self._handle_utterance(speech_buffer)
                        speech_buffer = []
                        silence_counter = 0
                        onset_counter = 0
                else:
                    silence_counter = 0

                # Hard cap — force transcription if buffer grows too long
                buffered_samples = len(speech_buffer) * chunk_size
                if buffered_samples >= max_samples:
                    logger.warning("VAD: max speech duration reached — forcing transcription")
                    state = _WAITING
                    self._publish_vad(is_speaking=False)
                    self._vad.reset_states()
                    self._handle_utterance(speech_buffer)
                    speech_buffer = []
                    silence_counter = 0
                    onset_counter = 0

    # ------------------------------------------------------------------
    # Utterance handling
    # ------------------------------------------------------------------

    def _handle_utterance(self, chunks: list[np.ndarray]) -> None:
        audio = np.concatenate(chunks)
        result = self._stt.transcribe(audio)

        if not result.text:
            logger.debug("STT: empty transcript — skipping")
            return

        logger.info(
            "STT: \"%s\" (confidence=%.2f, no_speech=%.2f)",
            result.text, result.confidence, result.no_speech_prob,
        )

        low_confidence = result.confidence < self._stt_cfg["confidence_threshold"]

        payload = {
            "text":               result.text,
            "confidence":         round(result.confidence, 3),
            "speaker_id":         None,   # populated by diarization in Layer 3
            "speaker_confidence": None,
            "language":           result.language,
            "low_confidence":     low_confidence,
        }
        self._mqtt.publish(Perception.TRANSCRIPT, payload)

        if self._dc_cfg["enabled"]:
            save_all = self._dc_cfg.get("save_all", True)
            if save_all or low_confidence:
                self._save_audio(audio, payload)

    def _publish_vad(self, is_speaking: bool) -> None:
        self._mqtt.publish(Perception.VAD, {"is_speaking": is_speaking})

    def _save_audio(self, audio: np.ndarray, transcript_payload: dict) -> None:
        """Save audio WAV + transcript JSON sidecar for the fine-tuning flywheel."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        base = Path(self._dc_cfg["audio_dir"]) / ts

        # WAV
        wav_path = base.with_suffix(".wav")
        audio_int16 = (audio * 32767).astype(np.int16)
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)   # 16-bit
            wf.setframerate(self._audio_cfg["sample_rate"])
            wf.writeframes(audio_int16.tobytes())

        # Sidecar JSON
        meta_path = base.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(transcript_payload, f, indent=2)

        logger.debug("Saved utterance: %s", wav_path)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    config = load_config()
    service = PerceptionService(config)

    def _shutdown(sig, frame):
        logger.info("Shutting down perception service...")
        service.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    service.start()


if __name__ == "__main__":
    main()
