"""
Pre-session health check for data collection.

Runs four checks in <10s and prints ✓/✗ for each. Exit 0 = all pass, 1 = any failure.
Run this before every collection session to catch config/hardware issues early.

Usage:
    python scripts/check_collection.py
"""

import sys
import tempfile
from pathlib import Path

import yaml

CONFIG_PATH = Path("config/default.yaml")

BOLD  = "\033[1m"
RESET = "\033[0m"
GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW = "\033[93m"

_ok  = f"{GREEN}✓{RESET}"
_err = f"{RED}✗{RESET}"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_mqtt(cfg: dict) -> tuple[bool, str]:
    """Connect to broker, publish a test message, disconnect."""
    try:
        from src.transport.client import MQTTClient

        host = cfg["mqtt"]["host"]
        port = cfg["mqtt"]["port"]

        client = MQTTClient(host=host, port=port, client_id="check-collection", keepalive=10)
        client.connect()
        client.publish("system/check_collection", {"status": "test"})
        client.disconnect()
        return True, f"broker reachable at {host}:{port}"
    except Exception as e:
        return False, (
            f"cannot connect to MQTT broker ({e})\n"
            f"  → run: brew services start mosquitto"
        )


def check_camera() -> tuple[bool, str]:
    """Open camera device 0, capture one frame, close."""
    try:
        from src.perception.camera import Camera

        with Camera() as cam:
            frame = cam.capture()

        if frame is None:
            return False, "camera opened but capture() returned None"

        w, h = frame.size
        return True, f"camera OK — frame {w}x{h}"
    except RuntimeError as e:
        return False, (
            f"{e}\n"
            f"  → check System Preferences → Privacy → Camera"
        )
    except Exception as e:
        return False, str(e)


def check_microphone() -> tuple[bool, str]:
    """Record 0.5s of audio and verify it has non-zero signal."""
    try:
        import numpy as np
        import sounddevice as sd

        audio = sd.rec(
            int(0.5 * 16000),
            samplerate=16000,
            channels=1,
            dtype="float32",
            blocking=True,
        )
        peak = float(np.abs(audio).max())

        if peak < 0.001:
            return False, (
                f"microphone captured silence (peak={peak:.4f})\n"
                f"  → check System Preferences → Privacy → Microphone\n"
                f"  → check that the built-in microphone is the default input device"
            )
        return True, f"microphone OK — peak amplitude {peak:.3f}"
    except Exception as e:
        return False, (
            f"cannot access microphone ({e})\n"
            f"  → check System Preferences → Privacy → Microphone"
        )


def check_directories(cfg: dict) -> tuple[bool, str]:
    """Verify collection directories exist and are writable."""
    audio_dir  = Path(cfg["data_collection"]["audio_dir"])
    images_dir = Path(cfg["data_collection"]["images_dir"])

    failures = []
    for d in (audio_dir, images_dir):
        d.mkdir(parents=True, exist_ok=True)
        try:
            sentinel = d / ".check_write_test"
            sentinel.write_text("ok")
            sentinel.unlink()
        except OSError as e:
            failures.append(f"{d}: {e}")

    if failures:
        return False, "directory not writable:\n  " + "\n  ".join(failures)

    return True, f"audio → {audio_dir}  images → {images_dir}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        cfg = load_config()
    except FileNotFoundError:
        print(f"{RED}ERROR:{RESET} config/default.yaml not found — run from project root")
        sys.exit(1)

    print(f"\n{BOLD}Winston Collection Health Check{RESET}\n")

    checks = [
        ("MQTT broker",  lambda: check_mqtt(cfg)),
        ("Camera",       check_camera),
        ("Microphone",   check_microphone),
        ("Directories",  lambda: check_directories(cfg)),
    ]

    all_passed = True
    for name, fn in checks:
        passed, detail = fn()
        icon = _ok if passed else _err
        print(f"  {icon}  {BOLD}{name}{RESET}")
        if not passed:
            all_passed = False
        # Print detail on failure (or as subtle info on pass)
        indent = "     "
        if not passed:
            for line in detail.splitlines():
                print(f"{RED}{indent}{line}{RESET}")
        else:
            print(f"{indent}{detail}")

    print()
    if all_passed:
        print(f"{GREEN}{BOLD}All checks passed — ready to collect.{RESET}\n")
        sys.exit(0)
    else:
        print(f"{RED}{BOLD}One or more checks failed — fix before collecting.{RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
