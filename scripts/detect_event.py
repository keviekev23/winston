"""
Short-lived event detection runner for Phase A VLM evaluation.

Loads a scenario YAML (from prompts/event_detection/), constructs a composite
classification prompt from the event list, and runs the selected VLM adapter
in a loop. When any event accumulates --confirm-frames consecutive detections,
it triggers: saves the frame as a JPEG, emits an audio beep, and exits.

Run with:
    python scripts/detect_event.py --scenario prompts/event_detection/cooking_prep.yaml
    python scripts/detect_event.py --scenario prompts/event_detection/cooking_prep.yaml \\
        --confirm-frames 5 --interval 1.0 --adapter moondream2

Output:
    Per-frame log to stdout: detected label, confidence, latency_ms
    On trigger: saves JPEG + detection JSON to data/detection/
    Audio beep via afplay (macOS)

This is NOT a long-running service — it exits after the first confirmed event.
Evaluation cheatsheet: see docs/phases/phase_a_perception.md
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)

DETECTION_DIR = Path("data/detection")
BOLD  = "\033[1m"
GREEN = "\033[92m"
RESET = "\033[0m"
NONE_LABEL = "NONE"


# ---------------------------------------------------------------------------
# Scenario loading + prompt construction
# ---------------------------------------------------------------------------

def load_scenario(path: Path) -> dict:
    with open(path) as f:
        scenario = yaml.safe_load(f)
    events = scenario.get("events", [])
    if not events:
        logger.error("Scenario has no events: %s", path)
        sys.exit(1)
    return scenario


def build_prompt(events: list[dict]) -> str:
    """
    Auto-generate a structured classification prompt from the event list.

    Small VLMs respond more reliably to forced label classification than
    free-form description. The prompt asks for exactly one label from the list.
    """
    label_lines = "\n".join(
        f"- {e['label']}: {e['description']}"
        for e in events
    )
    return (
        "Classify the current activity in this kitchen image.\n"
        f"Respond with EXACTLY one of these labels:\n{label_lines}\n"
        f"- {NONE_LABEL}: none of the above are clearly visible\n\n"
        "Respond with the label ONLY. No explanation."
    )


def make_confirm_map(events: list[dict]) -> dict[str, int]:
    """Build {label: confirm_frames} from scenario events."""
    return {e["label"]: e.get("confirm_frames", 3) for e in events}


# ---------------------------------------------------------------------------
# Trigger: save image + metadata, emit beep
# ---------------------------------------------------------------------------

def trigger_event(
    frame: Image.Image,
    event_label: str,
    event_id: str,
    scenario_path: Path,
    prompt: str,
    result_description: str,
    latency_ms: float,
    confidence: float,
) -> Path:
    DETECTION_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    stem = f"{ts}_{event_id}"

    jpeg_path = DETECTION_DIR / f"{stem}.jpg"
    frame.save(str(jpeg_path), "JPEG", quality=85)

    meta = {
        "scenario":       str(scenario_path),
        "event_id":       event_id,
        "event_label":    event_label,
        "prompt":         prompt,
        "detected_label": event_label,
        "description":    result_description,
        "confidence":     confidence,
        "latency_ms":     latency_ms,
        "timestamp":      ts,
    }
    meta_path = DETECTION_DIR / f"{stem}_detection.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Audio cue — afplay is macOS native, no extra dependencies
    try:
        subprocess.run(
            ["afplay", "/System/Library/Sounds/Glass.aiff"],
            check=False,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.debug("afplay not available — skipping audio cue")

    return jpeg_path


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(
    scenario_path: Path,
    adapter_name: str,
    interval: float,
    confirm_frames_override: int | None,
) -> None:
    scenario = load_scenario(scenario_path)
    events   = scenario["events"]
    prompt   = build_prompt(events)

    confirm_map = make_confirm_map(events)
    if confirm_frames_override is not None:
        confirm_map = {label: confirm_frames_override for label in confirm_map}

    known_labels = set(confirm_map.keys()) | {NONE_LABEL}

    # Load adapter
    adapter = _load_adapter(adapter_name)
    adapter.load()

    # Load camera
    from src.perception.camera import Camera
    camera = Camera()
    camera.open()

    # Per-event consecutive hit counters
    counters: dict[str, int] = {label: 0 for label in confirm_map}

    print(f"\n{BOLD}Winston event detection — scenario: {scenario['scenario']}{RESET}")
    print(f"  Events: {', '.join(e['id'] for e in events)}")
    print(f"  Adapter: {adapter_name}  |  Interval: {interval}s\n")

    try:
        frame_num = 0
        while True:
            t_start = time.monotonic()
            frame_num += 1

            frame = camera.capture()
            if frame is None:
                logger.warning("Failed to capture frame — retrying")
                time.sleep(interval)
                continue

            result = adapter.detect(frame, prompt)

            label = result.detected_label
            is_known = label in known_labels

            # Update per-event counters
            for tracked_label in list(counters):
                if label == tracked_label:
                    counters[tracked_label] += 1
                else:
                    counters[tracked_label] = 0  # reset on any different detection

            # Status line
            counter_str = "  ".join(
                f"{lbl}={n}" for lbl, n in counters.items() if n > 0
            )
            logger.info(
                "Frame %d: label=%-25s  conf=%.1f  latency=%5.0fms  %s",
                frame_num, label, result.confidence, result.latency_ms,
                f"[{counter_str}]" if counter_str else "",
            )

            # Check for any event that has hit confirm threshold
            for tracked_label, count in counters.items():
                threshold = confirm_map.get(tracked_label)
                if threshold and count >= threshold and tracked_label != NONE_LABEL:
                    # Find the event id for this label
                    event_id = next(
                        (e["id"] for e in events if e["label"] == tracked_label),
                        tracked_label.lower(),
                    )
                    jpeg_path = trigger_event(
                        frame=frame,
                        event_label=tracked_label,
                        event_id=event_id,
                        scenario_path=scenario_path,
                        prompt=prompt,
                        result_description=result.description,
                        latency_ms=result.latency_ms,
                        confidence=result.confidence,
                    )
                    print(
                        f"\n{GREEN}{BOLD}EVENT TRIGGERED: {event_id}{RESET}\n"
                        f"  Label:      {tracked_label}\n"
                        f"  Confidence: {result.confidence:.1f}\n"
                        f"  Latency:    {result.latency_ms:.0f}ms\n"
                        f"  Saved:      {jpeg_path}\n"
                    )
                    return  # exit after first confirmed event

            # Sleep remainder of interval
            elapsed = time.monotonic() - t_start
            remaining = max(0.0, interval - elapsed)
            if remaining > 0:
                time.sleep(remaining)

    finally:
        camera.close()
        adapter.unload()


# ---------------------------------------------------------------------------
# Latency benchmark mode
# ---------------------------------------------------------------------------

def run_benchmark(
    scenario_path: Path,
    adapter_name: str,
    num_frames: int,
) -> None:
    """
    Measure warm inference latency in a controlled way.

    This is the ONLY valid source of latency numbers for Phase A exit gate
    assessment. Run it with all other applications closed — concurrent
    CPU/memory load invalidates measurements.

    Runs one JIT warmup frame (discarded from stats), then `num_frames` timed
    frames. Reports mean/min/max/p50/p90 and fps estimate.
    """
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  LATENCY BENCHMARK MODE  {RESET}")
    print(f"{'=' * 60}")
    print("  IMPORTANT: Close all other applications before proceeding.")
    print("  Concurrent CPU/memory load will skew latency measurements.")
    print("  This is the Phase A exit gate measurement — results will be")
    print("  recorded as the definitive latency baseline for VLM selection.")
    print(f"{'=' * 60}\n")

    try:
        answer = input("  Proceed with benchmark? [y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return
    if answer not in ("y", "yes"):
        print("Benchmark cancelled.")
        return

    scenario = load_scenario(scenario_path)
    prompt   = build_prompt(scenario["events"])

    adapter = _load_adapter(adapter_name)
    adapter.load()

    from src.perception.camera import Camera
    camera = Camera()
    camera.open()

    latencies: list[float] = []

    try:
        print(f"\n  Warmup frame (JIT cold — excluded from stats)...")
        frame  = camera.capture()
        warmup = adapter.detect(frame, prompt)
        print(f"  JIT cold latency: {warmup.latency_ms:.0f}ms  label={warmup.detected_label}\n")

        print(f"  Running {num_frames} benchmark frames...")
        for i in range(num_frames):
            frame  = camera.capture()
            result = adapter.detect(frame, prompt)
            latencies.append(result.latency_ms)
            print(f"    Frame {i+1:3d}: {result.latency_ms:6.0f}ms  label={result.detected_label}")
    finally:
        camera.close()
        adapter.unload()

    if not latencies:
        print("\nNo frames collected.")
        return

    s       = sorted(latencies)
    mean_ms = sum(latencies) / len(latencies)
    p50_ms  = s[len(s) // 2]
    p90_ms  = s[int(len(s) * 0.9)]

    gate_met = mean_ms < 1000
    gate_str = f"{GREEN}✓ MET{RESET}" if gate_met else f"{RED}✗ NOT MET{RESET}"

    print(f"\n{BOLD}  Benchmark results ({num_frames} frames, adapter={adapter_name}):{RESET}")
    print(f"    mean={mean_ms:.0f}ms  min={s[0]:.0f}ms  max={s[-1]:.0f}ms")
    print(f"    p50={p50_ms:.0f}ms  p90={p90_ms:.0f}ms")
    print(f"    estimated fps: {1000/mean_ms:.2f} (mean-based)")
    print(f"\n  Phase A exit gate (<1000ms mean): {gate_str}")
    print(
        f"\n  Record these numbers in LESSONS_LEARNED.md and the phase_a_perception.md "
        f"exit gate section before locking the VLM model.\n"
    )


def _load_adapter(name: str):
    if name == "internvl2_1b":
        from src.perception.vlm.internvl2 import InternVL2Adapter
        return InternVL2Adapter()
    elif name == "moondream2":
        from src.perception.vlm.moondream import MoondreamAdapter
        return MoondreamAdapter()
    else:
        logger.error("Unknown adapter: %s. Choices: internvl2_1b, moondream2", name)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect a kitchen event using on-device VLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/detect_event.py --scenario prompts/event_detection/cooking_prep.yaml
  python scripts/detect_event.py --scenario prompts/event_detection/cooking_prep.yaml \\
      --confirm-frames 5 --interval 1.5

Evaluation cheatsheet: docs/phases/phase_a_perception.md
        """,
    )
    parser.add_argument(
        "--scenario", required=True, type=Path,
        help="Path to scenario YAML (e.g. prompts/event_detection/cooking_prep.yaml)",
    )
    parser.add_argument(
        "--adapter", default="internvl2_1b",
        choices=["internvl2_1b", "moondream2"],
        help="VLM adapter to use (default: internvl2_1b)",
    )
    parser.add_argument(
        "--interval", type=float, default=1.0,
        help="Seconds between frames (default: 1.0)",
    )
    parser.add_argument(
        "--confirm-frames", type=int, default=None,
        dest="confirm_frames",
        help="Override confirm_frames for all events (default: per-event from YAML)",
    )
    parser.add_argument(
        "--benchmark-latency", type=int, default=None,
        metavar="N",
        dest="benchmark_latency",
        help=(
            "Latency benchmark mode: run N warm frames and report latency stats, then exit. "
            "Run with all other applications closed for valid Phase A exit gate measurements."
        ),
    )

    args = parser.parse_args()

    if not args.scenario.exists():
        logger.error("Scenario not found: %s", args.scenario)
        sys.exit(1)

    if args.benchmark_latency is not None:
        run_benchmark(
            scenario_path=args.scenario,
            adapter_name=args.adapter,
            num_frames=args.benchmark_latency,
        )
        return

    run(
        scenario_path=args.scenario,
        adapter_name=args.adapter,
        interval=args.interval,
        confirm_frames_override=args.confirm_frames,
    )


if __name__ == "__main__":
    main()
