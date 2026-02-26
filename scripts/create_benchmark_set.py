"""
Create a benchmark recording set for Whisper STT calibration.

Records each utterance from config/benchmark_utterances.yaml with ground-truth text.
Saves to data/benchmark/{id}.wav + data/benchmark/{id}.json.

Run once BEFORE any fine-tuning to establish a fixed evaluation baseline.
Re-running will prompt before overwriting existing recordings.

Usage:
    python scripts/create_benchmark_set.py
    python scripts/create_benchmark_set.py --difficulty easy        # only easy tier
    python scripts/create_benchmark_set.py --ids hard_011 hard_002  # specific utterances
    python scripts/create_benchmark_set.py --duration 6             # longer recording window
"""

import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
import yaml

SAMPLE_RATE   = 16000
DEFAULT_SECS  = 5        # recording window per utterance
BENCHMARK_DIR = Path("data/benchmark")
UTTERANCES_FILE = Path("config/benchmark_utterances.yaml")

DIFFICULTY_LABELS = {
    "easy":   "EASY  ",
    "medium": "MEDIUM",
    "hard":   "HARD  ",
}
DIFFICULTY_COLORS = {
    "easy":   "\033[92m",   # green
    "medium": "\033[93m",   # yellow
    "hard":   "\033[91m",   # red
}
RESET = "\033[0m"
BOLD  = "\033[1m"


def load_utterances(difficulty: str | None = None, ids: list[str] | None = None) -> list[dict]:
    with open(UTTERANCES_FILE) as f:
        data = yaml.safe_load(f)
    utterances = data["utterances"]
    if difficulty:
        utterances = [u for u in utterances if u["difficulty"] == difficulty]
    if ids:
        utterances = [u for u in utterances if u["id"] in ids]
    return utterances


def record_audio(duration_secs: int) -> np.ndarray:
    """Record mono float32 audio for the given duration."""
    samples = duration_secs * SAMPLE_RATE
    audio = sd.rec(samples, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0]   # 1-D


def save_wav(path: Path, audio: np.ndarray) -> None:
    audio_int16 = (audio * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())


def save_metadata(path: Path, utterance: dict) -> None:
    meta = {
        "id":         utterance["id"],
        "text":       utterance["text"],
        "difficulty": utterance["difficulty"],
        "notes":      utterance.get("notes", ""),
        "sample_rate": SAMPLE_RATE,
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def record_utterance(utterance: dict, duration_secs: int, force: bool) -> bool:
    """Interactively record one utterance. Returns True if recorded, False if skipped."""
    utt_id    = utterance["id"]
    difficulty = utterance["difficulty"]
    text      = utterance["text"]
    notes     = utterance.get("notes", "")

    wav_path  = BENCHMARK_DIR / f"{utt_id}.wav"
    meta_path = BENCHMARK_DIR / f"{utt_id}.json"

    # Check for existing recording
    if wav_path.exists() and not force:
        answer = input(f"  '{utt_id}' already recorded. Overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print(f"  Skipped {utt_id}.")
            return False

    color = DIFFICULTY_COLORS.get(difficulty, "")
    label = DIFFICULTY_LABELS.get(difficulty, difficulty.upper())

    print()
    print(f"  {color}{BOLD}[{label}]{RESET}  {utt_id}")
    print(f"  {BOLD}Say:{RESET} \"{text}\"")
    if notes:
        print(f"  {BOLD}Note:{RESET} {notes}")
    print(f"  Recording for {duration_secs}s. Press Enter when ready to record...")
    input()

    print(f"  {BOLD}Recording...{RESET}", end="", flush=True)
    audio = record_audio(duration_secs)
    print(f" done.")

    # Quick amplitude check — warn if the audio looks empty
    peak = float(np.abs(audio).max())
    if peak < 0.01:
        print(f"  \033[91mWARNING: Very low audio level (peak={peak:.4f}). Check microphone.\033[0m")
        answer = input("  Keep this recording? [y/N] ").strip().lower()
        if answer != "y":
            print("  Discarded — will retry.")
            return record_utterance(utterance, duration_secs, force=True)

    save_wav(wav_path, audio)
    save_metadata(meta_path, utterance)
    print(f"  Saved: {wav_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Record benchmark utterances for STT calibration")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"],
                        help="Record only a specific difficulty tier")
    parser.add_argument("--ids", nargs="+", metavar="ID",
                        help="Record specific utterance IDs only")
    parser.add_argument("--duration", type=int, default=DEFAULT_SECS,
                        help=f"Recording window per utterance in seconds (default: {DEFAULT_SECS})")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing recordings without prompting")
    args = parser.parse_args()

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    utterances = load_utterances(difficulty=args.difficulty, ids=args.ids)
    if not utterances:
        print("No utterances matched the filters.")
        sys.exit(1)

    tiers = {u["difficulty"] for u in utterances}
    print(f"\n{BOLD}Winston Benchmark Recording{RESET}")
    print(f"  {len(utterances)} utterances  |  tiers: {', '.join(sorted(tiers))}")
    print(f"  Recording window: {args.duration}s per utterance")
    print(f"  Output: {BENCHMARK_DIR.resolve()}")
    print()
    print("Tips:")
    print("  - Speak naturally at conversational distance from the microphone")
    print("  - Vary your rate/volume slightly — don't perform, just talk")
    print("  - For 'hard' utterances, say them as you naturally would in the kitchen")
    print("  - Re-record if you stumble — we'll prompt on low audio levels")
    print()

    by_difficulty = {"easy": [], "medium": [], "hard": []}
    for u in utterances:
        by_difficulty[u["difficulty"]].append(u)

    recorded = 0
    skipped  = 0

    for tier in ["easy", "medium", "hard"]:
        tier_utterances = by_difficulty.get(tier, [])
        if not tier_utterances:
            continue

        color = DIFFICULTY_COLORS[tier]
        print(f"\n{color}{BOLD}━━━ {tier.upper()} ({len(tier_utterances)} utterances) ━━━{RESET}")

        for i, utt in enumerate(tier_utterances, 1):
            print(f"\n  [{i}/{len(tier_utterances)}]", end="")
            ok = record_utterance(utt, args.duration, force=args.force)
            if ok:
                recorded += 1
            else:
                skipped += 1

    print(f"\n{BOLD}Done.{RESET} {recorded} recorded, {skipped} skipped.")
    print()
    print("Next step — run a baseline evaluation:")
    print("  python scripts/evaluate_whisper.py")


if __name__ == "__main__":
    main()
