"""
Upload training data to Google Drive for Colab fine-tuning.

Reads from data/collection/audio/ and syncs to the configured Drive folder.

Upload strategy:
  - ALL low-confidence utterances (below stt.confidence_threshold)
  - A random sample of high-confidence utterances (flywheel.high_confidence_sample_rate)
    to provide positive examples and prevent catastrophic forgetting

Run after a collection period (daily, or manually before a training run).

Usage:
    python scripts/upload_training_data.py
    python scripts/upload_training_data.py --dry-run       # show what would be uploaded
    python scripts/upload_training_data.py --all           # upload everything regardless of confidence
"""

import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

CONFIG_PATH = Path("config/default.yaml")
COLLECTION_DIR = Path("data/collection/audio")


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def check_rclone() -> None:
    if shutil.which("rclone") is None:
        print("ERROR: rclone not found. See docs/flywheel_setup.md for installation.")
        sys.exit(1)


def check_remote(remote: str) -> None:
    result = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True)
    configured = [r.strip().rstrip(":") for r in result.stdout.splitlines()]
    if remote not in configured:
        print(f"ERROR: rclone remote '{remote}' not configured.")
        print("Run 'rclone config' and create a Google Drive remote.")
        print("See docs/flywheel_setup.md for step-by-step instructions.")
        sys.exit(1)


def collect_upload_candidates(
    collection_dir: Path,
    confidence_threshold: float,
    high_conf_sample_rate: float,
    upload_all: bool,
) -> tuple[list[Path], dict]:
    """
    Return (list of WAV paths to upload, stats dict).
    Each WAV must have a matching .json sidecar.
    """
    all_wavs = sorted(collection_dir.glob("*.wav"))
    if not all_wavs:
        return [], {"total": 0, "low_conf": 0, "high_conf_sampled": 0}

    low_conf_wavs  = []
    high_conf_wavs = []

    for wav in all_wavs:
        meta_path = wav.with_suffix(".json")
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)

        if upload_all:
            low_conf_wavs.append(wav)
        elif meta.get("confidence", 1.0) < confidence_threshold:
            low_conf_wavs.append(wav)
        else:
            high_conf_wavs.append(wav)

    if upload_all:
        selected = low_conf_wavs
        sampled_high = []
    else:
        # Sample a fraction of high-confidence utterances for diversity
        k = max(1, int(len(high_conf_wavs) * high_conf_sample_rate))
        sampled_high = random.sample(high_conf_wavs, min(k, len(high_conf_wavs)))
        selected = low_conf_wavs + sampled_high

    stats = {
        "total_wavs":       len(all_wavs),
        "low_conf":         len(low_conf_wavs),
        "high_conf_total":  len(high_conf_wavs),
        "high_conf_sampled": len(sampled_high),
        "uploading":        len(selected),
    }
    return selected, stats


def stage_for_upload(wavs: list[Path]) -> Path:
    """
    Copy selected WAVs + JSON sidecars to a temp staging directory.
    rclone will copy this directory to Drive.
    """
    staging = Path(tempfile.mkdtemp(prefix="winston-upload-"))
    for wav in wavs:
        shutil.copy2(wav, staging / wav.name)
        json_path = wav.with_suffix(".json")
        if json_path.exists():
            shutil.copy2(json_path, staging / json_path.name)
    return staging


def rclone_copy(staging: Path, remote: str, drive_folder: str, dry_run: bool) -> bool:
    dest = f"{remote}:{drive_folder}/audio"
    cmd = ["rclone", "copy", str(staging), dest, "--progress"]
    if dry_run:
        cmd.append("--dry-run")

    print(f"  rclone copy → {dest}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload training data to Google Drive")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be uploaded without actually uploading")
    parser.add_argument("--all", action="store_true",
                        help="Upload all utterances regardless of confidence")
    args = parser.parse_args()

    cfg = load_config()
    flywheel_cfg = cfg["flywheel"]
    stt_cfg      = cfg["stt"]

    remote            = flywheel_cfg["rclone_remote"]
    drive_data_folder = flywheel_cfg["drive_data_folder"]
    confidence_thresh = stt_cfg["confidence_threshold"]
    sample_rate       = flywheel_cfg["high_confidence_sample_rate"]

    check_rclone()
    check_remote(remote)

    print("\nWinston Flywheel — Upload Training Data")
    if args.dry_run:
        print("  [DRY RUN — nothing will be uploaded]")
    print()

    wavs, stats = collect_upload_candidates(
        COLLECTION_DIR, confidence_thresh, sample_rate, upload_all=args.all
    )

    print(f"  Collection dir:        {COLLECTION_DIR.resolve()}")
    print(f"  Total recordings:      {stats['total_wavs']}")
    print(f"  Low-confidence (<{confidence_thresh}): {stats['low_conf']}")
    if not args.all:
        print(f"  High-confidence:       {stats['high_conf_total']} total, "
              f"{stats['high_conf_sampled']} sampled ({int(sample_rate*100)}%)")
    print(f"  Uploading:             {stats['uploading']} files")
    print()

    if stats["uploading"] == 0:
        print("Nothing to upload. Run the perception service to collect some data first.")
        sys.exit(0)

    staging = stage_for_upload(wavs)
    try:
        success = rclone_copy(staging, remote, drive_data_folder, dry_run=args.dry_run)
    finally:
        shutil.rmtree(staging)

    if success:
        if args.dry_run:
            print("\nDry run complete — no files were transferred.")
        else:
            print(f"\nUpload complete.")
            print(f"Open the Colab notebook to run fine-tuning:")
            print(f"  notebooks/whisper_finetune.ipynb")
    else:
        print("\nUpload failed. Check rclone output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
