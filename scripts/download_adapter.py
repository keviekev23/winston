"""
Download a fine-tuned Whisper model from Google Drive and convert it to MLX format
for edge inference.

After the Colab notebook saves a merged model to Drive, this script:
  1. Syncs it to data/adapters/{cycle_name}/hf/
  2. Converts it to MLX format at data/adapters/{cycle_name}/mlx/
  3. Updates config/default.yaml to point to the new model

Usage:
    python scripts/download_adapter.py --cycle cycle-1
    python scripts/download_adapter.py --cycle cycle-1 --dry-run
    python scripts/download_adapter.py --list    # list available cycles on Drive
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

CONFIG_PATH  = Path("config/default.yaml")
ADAPTERS_DIR = Path("data/adapters")


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def save_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def rclone_run(cmd: list[str], dry_run: bool = False) -> bool:
    if dry_run:
        cmd.append("--dry-run")
    result = subprocess.run(cmd)
    return result.returncode == 0


def list_drive_cycles(remote: str, drive_models_folder: str) -> None:
    dest = f"{remote}:{drive_models_folder}/merged"
    print(f"\nAvailable cycles on Drive ({dest}):\n")
    result = subprocess.run(["rclone", "lsd", dest], capture_output=True, text=True)
    if result.returncode != 0:
        print("  (none found, or folder doesn't exist yet)")
        print(f"  {result.stderr.strip()}")
    else:
        print(result.stdout or "  (empty)")


def download_merged_model(
    remote: str,
    drive_models_folder: str,
    cycle: str,
    local_hf_path: Path,
    dry_run: bool,
) -> bool:
    src  = f"{remote}:{drive_models_folder}/merged/{cycle}"
    dest = str(local_hf_path)
    local_hf_path.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading: {src} → {dest}")
    return rclone_run(["rclone", "sync", src, dest, "--progress"], dry_run=dry_run)


def convert_to_mlx(hf_path: Path, mlx_path: Path) -> bool:
    """
    Convert HuggingFace Whisper weights to MLX format using mlx_whisper.convert.
    Quantizes to INT4 (q4) to keep model under ~500MB.
    """
    mlx_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", "-m", "mlx_whisper.convert",
        "--hf-path", str(hf_path),
        "--mlx-path", str(mlx_path),
        "-q",             # enable quantization
        "--q-bits", "4",  # INT4 → ~500MB for small
    ]
    print(f"  Converting to MLX: {mlx_path}")
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def update_config_model_path(cfg: dict, mlx_path: Path) -> None:
    """Update stt.model in config to point to the new local MLX model."""
    old_model = cfg["stt"]["model"]
    cfg["stt"]["model"] = str(mlx_path)
    save_config(cfg)
    print(f"  config/default.yaml updated:")
    print(f"    stt.model: {old_model}")
    print(f"           → {mlx_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download fine-tuned adapter and convert to MLX")
    parser.add_argument("--cycle", help="Cycle name (e.g. 'cycle-1')")
    parser.add_argument("--list",  action="store_true", help="List available cycles on Drive")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without downloading or converting")
    parser.add_argument("--no-update-config", action="store_true",
                        help="Skip updating config/default.yaml after conversion")
    args = parser.parse_args()

    cfg = load_config()
    flywheel_cfg = cfg["flywheel"]
    remote              = flywheel_cfg["rclone_remote"]
    drive_models_folder = flywheel_cfg["drive_models_folder"]

    if args.list:
        list_drive_cycles(remote, drive_models_folder)
        return

    if not args.cycle:
        parser.error("--cycle is required (e.g. --cycle cycle-1). Use --list to see available cycles.")

    cycle = args.cycle
    local_hf_path  = ADAPTERS_DIR / cycle / "hf"
    local_mlx_path = ADAPTERS_DIR / cycle / "mlx"

    print(f"\nWinston Flywheel — Download Adapter")
    print(f"  Cycle: {cycle}")
    if args.dry_run:
        print("  [DRY RUN]")
    print()

    # Step 1: Download
    ok = download_merged_model(remote, drive_models_folder, cycle, local_hf_path, dry_run=args.dry_run)
    if not ok:
        print("\nDownload failed. Check rclone output above.")
        sys.exit(1)

    if args.dry_run:
        print("\nDry run complete.")
        return

    # Step 2: Convert to MLX
    print()
    ok = convert_to_mlx(local_hf_path, local_mlx_path)
    if not ok:
        print("\nMLX conversion failed.")
        print("Make sure mlx-whisper is installed: pip install mlx-whisper")
        sys.exit(1)

    # Step 3: Update config
    print()
    if not args.no_update_config:
        update_config_model_path(cfg, local_mlx_path)
    else:
        print(f"  Skipping config update. To use this model, set in config/default.yaml:")
        print(f"    stt.model: {local_mlx_path}")

    print(f"\nDone. Run an evaluation to measure improvement:")
    print(f"  python scripts/evaluate_whisper.py --label {cycle}")


if __name__ == "__main__":
    main()
