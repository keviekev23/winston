"""
Review collected scene images and correct activity labels.

For each image that has a Claude annotation (_gt.json) but no Kevin review yet:
  - Opens the JPEG in macOS Preview so you can see it
  - Prints SmolVLM2 output, Claude annotation, and the discrepancy flags
  - Prompts for confirmation or correction of the activity label
  - Writes kevin_activity to the _gt.json

Ground truth resolution order (highest wins):
  1. kevin_activity  — set by this script
  2. claude_annotation.activity — fallback if not reviewed
  3. "unknown" — if neither exists

Usage:
    python scripts/review_scene_labels.py             # all un-reviewed images
    python scripts/review_scene_labels.py --limit 20  # only the 20 most recent
    python scripts/review_scene_labels.py --force     # re-review already-reviewed images
    python scripts/review_scene_labels.py --report    # summary of review coverage, no prompts
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

COLLECTION_DIR = Path("data/collection/images")
VALID_ACTIVITIES = {"cooking", "eating", "cleaning", "idle", "unknown"}

BOLD   = "\033[1m"
RESET  = "\033[0m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(collection_dir: Path) -> None:
    gt_files = sorted(collection_dir.glob("*_gt.json"))
    if not gt_files:
        print("No annotated files found. Run label_scene_data.py first.")
        return

    reviewed = sum(1 for p in gt_files if _has_kevin_review(p))
    total    = len(gt_files)
    pct      = 100 * reviewed // total if total else 0

    print(f"\n{BOLD}Review Coverage{RESET}")
    print(f"  Annotated (Claude GT):  {total}")
    print(f"  Reviewed (Kevin):       {reviewed}  ({pct}%)")
    print(f"  Un-reviewed:            {total - reviewed}")

    # Activity distribution (Kevin labels win where present)
    from collections import Counter
    activity_counts: Counter = Counter()
    for gt_path in gt_files:
        label = _effective_activity(gt_path)
        activity_counts[label] += 1

    print(f"\n{BOLD}Activity Distribution (effective GT){RESET}")
    for act, cnt in sorted(activity_counts.items(), key=lambda x: -x[1]):
        print(f"  {act:<12} {cnt}")
    print()


def _has_kevin_review(gt_path: Path) -> bool:
    try:
        with open(gt_path) as f:
            gt = json.load(f)
        return gt.get("kevin_activity") is not None
    except Exception:
        return False


def _effective_activity(gt_path: Path) -> str:
    try:
        with open(gt_path) as f:
            gt = json.load(f)
        return gt.get("kevin_activity") or gt.get("claude_annotation", {}).get("activity", "unknown")
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Review loop
# ---------------------------------------------------------------------------

def open_in_preview(jpg_path: Path) -> None:
    """Open image in macOS Preview (non-blocking)."""
    try:
        subprocess.Popen(["open", "-a", "Preview", str(jpg_path)])
    except Exception:
        pass  # not fatal — user can find it manually


def review_image(jpg_path: Path, gt_path: Path) -> str | None:
    """
    Show metadata and prompt for Kevin's activity label.
    Returns the chosen activity string, or None if skipped.
    """
    with open(gt_path) as f:
        gt = json.load(f)

    smol   = gt.get("smolvlm_output", {})
    claude = gt.get("claude_annotation", {})
    disc   = gt.get("discrepancies", {})
    ts     = jpg_path.stem  # timestamp in filename

    # Derive display info
    smol_act   = smol.get("activity", "?")
    claude_act = claude.get("activity", "?")
    claude_desc = claude.get("description", "")
    already_reviewed = gt.get("kevin_activity")

    agree = smol_act == claude_act
    agree_str = f"{GREEN}agree{RESET}" if agree else f"{RED}disagree{RESET}"

    print(f"\n{'─'*60}")
    print(f"  {BOLD}{ts}{RESET}  ({jpg_path.name})")
    print(f"  SmolVLM2: {YELLOW}{smol_act}{RESET}  |  Claude: {YELLOW}{claude_act}{RESET}  [{agree_str}]")
    if claude_desc:
        print(f"  {DIM}{claude_desc[:100]}{RESET}")
    if disc.get("missing_objects"):
        print(f"  {DIM}Claude sees but SmolVLM2 missed: {', '.join(disc['missing_objects'][:4])}{RESET}")
    if already_reviewed:
        print(f"  {DIM}Previously labeled: {already_reviewed}{RESET}")

    open_in_preview(jpg_path)

    prompt = (
        f"\n  Activity [{'/'.join(sorted(VALID_ACTIVITIES))}]"
        f"\n  Enter = accept '{claude_act}', s = skip, q = quit: "
    )

    while True:
        try:
            raw = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        if raw == "q":
            raise SystemExit(0)
        if raw == "s":
            return None
        if raw == "":
            return claude_act
        if raw in VALID_ACTIVITIES:
            return raw
        print(f"  {RED}Invalid.{RESET} Must be one of: {', '.join(sorted(VALID_ACTIVITIES))}, s (skip), q (quit)")


def write_kevin_activity(gt_path: Path, activity: str | None) -> None:
    with open(gt_path) as f:
        gt = json.load(f)
    gt["kevin_activity"] = activity
    with open(gt_path, "w") as f:
        json.dump(gt, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Review scene image activity labels (Kevin corrects Claude GT)"
    )
    parser.add_argument("--limit", type=int, default=0,
                        help="Only review the N most recent images (0 = all)")
    parser.add_argument("--force", action="store_true",
                        help="Re-review images that already have a kevin_activity label")
    parser.add_argument("--report", action="store_true",
                        help="Show review coverage summary and exit")
    args = parser.parse_args()

    if not COLLECTION_DIR.exists():
        print(f"Collection directory not found: {COLLECTION_DIR}")
        print("Run the scene service to collect images, then label_scene_data.py to annotate.")
        sys.exit(1)

    if args.report:
        print_report(COLLECTION_DIR)
        return

    # Find JPEG + _gt.json pairs
    candidates = []
    for jpg in sorted(COLLECTION_DIR.glob("*.jpg"), reverse=True):
        gt_path = jpg.with_name(jpg.stem + "_gt.json")
        if not gt_path.exists():
            continue  # not annotated by Claude yet
        if _has_kevin_review(gt_path) and not args.force:
            continue  # already reviewed
        candidates.append((jpg, gt_path))

    if args.limit:
        candidates = candidates[: args.limit]

    if not candidates:
        if args.force:
            print("No annotated images found to re-review.")
        else:
            print("Nothing to review. All annotated images already have Kevin labels.")
            print("Use --force to re-review, or --report to see coverage.")
        return

    print(f"\n{BOLD}Winston Scene Label Review — {len(candidates)} images{RESET}")
    print(f"  Press Enter to accept Claude's label, or type the correct activity.")
    print(f"  Images will open in Preview. Type 'q' at any time to quit.\n")

    saved = skipped = 0

    for jpg_path, gt_path in candidates:
        try:
            chosen = review_image(jpg_path, gt_path)
        except SystemExit:
            break

        if chosen is not None:
            write_kevin_activity(gt_path, chosen)
            saved += 1
            print(f"  {GREEN}✓{RESET}  Saved: {chosen}")
        else:
            skipped += 1
            print(f"  —  Skipped")

    print(f"\n{BOLD}Done.{RESET}  saved={saved}  skipped={skipped}")
    print(f"\nRun report:  python scripts/review_scene_labels.py --report")
    print(f"Run eval:    python scripts/label_scene_data.py --report")


if __name__ == "__main__":
    main()
