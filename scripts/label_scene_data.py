"""
Label scene images — two modes:

MODE 1 (--manual): Manual verification of detect_event.py captures.
  For each JPEG + detection JSON in data/detection/:
    - Open the image in macOS Preview
    - Prompt "Event detected correctly? [y/n/skip]"
    - Save _verified.json for use by evaluate_vlm.py
  Run with:
    python scripts/label_scene_data.py --manual
    python scripts/label_scene_data.py --manual --dir data/detection/
    python scripts/label_scene_data.py --manual --force   # re-verify already-verified

MODE 2 (default): Annotate live-collected SmolVLM2 scene images with Claude ground truth.
  For each JPEG + SmolVLM2 JSON sidecar in data/collection/images/:
    - Send the JPEG to Claude claude-sonnet-4-6 with a structured labeling prompt
    - Save the annotation as {timestamp}_gt.json alongside the original
    - Flag discrepancies: activity mismatches, missing objects, hallucinated objects
  Run with:
    python scripts/label_scene_data.py              # annotate all un-annotated images
    python scripts/label_scene_data.py --limit 20   # only the 20 most recent
    python scripts/label_scene_data.py --dry-run    # show what would be sent, no API calls
    python scripts/label_scene_data.py --report     # summarize existing annotations only

NOTE (2026-02): Mode 2 (cloud annotation) is superseded by Mode 1 (manual verification)
for Phase A VLM evaluation, given smaller sample sizes from targeted detect_event.py runs.
Mode 2 remains available for the original SmolVLM2 collection flywheel workflow.
"""

import argparse
import base64
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

COLLECTION_DIR = Path("data/collection/images")
DETECTION_DIR  = Path("data/detection")

BOLD   = "\033[1m"
RESET  = "\033[0m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"

KITCHEN_OBJECTS = [
    "knife", "cutting board", "pot", "pan", "bowl", "plate", "cup", "mug",
    "spoon", "fork", "spatula", "whisk", "ladle", "tongs", "oven", "stove",
    "microwave", "refrigerator", "counter", "sink", "faucet",
    "vegetables", "fruit", "meat", "chicken", "fish", "onion", "garlic",
    "carrot", "tomato", "pepper", "potato", "herbs", "spices", "flour",
    "water", "oil", "butter", "eggs", "pasta", "rice", "person", "hand",
]

CLAUDE_LABEL_PROMPT = f"""You are annotating kitchen scenes for a cooking assistant AI. \
Respond with a JSON object only — no explanation, no markdown fences.

The JSON must have exactly these keys:
  "activity": one of ["cooking", "eating", "cleaning", "idle", "unknown"]
  "objects":  array of strings, choosing ONLY from this list: {json.dumps(KITCHEN_OBJECTS)}
  "description": 1-2 sentences describing the scene

Criteria:
- "cooking": someone is actively preparing food (chopping, stirring, baking, etc.)
- "eating": someone is eating or drinking
- "cleaning": someone is washing dishes, wiping counters, etc.
- "idle": kitchen is empty or person is not engaged in any of the above
- Only include objects from the provided list that are clearly visible
- Be conservative: if unsure whether an object is present, omit it"""


# ---------------------------------------------------------------------------
# MODE 1: Manual verification (Phase A event detection evaluation)
# ---------------------------------------------------------------------------

def run_manual_verification(detection_dir: Path, force: bool) -> None:
    """
    Walk through detect_event.py captures and manually verify each one.
    Opens each JPEG in Preview, asks y/n, saves _verified.json for evaluate_vlm.py.
    """
    if not detection_dir.exists():
        print(f"Detection directory not found: {detection_dir}")
        print("Run detect_event.py to capture events first.")
        sys.exit(1)

    candidates = []
    for jpg in sorted(detection_dir.glob("*.jpg")):
        detection_json = jpg.with_name(jpg.stem + "_detection.json")
        verified_json  = jpg.with_name(jpg.stem + "_verified.json")
        if not detection_json.exists():
            continue
        if verified_json.exists() and not force:
            continue
        candidates.append((jpg, detection_json, verified_json))

    if not candidates:
        msg = "No detection captures found" if force else "No unverified captures found"
        print(f"{msg} in {detection_dir}")
        if not force:
            print("Use --force to re-verify existing ones.")
        return

    print(f"\n{BOLD}Winston Manual Verification — {len(candidates)} images{RESET}")
    print("  Opens each image in Preview. Enter y/n/skip in terminal.\n")

    verified = skipped = errors = 0

    for jpg_path, detection_path, verified_path in candidates:
        with open(detection_path) as f:
            detection = json.load(f)

        event_id       = detection.get("event_id", "unknown")
        detected_label = detection.get("detected_label", "?")
        latency_ms     = detection.get("latency_ms")
        scenario       = detection.get("scenario", "")

        print(f"  {BOLD}{jpg_path.name}{RESET}")
        lat_str = f"  |  Latency: {latency_ms:.0f}ms" if latency_ms else ""
        print(f"    Event: {event_id}  |  Label: {detected_label}{lat_str}")

        try:
            subprocess.run(["open", "-a", "Preview", str(jpg_path)], check=False)
        except Exception:
            print(f"    [Could not open Preview — check image manually: {jpg_path}]")

        while True:
            try:
                answer = input(f"    Event '{event_id}' detected correctly? [y/n/skip]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nInterrupted — saving progress so far.")
                return

            if answer in ("y", "yes"):
                verified_correct: bool | None = True
                break
            elif answer in ("n", "no"):
                verified_correct = False
                break
            elif answer in ("s", "skip", ""):
                verified_correct = None
                break
            else:
                print("    Please enter y, n, or skip")

        if verified_correct is None:
            print(f"    → Skipped\n")
            skipped += 1
            continue

        record = {
            "image_path":       str(jpg_path),
            "scenario":         scenario,
            "event_id":         event_id,
            "detected_label":   detected_label,
            "latency_ms":       latency_ms,
            "verified_correct": verified_correct,
            "verified_by":      "kevin_manual",
            "timestamp":        datetime.now(timezone.utc).isoformat(),
        }

        try:
            with open(verified_path, "w") as f:
                json.dump(record, f, indent=2)
            status = f"{GREEN}✓ correct{RESET}" if verified_correct else f"{RED}✗ incorrect{RESET}"
            print(f"    → {status}\n")
            verified += 1
        except Exception as e:
            print(f"    {RED}ERROR saving:{RESET} {e}\n")
            errors += 1

    print(f"{BOLD}Done.{RESET}  verified={verified}  skipped={skipped}  errors={errors}")
    if verified:
        print(f"\nRun evaluation:  python scripts/evaluate_vlm.py --label phase_a_baseline")


# ---------------------------------------------------------------------------
# MODE 2: Cloud annotation (legacy SmolVLM2 flywheel workflow)
# ---------------------------------------------------------------------------

def annotate_image(jpeg_path: Path) -> dict:
    """Send a JPEG to Claude and return parsed annotation dict."""
    from anthropic import Anthropic

    client = Anthropic()
    with open(jpeg_path, "rb") as f:
        image_b64 = base64.standard_b64encode(f.read()).decode("utf-8")

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": CLAUDE_LABEL_PROMPT},
                ],
            }
        ],
    )
    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def compute_discrepancies(smolvlm_json: dict, claude_annotation: dict) -> dict:
    """
    Compare SmolVLM2 output to Claude ground truth. Returns flags dict.
    activity_mismatch: true/false
    missing_objects: objects Claude sees that SmolVLM2 missed
    hallucinated_objects: objects SmolVLM2 listed that Claude doesn't see
    """
    smol_activity = smolvlm_json.get("activity", "unknown")
    smol_objects  = set(smolvlm_json.get("objects", []))
    gt_activity   = claude_annotation.get("activity", "unknown")
    gt_objects    = set(claude_annotation.get("objects", []))

    return {
        "activity_mismatch":     smol_activity != gt_activity,
        "smolvlm_activity":      smol_activity,
        "claude_activity":       gt_activity,
        "missing_objects":       sorted(gt_objects - smol_objects),
        "hallucinated_objects":  sorted(smol_objects - gt_objects),
        "object_recall":         len(smol_objects & gt_objects) / max(len(gt_objects), 1),
        "object_precision":      len(smol_objects & gt_objects) / max(len(smol_objects), 1),
    }


def print_report(collection_dir: Path) -> None:
    gt_files = sorted(collection_dir.glob("*_gt.json"))
    if not gt_files:
        print("No annotated files found. Run without --report to create annotations.")
        return

    activity_mismatches = 0
    activity_smol: Counter = Counter()
    activity_gt:   Counter = Counter()
    all_missing:   Counter = Counter()
    all_hallucinated: Counter = Counter()
    recalls:   list[float] = []
    precisions: list[float] = []

    for gt_path in gt_files:
        with open(gt_path) as f:
            gt = json.load(f)
        disc = gt.get("discrepancies", {})
        if not disc:
            continue
        if disc.get("activity_mismatch"):
            activity_mismatches += 1
        activity_smol[disc.get("smolvlm_activity", "?")] += 1
        activity_gt[disc.get("claude_activity", "?")] += 1
        for obj in disc.get("missing_objects", []):
            all_missing[obj] += 1
        for obj in disc.get("hallucinated_objects", []):
            all_hallucinated[obj] += 1
        recalls.append(disc.get("object_recall", 0.0))
        precisions.append(disc.get("object_precision", 0.0))

    n = len(gt_files)
    avg_recall    = sum(recalls) / n if recalls else 0.0
    avg_precision = sum(precisions) / n if precisions else 0.0

    print(f"\n{BOLD}━━━ Live Data Annotation Report ({n} images) ━━━{RESET}")
    print(f"\n  Activity accuracy:  {n - activity_mismatches}/{n} "
          f"({100*(n-activity_mismatches)/n:.0f}%)")
    print(f"  Object recall:     {avg_recall:.3f}   (SmolVLM2 finds this % of Claude's objects)")
    print(f"  Object precision:  {avg_precision:.3f}  (this % of SmolVLM2's objects are real)")

    if activity_mismatches:
        print(f"\n  {YELLOW}Activity mismatches ({activity_mismatches}):{RESET}")
        print(f"    SmolVLM2 distribution: {dict(activity_smol.most_common(5))}")
        print(f"    Claude distribution:   {dict(activity_gt.most_common(5))}")

    if all_missing:
        print(f"\n  {RED}Most-missed objects (Claude sees, SmolVLM2 misses):{RESET}")
        for obj, count in all_missing.most_common(8):
            print(f"    {obj}: {count}")

    if all_hallucinated:
        print(f"\n  {YELLOW}Most-hallucinated objects (SmolVLM2 invents):{RESET}")
        for obj, count in all_hallucinated.most_common(8):
            print(f"    {obj}: {count}")


def run_cloud_annotation(
    collection_dir: Path,
    limit: int,
    dry_run: bool,
    force: bool,
) -> None:
    """MODE 2: Cloud annotation using Claude (legacy SmolVLM2 flywheel workflow)."""
    if not collection_dir.exists():
        print(f"Collection directory not found: {collection_dir}")
        print("Run the scene service to collect some images first.")
        sys.exit(1)

    candidates = []
    for jpg in sorted(collection_dir.glob("*.jpg"), reverse=True):
        sidecar = jpg.with_suffix(".json")
        gt_path = jpg.with_name(jpg.stem + "_gt.json")
        if not sidecar.exists():
            continue
        if gt_path.exists() and not force:
            continue
        candidates.append((jpg, sidecar, gt_path))

    if limit:
        candidates = candidates[:limit]

    if not candidates:
        print("Nothing to annotate. Use --force to re-annotate or --report to view existing.")
        return

    print(f"\n{BOLD}Winston Scene Labeling — {len(candidates)} images{RESET}")
    if dry_run:
        print("  [DRY RUN — no API calls]\n")

    annotated = errors = 0

    for jpg_path, sidecar_path, gt_path in candidates:
        with open(sidecar_path) as f:
            smolvlm_output = json.load(f)

        print(f"  {jpg_path.name}", end="", flush=True)

        if dry_run:
            print(" [would annotate]")
            continue

        try:
            annotation    = annotate_image(jpg_path)
            discrepancies = compute_discrepancies(smolvlm_output, annotation)

            gt_record = {
                "image":              str(jpg_path),
                "smolvlm_output":     smolvlm_output,
                "claude_annotation":  annotation,
                "discrepancies":      discrepancies,
                "annotation_source":  "claude-sonnet-4-6",
            }
            with open(gt_path, "w") as f:
                json.dump(gt_record, f, indent=2)

            flags = []
            if discrepancies["activity_mismatch"]:
                flags.append(f"{RED}activity mismatch{RESET}")
            if discrepancies["missing_objects"]:
                flags.append(f"{YELLOW}missed {len(discrepancies['missing_objects'])} objects{RESET}")
            if discrepancies["hallucinated_objects"]:
                flags.append(f"{YELLOW}hallucinated {len(discrepancies['hallucinated_objects'])}{RESET}")

            status = f"  [{', '.join(flags)}]" if flags else "  [OK]"
            print(f"  activity={annotation['activity']}  "
                  f"recall={discrepancies['object_recall']:.2f}  "
                  f"precision={discrepancies['object_precision']:.2f}"
                  f"{status}")
            annotated += 1

        except Exception as e:
            print(f"  {RED}ERROR:{RESET} {e}")
            errors += 1

    if not dry_run:
        print(f"\n{BOLD}Done.{RESET}  annotated={annotated}  errors={errors}")
        if annotated:
            print(f"\nRun report:  python scripts/label_scene_data.py --report")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label scene images: manual verification (Mode 1) or cloud annotation (Mode 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Mode 1 — manual verification (Phase A VLM evaluation):
  python scripts/label_scene_data.py --manual
  python scripts/label_scene_data.py --manual --dir data/detection/ --force

Mode 2 — cloud annotation (legacy SmolVLM2 flywheel):
  python scripts/label_scene_data.py
  python scripts/label_scene_data.py --limit 20 --dry-run
  python scripts/label_scene_data.py --report
        """,
    )
    parser.add_argument(
        "--manual", action="store_true",
        help="Manual verification mode (Phase A): walk through detect_event.py captures with y/n",
    )
    parser.add_argument(
        "--dir", type=Path, default=DETECTION_DIR,
        help="Directory for manual mode (default: data/detection/)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="[Mode 2] Only annotate the N most recent images (0 = all)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="[Mode 2] Show what would be annotated without API calls",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="[Mode 2] Print summary of existing cloud annotations and exit",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process images that already have output files",
    )
    args = parser.parse_args()

    if args.manual:
        run_manual_verification(args.dir, force=args.force)
    elif args.report:
        print_report(COLLECTION_DIR)
    else:
        run_cloud_annotation(COLLECTION_DIR, args.limit, args.dry_run, args.force)


if __name__ == "__main__":
    main()
