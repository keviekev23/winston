"""
Annotate live-collected scene images with Claude ground truth.

For each JPEG + SmolVLM2 JSON sidecar in data/collection/images/:
  - Send the JPEG to Claude claude-sonnet-4-6 with a structured labeling prompt
  - Save the annotation as {timestamp}_gt.json alongside the original
  - Flag discrepancies: activity mismatches, missing objects, hallucinated objects
  - Print a summary of disagreement patterns (useful for identifying failure modes)

Run with:
    python scripts/label_scene_data.py              # annotate all un-annotated images
    python scripts/label_scene_data.py --limit 20   # only the 20 most recent
    python scripts/label_scene_data.py --dry-run    # show what would be sent, no API calls
    python scripts/label_scene_data.py --report     # summarize existing annotations only

This is analogous to Whisper Large-v3 pseudo-labeling in the audio flywheel:
it uses a larger, more capable model to generate the ground truth that the
smaller edge model is trained/evaluated against.
"""

import argparse
import base64
import json
import sys
from collections import Counter
from pathlib import Path

COLLECTION_DIR = Path("data/collection/images")

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
# Annotation
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


# ---------------------------------------------------------------------------
# Report on existing annotations
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate collected scene images with Claude ground truth"
    )
    parser.add_argument("--limit", type=int, default=0,
                        help="Only annotate the N most recent images (0 = all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be annotated without API calls")
    parser.add_argument("--report", action="store_true",
                        help="Print summary of existing annotations and exit")
    parser.add_argument("--force", action="store_true",
                        help="Re-annotate images that already have _gt.json")
    args = parser.parse_args()

    if not COLLECTION_DIR.exists():
        print(f"Collection directory not found: {COLLECTION_DIR}")
        print("Run the scene service to collect some images first.")
        sys.exit(1)

    if args.report:
        print_report(COLLECTION_DIR)
        return

    # Find JPEGs that have a SmolVLM2 sidecar
    candidates = []
    for jpg in sorted(COLLECTION_DIR.glob("*.jpg"), reverse=True):
        sidecar = jpg.with_suffix(".json")
        gt_path = jpg.with_name(jpg.stem + "_gt.json")
        if not sidecar.exists():
            continue  # no SmolVLM2 output — skip
        if gt_path.exists() and not args.force:
            continue  # already annotated
        candidates.append((jpg, sidecar, gt_path))

    if args.limit:
        candidates = candidates[: args.limit]

    if not candidates:
        print("Nothing to annotate. Use --force to re-annotate or --report to view existing.")
        return

    print(f"\n{BOLD}Winston Scene Labeling — {len(candidates)} images{RESET}")
    if args.dry_run:
        print("  [DRY RUN — no API calls]\n")

    annotated = errors = 0

    for jpg_path, sidecar_path, gt_path in candidates:
        with open(sidecar_path) as f:
            smolvlm_output = json.load(f)

        print(f"  {jpg_path.name}", end="", flush=True)

        if args.dry_run:
            print(" [would annotate]")
            continue

        try:
            annotation  = annotate_image(jpg_path)
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

    if not args.dry_run:
        print(f"\n{BOLD}Done.{RESET}  annotated={annotated}  errors={errors}")
        if annotated:
            print(f"\nRun report:  python scripts/label_scene_data.py --report")


if __name__ == "__main__":
    main()
