"""
Create the VLM scene benchmark set.

For each scenario in config/scene_benchmark_scenarios.yaml:
  1. Show webcam preview so you can stage the scene
  2. Press SPACE to capture, ESC to skip
  3. Send the captured image to Claude for ground-truth annotation
  4. Save JPEG + annotated JSON to data/scene_benchmark/

Run with:
    python scripts/create_scene_benchmark.py

Flags:
    --ids easy_001 hard_002    Only record specific scenario IDs
    --force                    Re-record scenarios that already have JPEGs
    --skip-annotation          Save JPEG only, skip Claude API call (annotate later)
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import yaml

BENCHMARK_DIR = Path("data/scene_benchmark")
CONFIG_PATH   = Path("config/scene_benchmark_scenarios.yaml")

BOLD   = "\033[1m"
RESET  = "\033[0m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"

DIFFICULTY_COLOR = {"easy": GREEN, "medium": YELLOW, "hard": RED}

# Standard kitchen object list — must match scene.py and label_scene_data.py
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
# Claude annotation
# ---------------------------------------------------------------------------

def annotate_with_claude(jpeg_path: Path) -> dict:
    """Send JPEG to Claude claude-sonnet-4-6 and return parsed ground truth."""
    from anthropic import Anthropic

    client = Anthropic()

    with open(jpeg_path, "rb") as f:
        image_b64 = base64.standard_b64encode(f.read()).decode("utf-8")

    print("  Sending to Claude for annotation...", end="", flush=True)
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

    # Strip markdown fences if model adds them despite instructions
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    annotation = json.loads(raw)
    print(f" done  (activity={annotation.get('activity')}, objects={len(annotation.get('objects', []))})")
    return annotation


# ---------------------------------------------------------------------------
# Camera capture
# ---------------------------------------------------------------------------

def capture_frame(scenario: dict) -> bytes | None:
    """
    Open webcam preview. Returns JPEG bytes on SPACE, None on ESC/skip.
    """
    import cv2

    diff_color = DIFFICULTY_COLOR.get(scenario["difficulty"], "")
    print(f"\n{BOLD}[{diff_color}{scenario['difficulty'].upper()}{RESET}{BOLD}]{RESET} "
          f"{scenario['id']}")
    print(f"  Stage: {scenario.get('notes', '(no notes)')}")
    print(f"  Expected activity : {scenario['activity_ground_truth']}")
    print(f"  Expected objects  : {', '.join(scenario['objects_ground_truth'])}")
    print("  Press SPACE to capture, ESC to skip")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ERROR: Cannot open camera")
        return None

    # Warm up
    for _ in range(3):
        cap.read()

    result_bytes = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow(f"Winston Benchmark — {scenario['id']} (SPACE=capture, ESC=skip)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE
            ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ok2:
                result_bytes = buf.tobytes()
            break
        elif key == 27:  # ESC
            print("  Skipped.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return result_bytes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Record VLM scene benchmark")
    parser.add_argument("--ids", nargs="+", metavar="ID",
                        help="Only record these scenario IDs")
    parser.add_argument("--force", action="store_true",
                        help="Re-record scenarios that already have images")
    parser.add_argument("--skip-annotation", action="store_true",
                        help="Skip Claude API call; save JPEG only")
    args = parser.parse_args()

    if not CONFIG_PATH.exists():
        print(f"Config not found: {CONFIG_PATH}")
        sys.exit(1)

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    scenarios = cfg["scenarios"]
    if args.ids:
        scenarios = [s for s in scenarios if s["id"] in args.ids]
        if not scenarios:
            print(f"No scenarios matched: {args.ids}")
            sys.exit(1)

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    (BENCHMARK_DIR / "evals").mkdir(exist_ok=True)

    print(f"\n{BOLD}Winston Scene Benchmark — Recording{RESET}")
    print(f"  {len(scenarios)} scenarios  |  output: {BENCHMARK_DIR}")
    print()

    recorded = skipped = already_done = 0

    for scenario in scenarios:
        sid  = scenario["id"]
        jpg  = BENCHMARK_DIR / f"{sid}.jpg"
        meta = BENCHMARK_DIR / f"{sid}.json"

        if jpg.exists() and not args.force:
            print(f"  {sid}  already recorded (--force to overwrite)")
            already_done += 1
            continue

        jpeg_bytes = capture_frame(scenario)
        if jpeg_bytes is None:
            skipped += 1
            continue

        jpg.write_bytes(jpeg_bytes)
        print(f"  Saved: {jpg}")

        if args.skip_annotation:
            # Save a stub JSON so the scenario is tracked
            stub = {
                "id":                   sid,
                "difficulty":           scenario["difficulty"],
                "activity_ground_truth": scenario["activity_ground_truth"],
                "objects_ground_truth":  scenario["objects_ground_truth"],
                "notes":                scenario.get("notes", ""),
                "annotation_source":    "stub — run without --skip-annotation to annotate",
            }
            with open(meta, "w") as f:
                json.dump(stub, f, indent=2)
            print(f"  Stub saved (no Claude annotation): {meta}")
        else:
            try:
                annotation = annotate_with_claude(jpg)
                gt = {
                    "id":                    sid,
                    "difficulty":            scenario["difficulty"],
                    "activity_ground_truth":  scenario["activity_ground_truth"],
                    "objects_ground_truth":   scenario["objects_ground_truth"],
                    "notes":                 scenario.get("notes", ""),
                    "claude_activity":        annotation.get("activity"),
                    "claude_objects":         annotation.get("objects", []),
                    "claude_description":     annotation.get("description", ""),
                    "annotation_source":     "claude-sonnet-4-6",
                }
                # Flag if Claude's activity differs from the staged expectation
                if gt["claude_activity"] != gt["activity_ground_truth"]:
                    gt["annotation_flag"] = (
                        f"Claude says '{gt['claude_activity']}' "
                        f"but scenario expects '{gt['activity_ground_truth']}' — review image"
                    )
                    print(f"  {YELLOW}FLAG:{RESET} {gt['annotation_flag']}")
                with open(meta, "w") as f:
                    json.dump(gt, f, indent=2)
                print(f"  Ground truth saved: {meta}")
            except Exception as e:
                print(f"  {RED}Claude annotation failed:{RESET} {e}")
                print("  JPEG saved. Re-run without --skip-annotation to annotate.")

        recorded += 1

    print(f"\n{BOLD}Done.{RESET}  recorded={recorded}  skipped={skipped}  already_done={already_done}")
    print(f"\nNext steps:")
    print(f"  python scripts/evaluate_scene.py --label baseline")


if __name__ == "__main__":
    main()
