"""
Phase A VLM evaluation — computes accuracy and latency from verified detections.

Reads verified detection files (created by label_scene_data.py) and produces
a per-event, per-scenario accuracy + latency report. This is the Phase A exit
gate: run this to confirm Moondream2 meets acceptance criteria before locking
the VLM model and proceeding to Phase B.

Workflow:
    1. Run detect_event.py to capture events (saves data/detection/*.jpg + *_detection.json)
    2. Run label_scene_data.py to manually verify (saves *_verified.json)
    3. Run this script to compute accuracy + latency

    python scripts/evaluate_vlm.py
    python scripts/evaluate_vlm.py --label phase_a_baseline
    python scripts/evaluate_vlm.py --compare data/scene_benchmark/evals/a.json data/scene_benchmark/evals/b.json
    python scripts/evaluate_vlm.py --rerun --adapter moondream2   # re-run VLM for fresh latency

Outputs:
    Console: per-scenario, per-event accuracy + avg latency_ms
    File:    data/scene_benchmark/evals/{timestamp}_{label}.json
    Updates: evaluations[] list in each scenario YAML
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

DETECTION_DIR = Path("data/detection")
EVAL_DIR      = Path("data/scene_benchmark/evals")

BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"


# ---------------------------------------------------------------------------
# Load verified files
# ---------------------------------------------------------------------------

def load_verified_records(detection_dir: Path) -> list[dict]:
    """
    Load all _verified.json files from the detection directory.
    Only returns records where verified_correct is explicitly True or False
    (skips 'skip' entries).
    """
    records = []
    for path in sorted(detection_dir.glob("*_verified.json")):
        with open(path) as f:
            rec = json.load(f)
        if rec.get("verified_correct") is None:
            continue  # skipped during labeling
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Compute metrics
# ---------------------------------------------------------------------------

def compute_metrics(records: list[dict]) -> dict:
    """
    Compute per-scenario, per-event accuracy and latency from verified records.
    Returns a nested dict: metrics[scenario][event_id] = {correct, total, accuracy, avg_latency_ms}
    """
    # {scenario_path: {event_id: {correct, total, latencies}}}
    by_scenario: dict = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0, "latencies": []}))

    for rec in records:
        scenario  = rec.get("scenario", "unknown")
        event_id  = rec.get("event_id", "unknown")
        correct   = bool(rec.get("verified_correct"))
        latency   = rec.get("latency_ms")

        by_scenario[scenario][event_id]["total"] += 1
        if correct:
            by_scenario[scenario][event_id]["correct"] += 1
        if latency is not None:
            by_scenario[scenario][event_id]["latencies"].append(latency)

    metrics = {}
    for scenario, events in by_scenario.items():
        metrics[scenario] = {}
        for event_id, counts in events.items():
            total   = counts["total"]
            correct = counts["correct"]
            lats    = counts["latencies"]
            metrics[scenario][event_id] = {
                "correct":       correct,
                "total":         total,
                "accuracy":      round(correct / total, 3) if total else 0.0,
                "avg_latency_ms": round(sum(lats) / len(lats), 1) if lats else None,
                "min_latency_ms": round(min(lats), 1) if lats else None,
                "max_latency_ms": round(max(lats), 1) if lats else None,
            }

    return metrics


# ---------------------------------------------------------------------------
# Print report
# ---------------------------------------------------------------------------

def print_report(metrics: dict, records: list[dict]) -> None:
    total_correct = sum(
        ev["correct"] for sc in metrics.values() for ev in sc.values()
    )
    total_samples = sum(
        ev["total"] for sc in metrics.values() for ev in sc.values()
    )
    overall_accuracy = total_correct / total_samples if total_samples else 0.0

    print(f"\n{BOLD}━━━ Phase A VLM Evaluation Report ({total_samples} verified samples) ━━━{RESET}")
    print(f"\n  Overall accuracy: {total_correct}/{total_samples} ({overall_accuracy:.0%})\n")

    for scenario_path, events in metrics.items():
        sc_name = Path(scenario_path).stem
        print(f"  {BOLD}Scenario: {sc_name}{RESET}")
        for event_id, m in events.items():
            acc_str = f"{m['accuracy']:.0%}"
            color   = GREEN if m["accuracy"] >= 0.8 else (YELLOW if m["accuracy"] >= 0.5 else RED)
            lat_str = f"{m['avg_latency_ms']:.0f}ms" if m["avg_latency_ms"] else "n/a"
            fps_str = f"~{1000/m['avg_latency_ms']:.1f} fps" if m["avg_latency_ms"] else ""
            print(
                f"    {event_id:<30}  "
                f"accuracy={color}{acc_str:>5}{RESET}  ({m['correct']}/{m['total']})  "
                f"latency={lat_str} {fps_str}"
            )

    # Phase A exit gate assessment
    all_latencies = [
        ev["avg_latency_ms"]
        for sc in metrics.values()
        for ev in sc.values()
        if ev.get("avg_latency_ms") is not None
    ]
    avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else None

    print(f"\n  {BOLD}Phase A exit gate:{RESET}")
    if avg_latency:
        fps = 1000 / avg_latency
        lat_ok = avg_latency < 1000
        print(
            f"    Latency (≥1 fps target):  "
            f"{'✓' if lat_ok else '✗'}  avg={avg_latency:.0f}ms (~{fps:.1f} fps)"
        )
    print(
        f"    Accuracy (target: evaluate per event and set threshold from data):  "
        f"overall={overall_accuracy:.0%}  ({total_samples} samples)"
    )
    print()


# ---------------------------------------------------------------------------
# Compare two eval runs
# ---------------------------------------------------------------------------

def compare_evals(before_path: Path, after_path: Path) -> None:
    with open(before_path) as f:
        before = json.load(f)
    with open(after_path) as f:
        after = json.load(f)

    b_metrics = before.get("metrics", {})
    a_metrics = after.get("metrics", {})

    print(f"\n{BOLD}━━━ Eval Delta: {before_path.name} → {after_path.name} ━━━{RESET}\n")

    all_scenarios = set(b_metrics) | set(a_metrics)
    for scenario in sorted(all_scenarios):
        print(f"  {BOLD}{Path(scenario).stem}{RESET}")
        b_events = b_metrics.get(scenario, {})
        a_events = a_metrics.get(scenario, {})
        all_events = set(b_events) | set(a_events)
        for event_id in sorted(all_events):
            b = b_events.get(event_id, {})
            a = a_events.get(event_id, {})
            b_acc = b.get("accuracy")
            a_acc = a.get("accuracy")
            b_lat = b.get("avg_latency_ms")
            a_lat = a.get("avg_latency_ms")
            acc_delta = f"{(a_acc - b_acc):+.0%}" if b_acc is not None and a_acc is not None else "n/a"
            lat_delta = f"{(a_lat - b_lat):+.0f}ms" if b_lat is not None and a_lat is not None else "n/a"
            color = GREEN if a_acc and b_acc and a_acc > b_acc else (RED if a_acc and b_acc and a_acc < b_acc else RESET)
            print(
                f"    {event_id:<30}  "
                f"accuracy {b_acc:.0%}→{a_acc:.0%} ({color}{acc_delta}{RESET})  "
                f"latency {lat_delta}"
            )


# ---------------------------------------------------------------------------
# Update YAML evaluations[]
# ---------------------------------------------------------------------------

def update_yaml_evaluations(
    metrics: dict,
    adapter_name: str,
    label: str,
    date: str,
) -> None:
    """Append an entry to evaluations[] in each scenario YAML."""
    for scenario_path, events in metrics.items():
        path = Path(scenario_path)
        if not path.exists():
            continue
        with open(path) as f:
            scenario = yaml.safe_load(f)

        if "evaluations" not in scenario or scenario["evaluations"] is None:
            scenario["evaluations"] = []

        for event_id, m in events.items():
            scenario["evaluations"].append({
                "model":          adapter_name,
                "label":          label,
                "date":           date,
                "event_id":       event_id,
                "accuracy":       m["accuracy"],
                "total_samples":  m["total"],
                "avg_latency_ms": m["avg_latency_ms"],
            })

        with open(path, "w") as f:
            yaml.dump(scenario, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        print(f"  Updated evaluations[] in {path}")


# ---------------------------------------------------------------------------
# Save eval JSON
# ---------------------------------------------------------------------------

def save_eval(
    metrics: dict,
    records: list[dict],
    adapter_name: str,
    label: str,
) -> Path:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    ts    = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    fname = f"{ts}_{adapter_name}_{label}.json" if label else f"{ts}_{adapter_name}.json"
    path  = EVAL_DIR / fname

    with open(path, "w") as f:
        json.dump({
            "adapter":    adapter_name,
            "label":      label,
            "timestamp":  ts,
            "total_samples": len(records),
            "metrics":    metrics,
        }, f, indent=2)

    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Phase A VLM detection accuracy and latency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluate_vlm.py
  python scripts/evaluate_vlm.py --label phase_a_baseline
  python scripts/evaluate_vlm.py --compare data/scene_benchmark/evals/a.json \\
      data/scene_benchmark/evals/b.json
        """,
    )
    parser.add_argument(
        "--label", default="",
        help="Name for this eval run (e.g. phase_a_baseline, after_prompt_v2)",
    )
    parser.add_argument(
        "--adapter", default="internvl2_1b",
        choices=["internvl2_1b", "moondream2"],
        help="Adapter name to record in output (informational, default: internvl2_1b)",
    )
    parser.add_argument(
        "--dir", type=Path, default=DETECTION_DIR,
        help="Directory containing detection + verified JSONs (default: data/detection/)",
    )
    parser.add_argument(
        "--compare", nargs=2, metavar=("BEFORE", "AFTER"),
        help="Compare two eval JSON files and print delta",
    )
    parser.add_argument(
        "--no-update-yaml", action="store_true",
        help="Skip updating evaluations[] in scenario YAMLs",
    )
    args = parser.parse_args()

    if args.compare:
        compare_evals(Path(args.compare[0]), Path(args.compare[1]))
        return

    if not args.dir.exists():
        print(f"Detection directory not found: {args.dir}")
        print("Run detect_event.py to capture events first.")
        sys.exit(1)

    records = load_verified_records(args.dir)
    if not records:
        print(f"No verified records found in {args.dir}")
        print("Run label_scene_data.py to manually verify detections first.")
        sys.exit(1)

    metrics = compute_metrics(records)
    print_report(metrics, records)

    eval_path = save_eval(metrics, records, args.adapter, args.label)
    print(f"  Eval saved: {eval_path}")

    if not args.no_update_yaml:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        update_yaml_evaluations(metrics, args.adapter, args.label, date)


if __name__ == "__main__":
    main()
