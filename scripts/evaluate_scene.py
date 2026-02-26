"""
Evaluate SmolVLM2 scene understanding on the benchmark set.

Produces an accuracy report with three metrics:
  - Activity accuracy: % correct activity classification vs. Claude ground truth
  - Object recall:     % of ground truth objects correctly identified
  - Object precision:  % of SmolVLM2 objects actually present (hallucination rate)
  - Semantic similarity: cosine similarity between SmolVLM2 and Claude descriptions

Run BEFORE prompt/schema changes to establish baseline, and AFTER to measure delta.
Results are saved to data/scene_benchmark/evals/ with timestamps for comparison.

Usage:
    # Baseline (stock SmolVLM2-500M)
    python scripts/evaluate_scene.py

    # After prompt or schema changes
    python scripts/evaluate_scene.py --label after-prompt-v2

    # Compare two eval files
    python scripts/evaluate_scene.py --compare data/scene_benchmark/evals/before.json \\
                                                data/scene_benchmark/evals/after.json
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

BENCHMARK_DIR = Path("data/scene_benchmark")
EVALS_DIR     = Path("data/scene_benchmark/evals")

BOLD   = "\033[1m"
RESET  = "\033[0m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_benchmark() -> list[dict]:
    """Load benchmark items that have both a JPEG and a ground truth JSON."""
    items = []
    for gt_path in sorted(BENCHMARK_DIR.glob("*.json")):
        if gt_path.stem.endswith("_gt"):
            continue  # skip live-data annotations
        jpg_path = gt_path.with_suffix(".jpg")
        if not jpg_path.exists():
            print(f"  WARNING: {gt_path.name} has no matching JPEG — skipping")
            continue
        with open(gt_path) as f:
            gt = json.load(f)
        if "claude_activity" not in gt:
            print(f"  WARNING: {gt_path.name} has no Claude annotation yet — skipping")
            continue
        items.append({"jpg": jpg_path, "gt": gt})
    return items


def compute_object_metrics(
    smolvlm_objects: list[str],
    gt_objects: list[str],
) -> tuple[float, float]:
    """Returns (recall, precision)."""
    smol_set = set(smolvlm_objects)
    gt_set   = set(gt_objects)
    overlap  = smol_set & gt_set
    recall    = len(overlap) / max(len(gt_set),  1)
    precision = len(overlap) / max(len(smol_set), 1)
    return recall, precision


def embed_texts(texts: list[str]) -> np.ndarray:
    """Encode texts with sentence-transformers. Returns (N, D) float32 array."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two normalized vectors (already L2-normalized)."""
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation() -> list[dict]:
    import yaml

    with open("config/default.yaml") as f:
        cfg = yaml.safe_load(f)

    from src.perception.scene import SmolVLM2Scene

    vlm = SmolVLM2Scene(model_id=cfg["scene"]["model"])
    print("  Loading SmolVLM2...")
    t_load = time.monotonic()
    vlm.load()
    print(f"  Loaded in {time.monotonic() - t_load:.1f}s")

    items = load_benchmark()
    if not items:
        print(f"\nNo benchmark images found in {BENCHMARK_DIR}.")
        print("Run 'python scripts/create_scene_benchmark.py' first.")
        sys.exit(1)

    print(f"  {len(items)} benchmark images")
    print()

    # Optionally embed with sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        _st_available = True
    except ImportError:
        _st_available = False
        print("  (install sentence-transformers for semantic similarity metric)")

    results = []
    smolvlm_descs = []
    claude_descs  = []

    for item in items:
        jpg_path = item["jpg"]
        gt       = item["gt"]
        sid      = gt["id"]
        diff     = gt["difficulty"]

        from PIL import Image
        frame = Image.open(str(jpg_path)).convert("RGB")

        t1 = time.monotonic()
        result = vlm.describe(frame)
        t_infer = time.monotonic() - t1

        gt_activity = gt["claude_activity"]
        gt_objects  = gt["claude_objects"]

        activity_ok  = result.activity == gt_activity
        recall, precision = compute_object_metrics(result.objects, gt_objects)

        diff_color = {"easy": GREEN, "medium": YELLOW, "hard": RED}.get(diff, "")
        ok_str = f"{GREEN}OK{RESET}" if activity_ok else f"{RED}XX{RESET}"
        print(f"  [{diff_color}{diff[:3].upper()}{RESET}] {ok_str}  "
              f"activity={result.activity:<10}  gt={gt_activity:<10}  "
              f"recall={recall:.2f}  prec={precision:.2f}  "
              f"({t_infer:.1f}s)  {sid}")
        if not activity_ok:
            print(f"         SmolVLM2: {result.description[:80]}")

        results.append({
            "id":              sid,
            "difficulty":      diff,
            "activity_pred":   result.activity,
            "activity_gt":     gt_activity,
            "activity_ok":     activity_ok,
            "objects_pred":    result.objects,
            "objects_gt":      gt_objects,
            "object_recall":   round(recall, 4),
            "object_precision": round(precision, 4),
            "description_pred": result.description,
            "description_gt":  gt.get("claude_description", ""),
            "confidence":      result.confidence,
            "inference_s":     round(t_infer, 2),
        })
        smolvlm_descs.append(result.description)
        claude_descs.append(gt.get("claude_description", ""))

    # Semantic similarity (optional)
    if _st_available and smolvlm_descs:
        try:
            all_descs = smolvlm_descs + claude_descs
            embeddings = embed_texts(all_descs)
            n = len(smolvlm_descs)
            similarities = [
                cosine_similarity(embeddings[i], embeddings[i + n])
                for i in range(n)
            ]
            for i, sim in enumerate(similarities):
                results[i]["semantic_similarity"] = round(sim, 4)
        except Exception as e:
            print(f"\n  Semantic similarity skipped: {e}")

    vlm.unload()
    return results


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_report(results: list[dict]) -> None:
    print(f"\n{BOLD}━━━ Activity Accuracy by Difficulty ━━━{RESET}")
    for tier in ["easy", "medium", "hard"]:
        tier_r = [r for r in results if r["difficulty"] == tier]
        if not tier_r:
            continue
        n_ok  = sum(1 for r in tier_r if r["activity_ok"])
        color = {"easy": GREEN, "medium": YELLOW, "hard": RED}.get(tier, "")
        print(f"  {color}{tier.upper():6}{RESET}  {n_ok}/{len(tier_r)}")

    n_ok_total = sum(1 for r in results if r["activity_ok"])
    print(f"  {BOLD}OVERALL{RESET}  {n_ok_total}/{len(results)}  "
          f"({100 * n_ok_total / len(results):.0f}%)")

    print(f"\n{BOLD}━━━ Object Detection ━━━{RESET}")
    avg_recall    = float(np.mean([r["object_recall"] for r in results]))
    avg_precision = float(np.mean([r["object_precision"] for r in results]))
    print(f"  avg recall={avg_recall:.3f}  avg precision={avg_precision:.3f}  "
          f"(hallucination_rate={1 - avg_precision:.3f})")

    if "semantic_similarity" in results[0]:
        avg_sim = float(np.mean([r["semantic_similarity"] for r in results]))
        sim_label = (f"{GREEN}good{RESET}" if avg_sim > 0.7
                     else f"{YELLOW}weak{RESET}" if avg_sim > 0.5
                     else f"{RED}poor{RESET}")
        print(f"\n{BOLD}━━━ Semantic Similarity ━━━{RESET}")
        print(f"  avg cosine={avg_sim:.3f} — {sim_label}")
        print("  (>0.7 = descriptions are semantically similar to Claude's)")

    # Common failure modes
    missed: dict[str, int] = {}
    hallucinated: dict[str, int] = {}
    for r in results:
        for obj in set(r["objects_gt"]) - set(r["objects_pred"]):
            missed[obj] = missed.get(obj, 0) + 1
        for obj in set(r["objects_pred"]) - set(r["objects_gt"]):
            hallucinated[obj] = hallucinated.get(obj, 0) + 1

    if missed:
        top_missed = sorted(missed.items(), key=lambda x: -x[1])[:5]
        print(f"\n{BOLD}Most missed objects:{RESET}")
        for obj, cnt in top_missed:
            print(f"  {obj}: missed in {cnt}/{len(results)} scenes")

    if hallucinated:
        top_hall = sorted(hallucinated.items(), key=lambda x: -x[1])[:5]
        print(f"\n{BOLD}Most hallucinated objects:{RESET}")
        for obj, cnt in top_hall:
            print(f"  {obj}: hallucinated in {cnt}/{len(results)} scenes")


def print_comparison(before_path: str, after_path: str) -> None:
    with open(before_path) as f:
        before = json.load(f)
    with open(after_path) as f:
        after  = json.load(f)

    b_r = before["results"]
    a_r = after["results"]

    print(f"\n{BOLD}━━━ Scene Eval: Before vs After ━━━{RESET}")
    print(f"  Before: {before['timestamp']}  ({before.get('label', 'base')})")
    print(f"  After:  {after['timestamp']}  ({after.get('label', 'eval')})")
    print()
    print(f"  {'Metric':20}  {'Before':>8}  {'After':>8}  {'Delta':>8}  {'Better?':>8}")
    print(f"  {'-'*56}")

    def compare_metric(name: str, b_vals: list, a_vals: list, higher_is_better=True) -> None:
        b_avg = float(np.mean(b_vals)) if b_vals else 0.0
        a_avg = float(np.mean(a_vals)) if a_vals else 0.0
        delta = a_avg - b_avg
        better_threshold = 0.02
        if higher_is_better:
            better = f"{GREEN}YES{RESET}" if delta > better_threshold else (
                     f"{RED}NO{RESET}" if delta < -better_threshold else "—")
        else:
            better = f"{GREEN}YES{RESET}" if delta < -better_threshold else (
                     f"{RED}NO{RESET}" if delta > better_threshold else "—")
        print(f"  {name:20}  {b_avg:>8.3f}  {a_avg:>8.3f}  {delta:>+8.3f}  {better:>8}")

    compare_metric("activity_accuracy",
                   [1.0 if r["activity_ok"] else 0.0 for r in b_r],
                   [1.0 if r["activity_ok"] else 0.0 for r in a_r])
    compare_metric("object_recall",
                   [r["object_recall"] for r in b_r],
                   [r["object_recall"] for r in a_r])
    compare_metric("object_precision",
                   [r["object_precision"] for r in b_r],
                   [r["object_precision"] for r in a_r])

    if "semantic_similarity" in b_r[0] and "semantic_similarity" in a_r[0]:
        compare_metric("semantic_similarity",
                       [r["semantic_similarity"] for r in b_r],
                       [r["semantic_similarity"] for r in a_r])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SmolVLM2 on scene benchmark")
    parser.add_argument("--label",   default="",
                        help="Label for this eval run (e.g. 'baseline', 'after-prompt-v2')")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                        help="Compare two saved eval JSON files")
    args = parser.parse_args()

    if args.compare:
        print_comparison(args.compare[0], args.compare[1])
        return

    print(f"\n{BOLD}Winston Scene Benchmark Evaluation{RESET}")
    if args.label:
        print(f"  Label: {args.label}")
    print()

    results = run_evaluation()
    print_report(results)

    # Save
    EVALS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    label_slug = args.label.replace(" ", "-") if args.label else "eval"
    out_path = EVALS_DIR / f"{ts}_{label_slug}.json"

    n_ok = sum(1 for r in results if r["activity_ok"])
    output = {
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "label":              args.label or "eval",
        "n":                  len(results),
        "activity_accuracy":  round(n_ok / len(results), 4),
        "avg_object_recall":  round(float(np.mean([r["object_recall"]    for r in results])), 4),
        "avg_object_precision": round(float(np.mean([r["object_precision"] for r in results])), 4),
        "results":            results,
    }
    if "semantic_similarity" in results[0]:
        output["avg_semantic_similarity"] = round(
            float(np.mean([r["semantic_similarity"] for r in results])), 4
        )

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {out_path}")
    print()
    print("To compare after prompt/schema changes:")
    print(f"  python scripts/evaluate_scene.py --label v2 --compare {out_path} <new_eval.json>")


if __name__ == "__main__":
    main()
