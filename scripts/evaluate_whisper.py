"""
Evaluate Whisper STT on the benchmark set. Produces a WER report and
confidence calibration analysis.

Run this BEFORE and AFTER each fine-tuning cycle to measure flywheel improvement.
Results are saved to data/benchmark/evals/ with a timestamp for comparison.

Usage:
    # Baseline (stock Whisper Small.en)
    python scripts/evaluate_whisper.py

    # After fine-tuning (point to the downloaded adapter)
    python scripts/evaluate_whisper.py --adapter path/to/adapter

    # Compare two eval result files
    python scripts/evaluate_whisper.py --compare data/benchmark/evals/before.json data/benchmark/evals/after.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

BENCHMARK_DIR = Path("data/benchmark")
EVALS_DIR     = Path("data/benchmark/evals")
SAMPLE_RATE   = 16000

BOLD  = "\033[1m"
RESET = "\033[0m"
GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW = "\033[93m"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_benchmark() -> list[dict]:
    """Load all benchmark items that have both a WAV and a metadata JSON."""
    items = []
    for meta_path in sorted(BENCHMARK_DIR.glob("*.json")):
        wav_path = meta_path.with_suffix(".wav")
        if not wav_path.exists():
            print(f"  WARNING: {meta_path.name} has no matching WAV — skipping")
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        items.append({"wav": wav_path, "meta": meta})
    return items


def load_wav(path: Path) -> np.ndarray:
    import wave
    with wave.open(str(path), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    return audio


def normalize(text: str) -> str:
    """
    Normalize transcript for WER comparison.
    Strips punctuation, lowercases, normalizes digit/word number formatting.
    Uses jiwer's WhisperNormalizer which matches what Whisper itself was trained on.
    """
    from jiwer import transforms as tr
    normalizer = tr.Compose([
        tr.ToLowerCase(),
        tr.RemovePunctuation(),
        tr.RemoveMultipleSpaces(),
        tr.Strip(),
    ])
    return normalizer([text])[0]


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word error rate using jiwer with transcript normalization applied to both sides."""
    from jiwer import wer
    return wer(normalize(reference), normalize(hypothesis))


def calibration_table(results: list[dict], n_buckets: int = 5) -> list[dict]:
    """
    Bucket results by confidence, compute accuracy per bucket.
    Accuracy = fraction of utterances where WER < 0.10 (≤1 word error in 10).
    Returns list of bucket dicts for display/saving.
    """
    edges = np.linspace(0.0, 1.0, n_buckets + 1)
    buckets = []
    for i in range(n_buckets):
        lo, hi = edges[i], edges[i + 1]
        bucket_items = [r for r in results if lo <= r["confidence"] < hi]
        if not bucket_items:
            continue
        avg_conf = float(np.mean([r["confidence"] for r in bucket_items]))
        avg_wer  = float(np.mean([r["wer"] for r in bucket_items]))
        accuracy = float(np.mean([1.0 if r["wer"] < 0.10 else 0.0 for r in bucket_items]))
        buckets.append({
            "range":    f"{lo:.1f}–{hi:.1f}",
            "count":    len(bucket_items),
            "avg_conf": avg_conf,
            "avg_wer":  avg_wer,
            "accuracy": accuracy,
        })
    return buckets


def confidence_wer_correlation(results: list[dict]) -> float:
    """Spearman correlation between confidence and (1 - WER). Higher = better calibrated."""
    from scipy.stats import spearmanr
    confs = [r["confidence"] for r in results]
    accs  = [max(0.0, 1.0 - r["wer"]) for r in results]
    corr, _ = spearmanr(confs, accs)
    return float(corr)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation(adapter: str | None = None) -> dict:
    from src.perception.stt import WhisperSTT

    import yaml
    with open("config/default.yaml") as f:
        cfg = yaml.safe_load(f)

    model_path = cfg["stt"]["model"]
    if adapter:
        print(f"  Using adapter: {adapter}")
        # TODO: load adapter — requires merging LoRA weights into base model
        # For now, print a reminder; implement when first adapter is available
        print("  NOTE: adapter loading not yet implemented — evaluating base model")

    print(f"  Loading model: {model_path}")
    stt = WhisperSTT(model=model_path, language=cfg["stt"]["language"])

    items = load_benchmark()
    if not items:
        print(f"\nNo benchmark recordings found in {BENCHMARK_DIR}.")
        print("Run 'python scripts/create_benchmark_set.py' first.")
        sys.exit(1)

    print(f"  {len(items)} utterances found")
    print()

    results = []
    for item in items:
        wav   = load_wav(item["wav"])
        meta  = item["meta"]
        ref   = meta["text"]
        utt_id = meta["id"]
        diff  = meta["difficulty"]

        result = stt.transcribe(wav)
        wer_val = compute_wer(ref, result.text)

        ok = "OK " if wer_val < 0.10 else "ERR"
        diff_color = {"easy": GREEN, "medium": YELLOW, "hard": RED}.get(diff, "")
        print(f"  [{diff_color}{diff[:3].upper()}{RESET}] {ok}  conf={result.confidence:.2f}  "
              f"WER={wer_val:.2f}  {utt_id}")
        if wer_val >= 0.10:
            print(f"         REF: {ref}")
            print(f"         HYP: {result.text}")

        results.append({
            "id":           utt_id,
            "difficulty":   diff,
            "reference":    ref,
            "hypothesis":   result.text,
            "confidence":   round(result.confidence, 4),
            "wer":          round(wer_val, 4),
            "no_speech_prob": round(result.no_speech_prob, 4),
            "avg_logprob":  round(result.avg_logprob, 4),
        })

    return results


def print_report(results: list[dict]) -> None:
    print(f"\n{BOLD}━━━ WER by Difficulty ━━━{RESET}")
    for tier in ["easy", "medium", "hard"]:
        tier_results = [r for r in results if r["difficulty"] == tier]
        if not tier_results:
            continue
        avg_wer  = float(np.mean([r["wer"] for r in tier_results]))
        avg_conf = float(np.mean([r["confidence"] for r in tier_results]))
        n_ok     = sum(1 for r in tier_results if r["wer"] < 0.10)
        color = {
            "easy": GREEN, "medium": YELLOW, "hard": RED
        }.get(tier, "")
        print(f"  {color}{tier.upper():6}{RESET}  avg WER={avg_wer:.3f}  avg conf={avg_conf:.3f}  "
              f"accuracy={n_ok}/{len(tier_results)}")

    overall_wer  = float(np.mean([r["wer"] for r in results]))
    overall_conf = float(np.mean([r["confidence"] for r in results]))
    n_ok_total   = sum(1 for r in results if r["wer"] < 0.10)
    print(f"  {BOLD}OVERALL{RESET}  avg WER={overall_wer:.3f}  avg conf={overall_conf:.3f}  "
          f"accuracy={n_ok_total}/{len(results)}")

    print(f"\n{BOLD}━━━ Confidence Calibration ━━━{RESET}")
    buckets = calibration_table(results)
    print(f"  {'Conf range':12}  {'N':>4}  {'avg conf':>8}  {'avg WER':>8}  {'accuracy':>8}")
    print(f"  {'-'*50}")
    for b in buckets:
        print(f"  {b['range']:12}  {b['count']:>4}  {b['avg_conf']:>8.3f}  {b['avg_wer']:>8.3f}  {b['accuracy']:>8.3f}")

    try:
        corr = confidence_wer_correlation(results)
        corr_label = (f"{GREEN}good{RESET}" if corr > 0.5
                      else f"{YELLOW}weak{RESET}" if corr > 0.2
                      else f"{RED}poor{RESET}")
        print(f"\n  Confidence↔accuracy correlation (Spearman): {corr:.3f} — {corr_label}")
        print("  (Target: >0.5 means confidence scores are meaningful signals)")
    except ImportError:
        print("  (install scipy for correlation metric: pip install scipy)")

    print()
    print(f"{BOLD}Threshold guidance:{RESET}")
    print("  Review the calibration table above. The 'right' confidence_threshold in")
    print("  config/default.yaml is the value where:")
    print("    - accuracy above threshold is consistently high (>90%)")
    print("    - most utterances below threshold have meaningful WER")
    print("  If calibration correlation is poor (<0.3), the confidence formula in")
    print("  src/perception/stt.py needs rethinking before the flywheel is meaningful.")


def print_comparison(before_path: str, after_path: str) -> None:
    with open(before_path) as f:
        before_data = json.load(f)
    with open(after_path) as f:
        after_data = json.load(f)

    before_results = before_data["results"]
    after_results  = after_data["results"]

    # Index by id for comparison
    before_by_id = {r["id"]: r for r in before_results}
    after_by_id  = {r["id"]: r for r in after_results}

    print(f"\n{BOLD}━━━ Flywheel Improvement: Before vs After ━━━{RESET}")
    print(f"  Before: {before_data['timestamp']}  ({before_data.get('label', 'base')})")
    print(f"  After:  {after_data['timestamp']}  ({after_data.get('label', 'fine-tuned')})")
    print()
    print(f"  {'Difficulty':10}  {'Before WER':>10}  {'After WER':>10}  {'Delta':>10}  {'Better?':>8}")
    print(f"  {'-'*55}")

    for tier in ["easy", "medium", "hard", "overall"]:
        if tier == "overall":
            b_tier = before_results
            a_tier = after_results
        else:
            b_tier = [r for r in before_results if r["difficulty"] == tier]
            a_tier = [r for r in after_results  if r["difficulty"] == tier]

        if not b_tier or not a_tier:
            continue

        b_wer = float(np.mean([r["wer"] for r in b_tier]))
        a_wer = float(np.mean([r["wer"] for r in a_tier]))
        delta = a_wer - b_wer
        better = f"{GREEN}YES{RESET}" if delta < -0.01 else (f"{RED}NO{RESET}" if delta > 0.01 else "—")
        print(f"  {tier:10}  {b_wer:>10.3f}  {a_wer:>10.3f}  {delta:>+10.3f}  {better:>8}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Whisper STT on benchmark set")
    parser.add_argument("--adapter", help="Path to fine-tuned LoRA adapter (optional)")
    parser.add_argument("--label",   default="",
                        help="Label for this eval (e.g. 'base', 'cycle-1')")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                        help="Compare two saved eval JSON files")
    args = parser.parse_args()

    if args.compare:
        print_comparison(args.compare[0], args.compare[1])
        return

    print(f"\n{BOLD}Winston Benchmark Evaluation{RESET}")
    if args.label:
        print(f"  Label: {args.label}")
    print()

    results = run_evaluation(adapter=args.adapter)
    print_report(results)

    # Save results
    EVALS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    label_slug = args.label.replace(" ", "-") if args.label else "eval"
    out_path = EVALS_DIR / f"{ts}_{label_slug}.json"

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label":     args.label or "eval",
        "adapter":   args.adapter,
        "n":         len(results),
        "overall_wer": round(float(np.mean([r["wer"] for r in results])), 4),
        "results":   results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {out_path}")
    print()
    print("To compare after fine-tuning:")
    print(f"  python scripts/evaluate_whisper.py --label cycle-1 --compare {out_path} <new_eval.json>")


if __name__ == "__main__":
    main()
