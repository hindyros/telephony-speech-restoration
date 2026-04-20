"""
eval_runner.py

Runs the three-condition evaluation (clean / distorted / restored) against
the sandbox action list and writes per-condition correct-action rates to results/.

Usage:
    python evals/eval_runner.py --conditions clean distorted restored --output results/

Prerequisites:
    - data/clean/    : original audio samples named <action_id>_<n>.wav
    - data/distorted/: noise-augmented counterparts
    - data/restored/ : output of the Colab restoration pipeline
    - sandbox/actions.md parsed into GROUND_TRUTH below (or a JSON sidecar)

TODO: plug in your actual agent call in `query_agent()`.
"""

import argparse
import csv
import json
import os
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Ground truth: maps action_id -> expected agent action label
# ---------------------------------------------------------------------------
GROUND_TRUTH = {
    "A01": "initiate_return",
    "A02": "check_order_status",
    "A03": "transfer_to_human",
    "A04": "cancel_subscription",
    "A05": "update_address",
    "A06": "report_missing_package",
    "A07": "apply_promo_code",
    "A08": "track_delivery",
    "A09": "initiate_refund",
    "A10": "update_password",
}

DATA_DIRS = {
    "clean":     Path("data/clean"),
    "distorted": Path("data/distorted"),
    "restored":  Path("data/restored"),
}


# ---------------------------------------------------------------------------
# Agent interface — replace with your actual implementation
# ---------------------------------------------------------------------------
def query_agent(audio_path: Path) -> str:
    """
    Send audio (or restored text) to the agent and return the action label it takes.

    TODO: implement this. Options:
      - Transcribe audio -> feed text to LLM agent
      - Feed audio directly if the agent supports it
      - If using restored text output from Colab, load the .txt sidecar instead

    Returns one of the action labels in GROUND_TRUTH.values(), or "unknown".
    """
    raise NotImplementedError("Implement query_agent() with your agent call.")


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------
def run_condition(condition: str, data_dir: Path) -> list[dict]:
    results = []
    for audio_file in sorted(data_dir.glob("*.wav")):
        # Expect filename format: <action_id>_<n>.wav  e.g. A01_01.wav
        action_id = audio_file.stem.split("_")[0]
        ground_truth = GROUND_TRUTH.get(action_id, "unknown")

        try:
            predicted = query_agent(audio_file)
        except NotImplementedError:
            raise
        except Exception as e:
            predicted = "error"
            print(f"  [WARN] {audio_file.name}: {e}")

        correct = int(predicted == ground_truth)
        results.append({
            "condition": condition,
            "file": audio_file.name,
            "action_id": action_id,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "correct": correct,
        })
    return results


def summarize(results: list[dict]) -> dict:
    by_condition: dict[str, list] = {}
    for r in results:
        by_condition.setdefault(r["condition"], []).append(r["correct"])
    return {
        cond: {
            "n": len(scores),
            "correct": sum(scores),
            "correct_action_rate": round(sum(scores) / len(scores), 4) if scores else 0.0,
        }
        for cond, scores in by_condition.items()
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conditions", nargs="+", default=["clean", "distorted", "restored"],
        choices=["clean", "distorted", "restored"],
        help="Which conditions to run",
    )
    parser.add_argument("--output", default="results/", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for condition in args.conditions:
        data_dir = DATA_DIRS[condition]
        if not data_dir.exists():
            print(f"[SKIP] {condition}: {data_dir} not found")
            continue
        print(f"Running condition: {condition} ({data_dir})")
        results = run_condition(condition, data_dir)
        all_results.extend(results)
        print(f"  {len(results)} samples evaluated")

    # Write raw results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = output_dir / f"raw_{timestamp}.csv"
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "file", "action_id", "ground_truth", "predicted", "correct"])
        writer.writeheader()
        writer.writerows(all_results)

    # Write summary
    summary = summarize(all_results)
    summary_path = output_dir / f"summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Results ===")
    for cond, stats in summary.items():
        print(f"  {cond:12s}  correct_action_rate={stats['correct_action_rate']:.1%}  ({stats['correct']}/{stats['n']})")

    print(f"\nRaw results -> {raw_path}")
    print(f"Summary     -> {summary_path}")


if __name__ == "__main__":
    main()
