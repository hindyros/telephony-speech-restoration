"""
eval_runner.py

Runs the three-condition evaluation (clean / distorted / restored) and writes
per-condition correct-action rate (CAR) and word-error rate (WER) to results/.

Usage
-----
    python evals/eval_runner.py --conditions clean distorted restored --output results/
    python evals/eval_runner.py --conditions distorted --output results/

Prerequisites
-------------
    data/clean/     : caller audio named <action_id>_<n>.wav  (B01..B08)
    data/distorted/ : telephony-degraded counterparts
    data/restored/  : output of the Colab restoration pipeline
    sandbox/actions.md and GROUND_TRUTH below define the 8 banking tasks

Agent implementation
--------------------
query_agent() is implemented here using:
  1. Whisper "base" model for speech-to-text transcription.
  2. Keyword-scoring classifier over the 8 known banking tasks.
     Each task has a list of keywords; the task with the most keyword hits wins.
     Ties broken by iteration order; returns "unknown" if no keyword matches.

WER metric
----------
If clean audio is available, it is transcribed first to obtain reference
transcripts.  Every distorted / restored file is then transcribed and its WER
computed against the matched clean reference (same file stem).  WER is output
alongside CAR in the JSON summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ground truth & task keywords
# ---------------------------------------------------------------------------
GROUND_TRUTH: dict[str, str] = {
    "B01": "replace_card",
    "B02": "transfer_money",
    "B03": "check_balance",
    "B04": "order_checks",
    "B05": "pay_bill",
    "B06": "reset_password",
    "B07": "schedule_appointment",
    "B08": "get_branch_hours",
}

# Each task is scored by how many of its keywords appear in the transcript.
# Keywords are matched as substrings of the lowercased transcript.
TASK_KEYWORDS: dict[str, list[str]] = {
    "replace_card":         ["replace", "card", "lost", "stolen", "reissue", "new card"],
    "transfer_money":       ["transfer", "send money", "wire", "move funds", "payment"],
    "check_balance":        ["balance", "how much", "statement", "account balance"],
    "order_checks":         ["checks", "checkbook", "order checks"],
    "pay_bill":             ["pay bill", "utility", "auto pay", "bill payment"],
    "reset_password":       ["password", "reset", "locked out", "access", "login"],
    "schedule_appointment": ["appointment", "schedule", "meet", "visit branch"],
    "get_branch_hours":     ["hours", "open", "close", "location", "branch hours"],
}

DATA_DIRS: dict[str, Path] = {
    "clean":     Path("data/clean"),
    "distorted": Path("data/distorted"),
    "restored":  Path("data/restored"),
}


# ---------------------------------------------------------------------------
# Whisper: lazy-loaded once on first call
# ---------------------------------------------------------------------------

_whisper_model = None  # loaded on demand


def _load_whisper() -> Any:
    global _whisper_model
    if _whisper_model is None:
        import whisper
        print("Loading Whisper 'base' model (one-time) …", flush=True)
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def _transcribe(audio_path: Path) -> str:
    """Return the lowercased Whisper transcript for *audio_path*."""
    model = _load_whisper()
    result = model.transcribe(str(audio_path), language="en")
    return result["text"].lower().strip()


# ---------------------------------------------------------------------------
# Banking-task classifier
# ---------------------------------------------------------------------------

def _classify(transcript: str) -> str:
    """Return the most likely banking action label for *transcript*.

    Scores each task by counting keyword hits; ties broken by dict order.
    Returns ``"unknown"`` if no keyword matches.
    """
    scores: dict[str, int] = {}
    for task, keywords in TASK_KEYWORDS.items():
        scores[task] = sum(1 for kw in keywords if kw in transcript)
    best_task = max(scores, key=scores.get)
    return best_task if scores[best_task] > 0 else "unknown"


# ---------------------------------------------------------------------------
# Agent interface
# ---------------------------------------------------------------------------

def query_agent(audio_path: Path) -> tuple[str, str]:
    """Transcribe audio and classify the banking task.

    Returns
    -------
    (action_label, transcript) where action_label is one of the values in
    GROUND_TRUTH (or "unknown") and transcript is the raw Whisper output.
    """
    transcript = _transcribe(audio_path)
    action = _classify(transcript)
    return action, transcript


# ---------------------------------------------------------------------------
# WER helpers (pure stdlib — no jiwer dependency)
# ---------------------------------------------------------------------------

def _edit_distance(a: list[str], b: list[str]) -> int:
    """Standard dynamic-programming word-level edit distance."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_wer(reference: str, hypothesis: str) -> float:
    """Return word error rate = edit_distance(ref, hyp) / len(ref).

    Returns 0.0 if reference is empty; values > 1.0 are possible when the
    hypothesis is much longer than the reference.
    """
    ref = reference.split()
    hyp = hypothesis.split()
    if not ref:
        return 0.0
    return round(_edit_distance(ref, hyp) / len(ref), 4)


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

def build_reference_transcripts(data_dir: Path) -> dict[str, str]:
    """Transcribe all WAVs in *data_dir* and return {stem: transcript}.

    Used to obtain clean-audio references for WER computation.
    """
    refs: dict[str, str] = {}
    wav_files = sorted(data_dir.glob("*.wav"))
    if not wav_files:
        return refs
    print(f"  Transcribing {len(wav_files)} clean files for WER reference …",
          flush=True)
    for wav in wav_files:
        refs[wav.stem] = _transcribe(wav)
    return refs


def run_condition(
    condition: str,
    data_dir: Path,
    reference_transcripts: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate all WAVs in *data_dir* and return per-file result dicts.

    Parameters
    ----------
    reference_transcripts : {stem: transcript} from the clean condition.
        When provided, WER is computed for each file against its matched
        clean reference.  Pass None to skip WER.
    """
    results: list[dict[str, Any]] = []
    for audio_file in sorted(data_dir.glob("*.wav")):
        action_id    = audio_file.stem.split("_")[0]
        ground_truth = GROUND_TRUTH.get(action_id, "unknown")

        try:
            predicted, transcript = query_agent(audio_file)
        except Exception as e:
            predicted, transcript = "error", ""
            print(f"  [WARN] {audio_file.name}: {e}")

        correct = int(predicted == ground_truth)

        # WER against matched clean reference
        wer: float | None = None
        if reference_transcripts is not None:
            ref = reference_transcripts.get(audio_file.stem)
            if ref:
                wer = compute_wer(ref, transcript)

        results.append({
            "condition":   condition,
            "file":        audio_file.name,
            "action_id":   action_id,
            "ground_truth": ground_truth,
            "predicted":   predicted,
            "correct":     correct,
            "wer":         wer,
            "transcript":  transcript,
        })
    return results


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        by_condition.setdefault(r["condition"], []).append(r)

    summary: dict[str, Any] = {}
    for cond, rows in by_condition.items():
        n       = len(rows)
        correct = sum(r["correct"] for r in rows)
        wer_vals = [r["wer"] for r in rows if r["wer"] is not None]
        summary[cond] = {
            "n":                  n,
            "correct":            correct,
            "correct_action_rate": round(correct / n, 4) if n else 0.0,
            "mean_wer":           round(sum(wer_vals) / len(wer_vals), 4)
                                  if wer_vals else None,
        }
    return summary


def _build_meta(args: argparse.Namespace, conditions: list[str]) -> dict[str, Any]:
    from datetime import datetime
    restored_dir = Path(args.restored_dir)
    label = args.label or restored_dir.name
    return {
        "timestamp":           datetime.now().strftime("%Y%m%d_%H%M%S"),
        "label":               label,
        "restored_dir":        str(restored_dir),
        "conditions_evaluated": conditions,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Three-condition audio evaluation: CAR + WER."
    )
    parser.add_argument(
        "--conditions", nargs="+", default=["clean", "distorted", "restored"],
        choices=["clean", "distorted", "restored"],
        help="Which conditions to run (default: all three).",
    )
    parser.add_argument("--output", default="results/",
                        help="Output directory (default: results/).")
    parser.add_argument(
        "--restored-dir", default=None, metavar="PATH",
        help="Directory containing restored WAVs "
             "(default: data/restored). Use this to point at a specific "
             "pipeline-version output, e.g. data/restored_wavlm_silero/.",
    )
    parser.add_argument(
        "--label", default=None, metavar="NAME",
        help="Human-readable label for the restored condition in outputs "
             "(default: basename of --restored-dir, or 'restored').",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve restored directory and label ───────────────────────────────
    restored_path = Path(args.restored_dir) if args.restored_dir else DATA_DIRS["restored"]
    label = args.label or (restored_path.name if args.restored_dir else "restored")

    # Build a per-run directory map; "restored" key always maps to restored_path
    run_dirs: dict[str, Path] = {**DATA_DIRS, "restored": restored_path}

    # ── Determine evaluation order and WER reference strategy ─────────────
    conditions = args.conditions
    if "clean" in conditions and conditions[0] != "clean":
        conditions = ["clean"] + [c for c in conditions if c != "clean"]

    reference_transcripts: dict[str, str] | None = None
    if "clean" not in conditions:
        clean_dir = DATA_DIRS["clean"]
        if clean_dir.exists():
            reference_transcripts = build_reference_transcripts(clean_dir)
            if reference_transcripts:
                print(f"  {len(reference_transcripts)} clean references ready.\n")
        else:
            print("[INFO] data/clean/ not found — WER will be skipped.\n")

    # ── Run each condition ─────────────────────────────────────────────────
    all_results: list[dict[str, Any]] = []
    conditions_run: list[str] = []
    for condition in conditions:
        data_dir = run_dirs[condition]
        # Use the user-supplied label as the condition name for "restored"
        cond_name = label if condition == "restored" else condition

        if not data_dir.exists():
            print(f"[SKIP] {cond_name}: {data_dir} not found")
            continue

        print(f"Running condition: {cond_name} ({data_dir})")
        results = run_condition(cond_name, data_dir,
                                reference_transcripts=reference_transcripts)
        if condition == "clean" and reference_transcripts is None:
            reference_transcripts = {
                Path(r["file"]).stem: r["transcript"] for r in results
            }
            print(f"  {len(reference_transcripts)} clean references ready for WER.\n")
        all_results.extend(results)
        conditions_run.append(cond_name)
        print(f"  {len(results)} samples evaluated")

    # ── Write outputs ──────────────────────────────────────────────────────
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Raw CSV (one row per file, includes transcript + WER)
    raw_path = output_dir / f"raw_{timestamp}.csv"
    fieldnames = ["condition", "file", "action_id", "ground_truth",
                  "predicted", "correct", "wer", "transcript"]
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    # Summary JSON (CAR + WER per condition + run metadata)
    summary = summarize(all_results)
    summary["meta"] = _build_meta(args, conditions_run)
    summary_path = output_dir / f"summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print
    print("\n=== Results ===")
    for cond, stats in summary.items():
        if cond == "meta":
            continue
        wer_str = (f"  mean_wer={stats['mean_wer']:.3f}"
                   if stats["mean_wer"] is not None else "  wer=n/a")
        print(f"  {cond:20s}  "
              f"CAR={stats['correct_action_rate']:.1%} "
              f"({stats['correct']}/{stats['n']})"
              f"{wer_str}")

    print(f"\nRaw results -> {raw_path}")
    print(f"Summary     -> {summary_path}")


if __name__ == "__main__":
    main()
