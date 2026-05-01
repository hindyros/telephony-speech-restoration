"""compare_runs.py

Reads one or more eval_runner.py summary JSONs and prints a unified benchmark
table so you can compare pipeline versions side-by-side.

Usage
-----
    # Compare every summary JSON in results/:
    python evals/compare_runs.py results/summary_*.json

    # Compare specific files and optionally attach audio-quality reports:
    python evals/compare_runs.py \\
        results/summary_wavlm_silero.json \\
        results/summary_rms_only.json \\
        --audio-quality \\
            output/wavlm_silero_audio_quality.json \\
            output/rms_only_audio_quality.json

    # Discover all summaries in a directory automatically:
    python evals/compare_runs.py --results-dir results/

Output example
--------------
  Pipeline / Condition      CAR       WER    mel_lift  stoi_lift
  ──────────────────────────────────────────────────────────────
  clean       (ceiling)     87.5%    0.121      —         —
  distorted   (floor)       51.2%    0.341      —         —
  ──────────────────────────────────────────────────────────────
  wavlm_silero              78.8%    0.183    +5.20     +0.080
  rms_only                  71.2%    0.218    +3.10     +0.041
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_REFERENCE_CONDITIONS = {"clean", "distorted"}


def _load_summary(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _label_from_summary(data: dict[str, Any], path: Path) -> str:
    """Return the human label for the pipeline-specific condition in this file."""
    meta = data.get("meta", {})
    if meta.get("label"):
        return meta["label"]
    # Fallback: use the non-reference key, or the filename stem
    pipeline_keys = [k for k in data if k not in _REFERENCE_CONDITIONS and k != "meta"]
    if len(pipeline_keys) == 1:
        return pipeline_keys[0]
    return path.stem


def _pipeline_stats(data: dict[str, Any]) -> dict[str, Any] | None:
    """Return stats for the pipeline-specific condition (not clean/distorted)."""
    for k, v in data.items():
        if k not in _REFERENCE_CONDITIONS and k != "meta" and isinstance(v, dict):
            return v
    return None


def _reference_stats(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return {condition_name: stats} for clean and distorted."""
    return {
        k: v for k, v in data.items()
        if k in _REFERENCE_CONDITIONS and isinstance(v, dict)
    }


# ---------------------------------------------------------------------------
# Audio-quality JSON helpers
# ---------------------------------------------------------------------------

def _load_audio_quality(path: Path) -> dict[str, Any] | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _aq_aggregate(data: dict[str, Any]) -> dict[str, float] | None:
    agg = data.get("aggregate")
    if not isinstance(agg, dict):
        return None
    return {
        "avg_mel_lift":  float(agg.get("avg_mel_lift",  0.0)),
        "avg_stoi_lift": float(agg.get("avg_stoi_lift", 0.0)),
        "n_improved":    int(agg.get("n_improved_both", 0)),
        "n":             int(agg.get("n",               0)),
    }


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------

def _car_str(stats: dict[str, Any]) -> str:
    car = stats.get("correct_action_rate")
    return f"{car:.1%}" if car is not None else "n/a"


def _wer_str(stats: dict[str, Any]) -> str:
    wer = stats.get("mean_wer")
    return f"{wer:.3f}" if wer is not None else "n/a"


def _aq_str(aq: dict[str, float] | None, key: str, fmt: str) -> str:
    if aq is None:
        return "—"
    v = aq.get(key)
    return format(v, fmt) if v is not None else "—"


def print_table(
    runs: list[dict[str, Any]],
    reference: dict[str, dict[str, Any]],
    show_aq: bool,
) -> None:
    """
    runs: list of {label, stats, aq} dicts
    reference: {condition_name: stats} for clean/distorted
    """
    CAR_W, WER_W, AQ_W = 8, 8, 10
    label_w = max(
        max((len(r["label"]) for r in runs), default=0),
        max((len(k) for k in reference), default=0),
        20,
    )

    hdr_parts = [f"  {'Pipeline / Condition':<{label_w}}  {'CAR':>{CAR_W}}  {'WER':>{WER_W}}"]
    if show_aq:
        hdr_parts += [f"  {'mel_lift':>{AQ_W}}  {'stoi_lift':>{AQ_W}}"]
    header = "".join(hdr_parts)
    sep = "  " + "─" * (len(header) - 2)

    print(header)
    print(sep)

    # Reference conditions: clean first, then distorted
    for cond in ["clean", "distorted"]:
        if cond not in reference:
            continue
        stats = reference[cond]
        tag = "(ceiling)" if cond == "clean" else "(floor)  "
        row = f"  {cond + ' ' + tag:<{label_w}}  {_car_str(stats):>{CAR_W}}  {_wer_str(stats):>{WER_W}}"
        if show_aq:
            row += f"  {'—':>{AQ_W}}  {'—':>{AQ_W}}"
        print(row)

    print(sep)

    # Pipeline versions
    for run in runs:
        aq = run.get("aq")
        row = (
            f"  {run['label']:<{label_w}}  "
            f"{_car_str(run['stats']):>{CAR_W}}  "
            f"{_wer_str(run['stats']):>{WER_W}}"
        )
        if show_aq:
            mel_s  = _aq_str(aq, "avg_mel_lift",  "+.2f")
            stoi_s = _aq_str(aq, "avg_stoi_lift", "+.3f")
            row += f"  {mel_s:>{AQ_W}}  {stoi_s:>{AQ_W}}"
        print(row)

    print()

    # N-sample footnote
    sample_ns = set()
    for r in runs:
        n = r["stats"].get("n")
        if n:
            sample_ns.add(n)
    for cond, stats in reference.items():
        n = stats.get("n")
        if n:
            sample_ns.add(n)
    if sample_ns:
        print(f"  Samples per condition: {', '.join(str(n) for n in sorted(sample_ns))}")

    if show_aq:
        print("  mel_lift  = avg (mel_MSE_corrupted - mel_MSE_inpainted); higher = better reconstruction")
        print("  stoi_lift = avg (STOI_inpainted - STOI_corrupted);       higher = more intelligible")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare multiple eval_runner.py summary JSONs in a single table."
    )
    p.add_argument(
        "summaries", nargs="*", metavar="SUMMARY_JSON",
        help="One or more summary JSON files produced by eval_runner.py.",
    )
    p.add_argument(
        "--results-dir", metavar="DIR",
        help="Auto-discover all summary_*.json files in this directory "
             "(used instead of positional args).",
    )
    p.add_argument(
        "--audio-quality", nargs="*", metavar="AQ_JSON",
        help="Optional audio-quality JSON files (from the Colab notebook) "
             "in the same order as the summary JSONs. Adds mel_lift and stoi_lift columns.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Collect summary paths ──────────────────────────────────────────────
    summary_paths: list[Path] = []
    if args.results_dir:
        summary_paths = sorted(Path(args.results_dir).glob("summary_*.json"))
        if not summary_paths:
            print(f"No summary_*.json files found in {args.results_dir}", file=sys.stderr)
            sys.exit(1)
    elif args.summaries:
        summary_paths = [Path(p) for p in args.summaries]
    else:
        print("Provide summary JSON paths or --results-dir.", file=sys.stderr)
        sys.exit(1)

    # ── Collect audio-quality paths (optional) ────────────────────────────
    aq_paths: list[Path | None] = []
    if args.audio_quality:
        aq_paths = [Path(p) for p in args.audio_quality]
        if len(aq_paths) != len(summary_paths):
            print(
                f"[WARN] --audio-quality has {len(aq_paths)} file(s) but "
                f"{len(summary_paths)} summary file(s) — audio quality will be skipped.",
                file=sys.stderr,
            )
            aq_paths = [None] * len(summary_paths)
    else:
        aq_paths = [None] * len(summary_paths)

    show_aq = any(p is not None for p in aq_paths)

    # ── Load and parse ─────────────────────────────────────────────────────
    print(f"Reading {len(summary_paths)} result file(s):\n")
    runs: list[dict[str, Any]] = []
    reference: dict[str, dict[str, Any]] = {}

    for path, aq_path in zip(summary_paths, aq_paths):
        data = _load_summary(path)
        label = _label_from_summary(data, path)
        pipeline_stats = _pipeline_stats(data)
        ref_stats = _reference_stats(data)

        meta = data.get("meta", {})
        ts = meta.get("timestamp", "")
        print(f"  {path.name}  [label={label}]  {ts}")

        # Merge reference conditions — first file wins for duplicates
        for cond, stats in ref_stats.items():
            reference.setdefault(cond, stats)

        if pipeline_stats is None:
            print(f"    [WARN] no pipeline condition found in {path.name} — skipping")
            continue

        aq_data = _load_audio_quality(aq_path) if aq_path else None
        aq = _aq_aggregate(aq_data) if aq_data else None

        runs.append({"label": label, "stats": pipeline_stats, "aq": aq})

    if not runs and not reference:
        print("\nNo usable data found.", file=sys.stderr)
        sys.exit(1)

    # ── Print table ────────────────────────────────────────────────────────
    print()
    print_table(runs, reference, show_aq=show_aq)


if __name__ == "__main__":
    main()
