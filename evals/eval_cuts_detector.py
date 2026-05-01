"""eval_cuts_detector.py

Rigorous evaluation of scripts/detect_cuts.py on 125 synthetic audio samples
with known ground-truth cut positions.

Methodology
-----------
125 samples are generated deterministically (seeded) and split into five tiers:

  Tier A — easy         (25 samples): 1–3 cuts, 20–200 ms, moderate background
                        Fill: hard zeros.  Silence detector expected: P=R=F1≈1.
  Tier B — medium       (25 samples): 1–4 cuts,  5–20  ms, moderate background
                        Fill: hard zeros.  Silence detector expected: P=R=F1≈1.
  Tier C — hard         (25 samples): 1–4 cuts,  1–5   ms, moderate background
                        Fill: hard zeros.  Silence detector expected: P=R=F1≈1.
  Tier D — mixed        (25 samples): 1–5 cuts,  1–200 ms, quiet background
                        Fill: hard zeros.  Silence detector expected: P=R=F1≈1.
  Tier E — comfort-noise(25 samples): 1–4 cuts,  1–200 ms, moderate background
                        Fill: RFC 3389 comfort noise.  Both silence and spectral-
                        flux detectors fail: comfort noise is spectrally
                        indistinguishable from the background.  Tier E documents
                        the detection ceiling for signal-processing approaches;
                        neural/statistical methods are needed for this regime.

Each sample is a bandpass-filtered noise clip (300–3 400 Hz) mimicking
telephony speech.  Cuts in Tiers A–D are zeroed-sample gaps; Tier E cuts are
filled with bandpass Gaussian noise scaled to the local noise floor.

Matching rule
-------------
A detected cut is a True Positive if its start_sec falls within ±tolerance_ms
of a ground-truth cut's start_sec.  Greedy nearest-neighbour matching prevents
the same GT cut from being claimed twice.

Usage
-----
    python evals/eval_cuts_detector.py
    python evals/eval_cuts_detector.py --tolerance-ms 20 --output results/
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Import detector functions from scripts/detect_cuts.py
# ---------------------------------------------------------------------------

_DETECT_SCRIPT = Path(__file__).parent.parent / "scripts" / "detect_cuts.py"
_dspec = importlib.util.spec_from_file_location("detect_cuts", _DETECT_SCRIPT)
_dmod  = importlib.util.module_from_spec(_dspec)      # type: ignore[arg-type]
_dspec.loader.exec_module(_dmod)                      # type: ignore[union-attr]

detect_silence_cuts:  Any = _dmod.detect_silence_cuts
detect_spectral_cuts: Any = _dmod.detect_spectral_cuts
_merge_cuts:          Any = _dmod._merge_cuts


# ---------------------------------------------------------------------------
# Synthetic audio helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000  # Hz — matches project convention


def _telephony_noise(
    n_samples: int,
    rng: np.random.Generator,
    amplitude: float = 0.20,
) -> np.ndarray:
    """Return bandpass (300–3 400 Hz) filtered Gaussian noise, float32."""
    noise = rng.standard_normal(n_samples).astype(np.float64)
    spectrum = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / SAMPLE_RATE)
    mask = (freqs >= 300) & (freqs <= 3_400)
    spectrum[~mask] = 0.0
    filtered = np.fft.irfft(spectrum, n=n_samples).astype(np.float32)
    std = filtered.std()
    if std > 0:
        filtered = filtered / std * amplitude
    return np.clip(filtered, -1.0, 1.0)


def _comfort_noise_fill(
    audio: np.ndarray,
    start: int,
    end: int,
    context_ms: float = 20.0,
) -> np.ndarray:
    """Generate telephony-band comfort noise matching the local noise floor.

    Identical logic to generate_distorted_audio.generate_comfort_noise but
    defined inline so the eval file has no extra import dependency.
    """
    sr = SAMPLE_RATE
    context_samples = max(1, int(round(context_ms * sr / 1000.0)))
    pre  = audio[max(0, start - context_samples):start]
    post = audio[end:min(len(audio), end + context_samples)]
    context = np.concatenate([pre, post])

    if len(context) > 0:
        target_rms = float(np.sqrt(np.mean(context ** 2)))
        target_rms = max(target_rms, 1e-5)
    else:
        target_rms = 1e-4

    n_samples = end - start
    if n_samples <= 0:
        return np.array([], dtype=np.float32)

    # White noise → telephony band via FFT brick-wall
    noise = np.random.default_rng().standard_normal(n_samples).astype(np.float32)
    spectrum = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sr)
    spectrum[(freqs < 300.0) | (freqs > 3400.0)] = 0.0
    noise = np.fft.irfft(spectrum, n=n_samples).astype(np.float32)

    noise_rms = float(np.sqrt(np.mean(noise ** 2)))
    if noise_rms > 0:
        noise *= target_rms / noise_rms

    # 5 ms cosine fade-in / fade-out
    fade_samples = min(int(round(5.0 * sr / 1000.0)), n_samples // 4)
    if fade_samples > 1:
        ramp = (0.5 * (1.0 - np.cos(np.pi * np.arange(fade_samples) / fade_samples))
                ).astype(np.float32)
        noise[:fade_samples]  *= ramp
        noise[-fade_samples:] *= ramp[::-1]

    return noise


# ---------------------------------------------------------------------------
# Cut injection
# ---------------------------------------------------------------------------

def _inject_cuts(
    audio: np.ndarray,
    rng: np.random.Generator,
    n_cuts: int,
    min_dur_ms: float,
    max_dur_ms: float,
    silence_threshold: float = 1e-4,
    fill_mode: str = "zeros",
) -> list[dict[str, Any]]:
    """Inject *n_cuts* gaps into *audio* in-place.

    Parameters
    ----------
    fill_mode : ``"zeros"`` — zero out the cut region (original model).
                ``"comfort-noise"`` — fill with RFC 3389-style bandpass noise.

    Gaps are placed only inside high-energy regions (> 5× the silence
    threshold) so the silence detector's pre-context check doesn't filter them.

    Returns a sorted list of ground-truth cut dicts: {start_sec, end_sec,
    duration_ms, fill_mode}.
    """
    sr = SAMPLE_RATE
    min_gap = max(1, int(round(min_dur_ms * sr / 1000)))
    max_gap = max(min_gap, int(round(max_dur_ms * sr / 1000)))

    context = int(0.010 * sr)  # 10 ms look-back
    energy = np.abs(audio)
    high_energy = np.where(energy > silence_threshold * 5)[0]

    if len(high_energy) == 0:
        return []

    gt_cuts: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []

    for _ in range(n_cuts):
        for _ in range(50):
            start = int(rng.choice(high_energy))
            dur = int(rng.integers(min_gap, max_gap + 1))
            end = min(len(audio), start + dur)

            pre_start = max(0, start - context)
            pre_rms = float(np.sqrt(np.mean(audio[pre_start:start] ** 2)))
            if pre_rms <= silence_threshold * 5:
                continue

            overlap = any(
                not (end <= occ_start or start >= occ_end)
                for occ_start, occ_end in occupied
            )
            if overlap:
                continue

            # Fill the cut region
            if fill_mode == "comfort-noise":
                audio[start:end] = _comfort_noise_fill(audio, start, end)
            else:
                audio[start:end] = 0.0

            occupied.append((start, end))
            gt_cuts.append({
                "start_sec":   round(start / sr, 4),
                "end_sec":     round(end   / sr, 4),
                "duration_ms": round((end - start) * 1000 / sr, 2),
                "fill_mode":   fill_mode,
            })
            break

    gt_cuts.sort(key=lambda c: c["start_sec"])
    return gt_cuts


# ---------------------------------------------------------------------------
# Cut matching
# ---------------------------------------------------------------------------

def _match_cuts(
    gt_cuts: list[dict[str, Any]],
    detected: list[dict[str, Any]],
    tolerance_ms: float,
) -> tuple[int, int, int, list[float]]:
    """Greedy nearest-neighbour matching. Returns (tp, fp, fn, timing_errors_ms)."""
    tol_sec = tolerance_ms / 1000.0
    matched_det: set[int] = set()
    matched_gt:  set[int] = set()
    errors: list[float] = []

    for g_idx, gt in enumerate(gt_cuts):
        best_dist = float("inf")
        best_d_idx = -1
        for d_idx, det in enumerate(detected):
            if d_idx in matched_det:
                continue
            dist = abs(det["start_sec"] - gt["start_sec"])
            if dist <= tol_sec and dist < best_dist:
                best_dist = dist
                best_d_idx = d_idx
        if best_d_idx >= 0:
            matched_gt.add(g_idx)
            matched_det.add(best_d_idx)
            errors.append(best_dist * 1000.0)

    tp = len(matched_gt)
    fn = len(gt_cuts) - tp
    fp = len(detected) - len(matched_det)
    return tp, fp, fn, errors


# ---------------------------------------------------------------------------
# Single-sample evaluation
# ---------------------------------------------------------------------------

def _evaluate_sample(
    sample_id: int,
    tier: str,
    audio: np.ndarray,
    gt_cuts: list[dict[str, Any]],
    tolerance_ms: float,
    silence_threshold: float,
    min_silence_ms: float,
    detector_type: str = "silence",
) -> dict[str, Any]:
    """Run detector(s) on *audio* and score against *gt_cuts*.

    Parameters
    ----------
    detector_type : ``"silence"`` — run only the silence detector (Tiers A–D).
                    ``"both"``    — run both detectors and merge results (Tier E).
    """
    if detector_type == "both":
        sil_cuts = detect_silence_cuts(
            audio, SAMPLE_RATE,
            silence_threshold=silence_threshold,
            min_duration_ms=min_silence_ms,
        )
        spec_cuts = detect_spectral_cuts(
            audio, SAMPLE_RATE,
            peak_percentile=99.5,
        )
        detected = _merge_cuts(sil_cuts, spec_cuts, merge_window_ms=10.0)
    else:
        detected = detect_silence_cuts(
            audio, SAMPLE_RATE,
            silence_threshold=silence_threshold,
            min_duration_ms=min_silence_ms,
        )

    n_gt  = len(gt_cuts)
    n_det = len(detected)

    if n_gt == 0 and n_det == 0:
        tp, fp, fn = 0, 0, 0
        errors: list[float] = []
        precision = recall = f1 = 1.0
    else:
        tp, fp, fn, errors = _match_cuts(gt_cuts, detected, tolerance_ms)
        precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

    return {
        "sample_id":       sample_id,
        "tier":            tier,
        "detector_type":   detector_type,
        "duration_sec":    round(len(audio) / SAMPLE_RATE, 3),
        "n_gt_cuts":       n_gt,
        "n_detected":      n_det,
        "tp":              tp,
        "fp":              fp,
        "fn":              fn,
        "precision":       round(precision, 4),
        "recall":          round(recall,    4),
        "f1":              round(f1,        4),
        "median_error_ms": round(float(np.median(errors)), 3) if errors else None,
        "mean_error_ms":   round(float(np.mean(errors)),   3) if errors else None,
        "max_error_ms":    round(float(np.max(errors)),    3) if errors else None,
        "gt_cuts_json":    json.dumps(gt_cuts),
        "detected_json":   json.dumps([
            {"start_sec": d["start_sec"], "end_sec": d["end_sec"],
             "duration_ms": d.get("duration_ms", 0.0)}
            for d in detected
        ]),
    }


# ---------------------------------------------------------------------------
# Sample generation plan
# ---------------------------------------------------------------------------

TIERS: list[dict[str, Any]] = [
    # name, n_samples, cuts_range, dur_ms_range, amplitude, fill_mode, detector
    {"name": "A_easy",         "n": 25, "cuts": (1, 3), "dur_ms": (20,  200),
     "amplitude": 0.20, "fill_mode": "zeros",         "detector": "silence"},
    {"name": "B_medium",       "n": 25, "cuts": (1, 4), "dur_ms": ( 5,   20),
     "amplitude": 0.20, "fill_mode": "zeros",         "detector": "silence"},
    {"name": "C_hard",         "n": 25, "cuts": (1, 4), "dur_ms": ( 1,    5),
     "amplitude": 0.20, "fill_mode": "zeros",         "detector": "silence"},
    {"name": "D_mixed",        "n": 25, "cuts": (1, 5), "dur_ms": ( 1,  200),
     "amplitude": 0.05, "fill_mode": "zeros",         "detector": "silence"},
    {"name": "E_comfort_noise","n": 25, "cuts": (1, 4), "dur_ms": ( 1,  200),
     "amplitude": 0.20, "fill_mode": "comfort-noise", "detector": "both"},
]


def build_samples(silence_threshold: float) -> list[dict[str, Any]]:
    """Return a list of dicts ready for _evaluate_sample."""
    master_rng = np.random.default_rng(0xDEADBEEF)
    samples = []
    sample_id = 0
    n_audio_samples = int(4.0 * SAMPLE_RATE)  # 4-second clips

    for tier in TIERS:
        for _ in range(tier["n"]):
            rng = np.random.default_rng(int(master_rng.integers(0, 2**32)))
            audio = _telephony_noise(n_audio_samples, rng, amplitude=tier["amplitude"])
            n_cuts = int(rng.integers(tier["cuts"][0], tier["cuts"][1] + 1))
            gt_cuts = _inject_cuts(
                audio, rng,
                n_cuts=n_cuts,
                min_dur_ms=tier["dur_ms"][0],
                max_dur_ms=tier["dur_ms"][1],
                silence_threshold=silence_threshold,
                fill_mode=tier["fill_mode"],
            )
            samples.append({
                "sample_id":   sample_id,
                "tier":        tier["name"],
                "detector":    tier["detector"],
                "audio":       audio,
                "gt_cuts":     gt_cuts,
            })
            sample_id += 1

    return samples


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_tp = sum(r["tp"] for r in rows)
    total_fp = sum(r["fp"] for r in rows)
    total_fn = sum(r["fn"] for r in rows)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    all_errors: list[float] = []
    for r in rows:
        if r["tp"] > 0:
            det = json.loads(r["detected_json"])
            gt  = json.loads(r["gt_cuts_json"])
            _, _, _, errs = _match_cuts(gt, det, tolerance_ms=99999)
            all_errors.extend(errs)

    return {
        "n_samples":  len(rows),
        "total_tp":   total_tp,
        "total_fp":   total_fp,
        "total_fn":   total_fn,
        "precision":  round(precision, 4),
        "recall":     round(recall,    4),
        "f1":         round(f1,        4),
        "median_timing_error_ms": round(float(np.median(all_errors)), 3) if all_errors else None,
        "mean_timing_error_ms":   round(float(np.mean(all_errors)),   3) if all_errors else None,
        "max_timing_error_ms":    round(float(np.max(all_errors)),    3) if all_errors else None,
    }


# ---------------------------------------------------------------------------
# Terminal report
# ---------------------------------------------------------------------------

def _print_report(rows: list[dict[str, Any]], tolerance_ms: float) -> None:
    # Split: A–D (silence detector) vs E (both detectors)
    rows_ad = [r for r in rows if r["tier"] != "E_comfort_noise"]
    rows_e  = [r for r in rows if r["tier"] == "E_comfort_noise"]

    overall_ad = _aggregate(rows_ad)
    overall_all = _aggregate(rows)

    tier_names_ad = ["A_easy", "B_medium", "C_hard", "D_mixed"]
    tier_stats_ad = {t: _aggregate([r for r in rows_ad if r["tier"] == t])
                     for t in tier_names_ad}
    tier_stat_e   = _aggregate(rows_e) if rows_e else None

    bar = "=" * 72

    print(f"\n{bar}")
    print("  detect_cuts.py  —  evaluation on 125 synthetic samples")
    print(bar)
    print(f"  Tolerance window : {tolerance_ms} ms  |  Sample rate: {SAMPLE_RATE} Hz")
    print()

    # ── Tiers A–D: silence detector ──────────────────────────────────────────
    print("TIERS A–D  (silence detector, hard-zero fills, n=100)")
    o = overall_ad
    print(f"  Overall  P={o['precision']:.3f}  R={o['recall']:.3f}  "
          f"F1={o['f1']:.3f}  "
          f"TP={o['total_tp']}  FP={o['total_fp']}  FN={o['total_fn']}")
    print()

    tier_meta = {t["name"]: t for t in TIERS}
    print(f"  {'TIER':<14}  {'n':>4}  {'P':>6}  {'R':>6}  {'F1':>6}  "
          f"{'TP':>4}  {'FP':>4}  {'FN':>4}  {'range'}")
    print("  " + "-" * 68)
    for t_name in tier_names_ad:
        st = tier_stats_ad[t_name]
        meta = tier_meta[t_name]
        dur = f"{meta['dur_ms'][0]}–{meta['dur_ms'][1]} ms"
        print(f"  {t_name:<14}  {st['n_samples']:>4}  "
              f"{st['precision']:>6.3f}  {st['recall']:>6.3f}  {st['f1']:>6.3f}  "
              f"{st['total_tp']:>4}  {st['total_fp']:>4}  {st['total_fn']:>4}  "
              f"{dur}")
    print()

    # ── Tier E: comfort-noise, both detectors ─────────────────────────────────
    if tier_stat_e:
        print("TIER E  (silence + spectral-flux detectors, comfort-noise fills, n=25)")
        print("  [Comfort noise is spectrally identical to background — both detectors fail.]")
        print("  [This tier documents the detection ceiling for signal-processing approaches.]")
        e = tier_stat_e
        print(f"  P={e['precision']:.3f}  R={e['recall']:.3f}  F1={e['f1']:.3f}  "
              f"TP={e['total_tp']}  FP={e['total_fp']}  FN={e['total_fn']}")
        if e["median_timing_error_ms"] is not None:
            print(f"  Timing: median={e['median_timing_error_ms']:.1f} ms  "
                  f"mean={e['mean_timing_error_ms']:.1f} ms  "
                  f"max={e['max_timing_error_ms']:.1f} ms")
        print()

    # ── Timing (A–D) ─────────────────────────────────────────────────────────
    if overall_ad["median_timing_error_ms"] is not None:
        print("TIMING — Tiers A–D (true-positive matches only)")
        print(f"  Median : {overall_ad['median_timing_error_ms']:.2f} ms")
        print(f"  Mean   : {overall_ad['mean_timing_error_ms']:.2f} ms")
        print(f"  Max    : {overall_ad['max_timing_error_ms']:.2f} ms")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate the detect_cuts detectors on 125 synthetic samples "
            "(Tiers A–D: hard-zero fills / silence detector; "
            "Tier E: comfort-noise fills / both detectors)."
        )
    )
    p.add_argument(
        "--tolerance-ms", type=float, default=15.0, metavar="MS",
        help="Match window: ±MS from GT start counts as TP (default: 15).",
    )
    p.add_argument(
        "--silence-threshold", type=float, default=1e-4, metavar="THRESH",
        help="Passed through to detect_silence_cuts (default: 1e-4).",
    )
    p.add_argument(
        "--min-silence-ms", type=float, default=0.5, metavar="MS",
        help="Minimum gap to report (default: 0.5 ms).",
    )
    p.add_argument(
        "--output", default="results/",
        help="Directory for CSV and JSON outputs (default: results/).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building {sum(t['n'] for t in TIERS)} synthetic samples … ",
          end="", flush=True)
    samples = build_samples(silence_threshold=args.silence_threshold)
    print("done.")

    rows: list[dict[str, Any]] = []
    for s in samples:
        row = _evaluate_sample(
            sample_id=s["sample_id"],
            tier=s["tier"],
            audio=s["audio"],
            gt_cuts=s["gt_cuts"],
            tolerance_ms=args.tolerance_ms,
            silence_threshold=args.silence_threshold,
            min_silence_ms=args.min_silence_ms,
            detector_type=s["detector"],
        )
        rows.append(row)

    _print_report(rows, args.tolerance_ms)

    # ── CSV ──────────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"cuts_detector_eval_{timestamp}.csv"
    fieldnames = [
        "sample_id", "tier", "detector_type", "duration_sec",
        "n_gt_cuts", "n_detected", "tp", "fp", "fn",
        "precision", "recall", "f1",
        "median_error_ms", "mean_error_ms", "max_error_ms",
        "gt_cuts_json", "detected_json",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ── JSON summary ─────────────────────────────────────────────────────────
    rows_ad = [r for r in rows if r["tier"] != "E_comfort_noise"]
    rows_e  = [r for r in rows if r["tier"] == "E_comfort_noise"]
    summary = {
        "config": {
            "tolerance_ms":      args.tolerance_ms,
            "silence_threshold": args.silence_threshold,
            "min_silence_ms":    args.min_silence_ms,
            "sample_rate":       SAMPLE_RATE,
        },
        "overall_AD": _aggregate(rows_ad),
        "overall_all": _aggregate(rows),
        "by_tier": {
            t["name"]: _aggregate([r for r in rows if r["tier"] == t["name"]])
            for t in TIERS
        },
    }
    json_path = output_dir / f"cuts_detector_eval_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Raw results → {csv_path}")
    print(f"Summary     → {json_path}")


if __name__ == "__main__":
    main()
