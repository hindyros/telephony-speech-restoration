#!/usr/bin/env python3
"""Generate packet-loss-corrupted audio for the eval's distorted condition.

Corruption model: random silent cuts (zeroed samples) inserted at random
positions **within detected speech regions**, matching the approach in
hindyros/phronetic-ai-audioinpainting.

Harper Valley caller audio is ~10% speech; placing cuts blindly means
almost all of them land in silence. VAD-aware placement ensures every
cut hits actual speech so the distortion is meaningful and the restoration
task is well-defined.

VAD uses a simple energy threshold — no extra dependencies beyond numpy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_mono_audio(path: Path) -> tuple[np.ndarray, int]:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return waveform.mean(axis=1).astype(np.float32), sample_rate


def save_audio(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(waveform, -1.0, 1.0), sample_rate)


def ms_to_samples(ms: float, sample_rate: int) -> int:
    return max(1, int(round(ms * sample_rate / 1000.0)))


def rng_for_file(path: Path) -> np.random.Generator:
    seed = int.from_bytes(hashlib.sha256(path.stem.encode()).digest()[:8], "big")
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Energy-based VAD
# ---------------------------------------------------------------------------

def find_speech_regions(
    audio: np.ndarray,
    sample_rate: int,
    frame_ms: float = 20.0,
    energy_threshold: float = 0.02,
    min_speech_ms: float = 100.0,
    merge_gap_ms: float = 200.0,
) -> list[tuple[int, int]]:
    """Return a list of (start_sample, end_sample) speech regions.

    Uses per-frame RMS energy. Frames above *energy_threshold* × peak RMS
    are labelled speech. Short fragments and nearby regions are merged to
    avoid fragmenting individual words.
    """
    frame_samples = ms_to_samples(frame_ms, sample_rate)
    peak_rms = float(np.sqrt(np.mean(audio ** 2)))
    if peak_rms == 0:
        return []
    threshold = energy_threshold * np.sqrt(np.mean(audio ** 2) +
                                           np.max(audio ** 2)) / 2
    # per-frame RMS
    n_frames = len(audio) // frame_samples
    frame_rms = np.array([
        np.sqrt(np.mean(audio[i * frame_samples:(i + 1) * frame_samples] ** 2))
        for i in range(n_frames)
    ])

    # adaptive threshold: fraction of the dynamic range above the noise floor
    noise_floor = np.percentile(frame_rms, 20)
    peak = frame_rms.max()
    threshold = noise_floor + energy_threshold * (peak - noise_floor)

    speech_mask = frame_rms > threshold

    # convert frame indices to sample ranges, then merge nearby regions
    merge_gap_frames = max(1, int(round(merge_gap_ms / frame_ms)))
    min_speech_frames = max(1, int(round(min_speech_ms / frame_ms)))

    regions: list[tuple[int, int]] = []
    in_speech = False
    start_frame = 0
    for i, is_speech in enumerate(speech_mask):
        if is_speech and not in_speech:
            start_frame = i
            in_speech = True
        elif not is_speech and in_speech:
            regions.append((start_frame, i))
            in_speech = False
    if in_speech:
        regions.append((start_frame, len(speech_mask)))

    # merge regions separated by less than merge_gap_frames
    merged: list[tuple[int, int]] = []
    for start, end in regions:
        if merged and start - merged[-1][1] <= merge_gap_frames:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    # drop regions shorter than min_speech_frames
    merged = [(s, e) for s, e in merged if e - s >= min_speech_frames]

    # convert back to sample indices
    return [(s * frame_samples, min(e * frame_samples, len(audio)))
            for s, e in merged]


# ---------------------------------------------------------------------------
# Core corruption: VAD-aware random silent cuts
# ---------------------------------------------------------------------------

def apply_random_cuts(
    audio: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    speech_regions: list[tuple[int, int]],
    num_cuts: int = 4,
    cut_ms_range: tuple[float, float] = (1.0, 200.0),
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Insert *num_cuts* silent gaps, sampled only from *speech_regions*.

    If no speech regions are detected, falls back to the full waveform so
    the script never silently skips a file.
    """
    if not speech_regions:
        # fallback: treat whole file as one region
        speech_regions = [(0, len(audio))]

    # build a pool of valid cut-start positions: all samples inside speech
    # regions where a cut of at least cut_min_ms fits before the region ends
    min_cut_samples = ms_to_samples(cut_ms_range[0], sample_rate)
    candidates: list[tuple[int, int]] = []   # (region_start, region_end)
    for reg_start, reg_end in speech_regions:
        if reg_end - reg_start >= min_cut_samples:
            candidates.append((reg_start, reg_end))

    if not candidates:
        candidates = [(0, len(audio))]

    min_ms, max_ms = cut_ms_range
    events: list[dict[str, Any]] = []

    for _ in range(num_cuts):
        cut_ms = float(rng.uniform(min_ms, max_ms))
        cut_samples = ms_to_samples(cut_ms, sample_rate)

        # pick a region weighted by its length
        weights = np.array([e - s for s, e in candidates], dtype=float)
        weights /= weights.sum()
        idx = int(rng.choice(len(candidates), p=weights))
        reg_start, reg_end = candidates[idx]

        max_start = reg_end - cut_samples
        if max_start <= reg_start:
            max_start = reg_start
        start = int(rng.integers(reg_start, max(reg_start + 1, max_start + 1)))
        end = min(len(audio), start + cut_samples)

        audio[start:end] = 0.0
        events.append({
            "start_sec": round(start / sample_rate, 4),
            "end_sec": round(end / sample_rate, 4),
            "duration_ms": round(cut_ms, 2),
        })

    events.sort(key=lambda e: e["start_sec"])
    return np.clip(audio, -1.0, 1.0), events


# ---------------------------------------------------------------------------
# Per-file entry point
# ---------------------------------------------------------------------------

def distort_file(
    input_path: Path,
    output_path: Path,
    num_cuts: int,
    cut_ms_range: tuple[float, float],
) -> dict[str, Any]:
    audio, sample_rate = load_mono_audio(input_path)
    rng = rng_for_file(input_path)

    speech_regions = find_speech_regions(audio, sample_rate)
    speech_sec = sum(e - s for s, e in speech_regions) / sample_rate

    corrupted, events = apply_random_cuts(
        audio.copy(), sample_rate, rng,
        speech_regions=speech_regions,
        num_cuts=num_cuts,
        cut_ms_range=cut_ms_range,
    )
    save_audio(output_path, corrupted, sample_rate)
    return {
        "filename": input_path.name,
        "duration_sec": round(len(audio) / sample_rate, 3),
        "speech_sec": round(speech_sec, 3),
        "speech_regions": len(speech_regions),
        "sample_rate": sample_rate,
        "num_cuts": len(events),
        "cuts": json.dumps(events),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create VAD-aware packet-loss-corrupted audio from clean WAV files."
    )
    parser.add_argument("--input-dir", default="data/clean",
                        help="Directory containing clean WAV files.")
    parser.add_argument("--output-dir", default="data/distorted",
                        help="Directory for distorted WAV files.")
    parser.add_argument("--manifest", default="data/distorted/distortion_manifest.csv",
                        help="CSV file to write corruption metadata.")
    parser.add_argument("--num-cuts", type=int, default=4,
                        help="Silent cuts per file (default: 4).")
    parser.add_argument("--cut-min-ms", type=float, default=1.0,
                        help="Minimum cut duration in ms (default: 1).")
    parser.add_argument("--cut-max-ms", type=float, default=200.0,
                        help="Maximum cut duration in ms (default: 200).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate files even if they already exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)
    cut_ms_range = (args.cut_min_ms, args.cut_max_ms)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    input_files = sorted(f for f in input_dir.glob("*.wav"))
    if not input_files:
        raise FileNotFoundError(f"No WAV files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for input_path in input_files:
        output_path = output_dir / input_path.name
        if output_path.exists() and not args.overwrite:
            continue
        row = distort_file(input_path, output_path, args.num_cuts, cut_ms_range)
        rows.append(row)
        print(
            f"Wrote {output_path.name}  "
            f"speech={row['speech_sec']:.1f}s/{row['duration_sec']:.1f}s  "
            f"regions={row['speech_regions']}  cuts={row['num_cuts']}"
        )

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "duration_sec", "speech_sec",
                           "speech_regions", "sample_rate", "num_cuts", "cuts"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nGenerated {len(rows)} distorted files in {output_dir}")
    print(f"Manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
