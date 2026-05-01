#!/usr/bin/env python3
"""Generate packet-loss-corrupted audio for the eval's distorted condition.

Two fill modes are supported (--fill-mode):

  zeros          Original model: zeroed samples.  Trivially detected by the
                 silence detector (P=R=F1=1.00), useful as a baseline.

  comfort-noise  Realistic model: cut region is filled with bandpass-filtered
                 Gaussian noise whose RMS matches the local noise floor
                 (RFC 3389 comfort-noise concept).  Forces the spectral-flux
                 detector to do the heavy lifting; silence detector will miss
                 most events.

Corruption is VAD-aware — cuts are placed only inside detected speech regions.
Harper Valley caller audio is ~10% speech; blind placement would mostly hit
silence.  Seeded by SHA256(stem) for full determinism without a manifest.
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
# Comfort-noise fill (RFC 3389 inspired)
# ---------------------------------------------------------------------------

def generate_comfort_noise(
    audio: np.ndarray,
    sample_rate: int,
    start: int,
    end: int,
    rng: np.random.Generator,
    *,
    context_ms: float = 20.0,
) -> np.ndarray:
    """Return telephony-band comfort noise matching the local noise floor.

    Steps
    -----
    1. Estimate target RMS from 20 ms of surrounding context.
    2. Generate white Gaussian noise and brick-wall bandpass filter it to
       300–3 400 Hz (standard telephony band) via FFT — pure numpy, no scipy.
    3. Scale filtered noise to target RMS.
    4. Apply 5 ms cosine fade-in / fade-out to eliminate click artefacts at
       cut boundaries.

    Parameters
    ----------
    audio       : full waveform (used only to read surrounding context)
    sample_rate : samples per second
    start, end  : sample indices of the region to fill
    context_ms  : how many ms on each side to use for noise-floor estimation
    """
    context_samples = ms_to_samples(context_ms, sample_rate)
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

    noise = rng.standard_normal(n_samples).astype(np.float32)
    spectrum = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
    spectrum[(freqs < 300.0) | (freqs > 3400.0)] = 0.0
    noise = np.fft.irfft(spectrum, n=n_samples).astype(np.float32)

    noise_rms = float(np.sqrt(np.mean(noise ** 2)))
    if noise_rms > 0:
        noise *= target_rms / noise_rms

    fade_samples = min(ms_to_samples(5.0, sample_rate), n_samples // 4)
    if fade_samples > 1:
        ramp = (0.5 * (1.0 - np.cos(np.pi * np.arange(fade_samples) / fade_samples))
                ).astype(np.float32)
        noise[:fade_samples]  *= ramp
        noise[-fade_samples:] *= ramp[::-1]

    return noise


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
    if float(np.sqrt(np.mean(audio ** 2))) == 0:
        return []
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
# Core corruption: VAD-aware random cuts
# ---------------------------------------------------------------------------

def apply_random_cuts(
    audio: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    speech_regions: list[tuple[int, int]],
    num_cuts: int = 4,
    cut_ms_range: tuple[float, float] = (1.0, 200.0),
    fill_mode: str = "zeros",
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Insert *num_cuts* gaps, sampled only from *speech_regions*.

    Parameters
    ----------
    fill_mode : ``"zeros"`` — zero out the cut (hard silence, original model).
                ``"comfort-noise"`` — fill with bandpass-filtered noise at the
                local noise floor (realistic RFC 3389 comfort noise).

    If no speech regions are detected, falls back to the full waveform so
    the script never silently skips a file.
    """
    if not speech_regions:
        speech_regions = [(0, len(audio))]

    min_cut_samples = ms_to_samples(cut_ms_range[0], sample_rate)
    candidates: list[tuple[int, int]] = []
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

        if fill_mode == "comfort-noise":
            audio[start:end] = generate_comfort_noise(audio, sample_rate, start, end, rng)
        else:
            audio[start:end] = 0.0

        events.append({
            "start_sec":   round(start / sample_rate, 4),
            "end_sec":     round(end   / sample_rate, 4),
            "duration_ms": round(cut_ms, 2),
            "fill_mode":   fill_mode,
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
    fill_mode: str = "zeros",
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
        fill_mode=fill_mode,
    )
    save_audio(output_path, corrupted, sample_rate)
    return {
        "filename":       input_path.name,
        "duration_sec":   round(len(audio) / sample_rate, 3),
        "speech_sec":     round(speech_sec, 3),
        "speech_regions": len(speech_regions),
        "sample_rate":    sample_rate,
        "num_cuts":       len(events),
        "fill_mode":      fill_mode,
        "cuts":           json.dumps(events),
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
                        help="Cuts per file (default: 4).")
    parser.add_argument("--cut-min-ms", type=float, default=1.0,
                        help="Minimum cut duration in ms (default: 1).")
    parser.add_argument("--cut-max-ms", type=float, default=200.0,
                        help="Maximum cut duration in ms (default: 200).")
    parser.add_argument(
        "--fill-mode", choices=["zeros", "comfort-noise"], default="zeros",
        help=(
            "How to fill cut regions.  "
            "'zeros' (default): hard silence — trivially detected by the silence "
            "detector, good as a controlled baseline.  "
            "'comfort-noise': bandpass-filtered noise at the local noise floor, "
            "modelling RFC 3389 CN injection — forces spectral-flux detection."
        ),
    )
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
        row = distort_file(input_path, output_path, args.num_cuts, cut_ms_range,
                           fill_mode=args.fill_mode)
        rows.append(row)
        print(
            f"Wrote {output_path.name}  "
            f"speech={row['speech_sec']:.1f}s/{row['duration_sec']:.1f}s  "
            f"regions={row['speech_regions']}  cuts={row['num_cuts']}  "
            f"fill={row['fill_mode']}"
        )

    fieldnames = ["filename", "duration_sec", "speech_sec", "speech_regions",
                  "sample_rate", "num_cuts", "fill_mode", "cuts"]
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nGenerated {len(rows)} distorted files in {output_dir}")
    print(f"Manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
