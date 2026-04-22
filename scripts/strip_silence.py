#!/usr/bin/env python3
"""Strip inter-utterance silence from caller audio in data/clean/.

Harper Valley caller files are 35-60 s long but contain only 10-30 s of
actual speech (the rest is dead air while the agent speaks on the other
channel). This script:

  1. Detects speech regions via an adaptive energy threshold (no extra deps).
  2. Pads each region by *pad_ms* on each side so word edges are not clipped.
  3. Concatenates all padded regions with a short *gap_ms* of silence between
     them so the result sounds natural.
  4. Overwrites the original files in-place (back up first if needed).

Typical result: 50 s → 10-15 s, all of which is actual speech.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_mono(path: Path) -> tuple[np.ndarray, int]:
    waveform, sr = sf.read(path, dtype="float32", always_2d=True)
    return waveform.mean(axis=1).astype(np.float32), sr


def save_audio(path: Path, audio: np.ndarray, sr: int) -> None:
    sf.write(path, np.clip(audio, -1.0, 1.0), sr)


def ms_to_samples(ms: float, sr: int) -> int:
    return max(1, int(round(ms * sr / 1000.0)))


# ---------------------------------------------------------------------------
# Adaptive energy VAD  (same logic as generate_distorted_audio.py)
# ---------------------------------------------------------------------------

def find_speech_regions(
    audio: np.ndarray,
    sr: int,
    frame_ms: float = 20.0,
    energy_threshold: float = 0.02,
    min_speech_ms: float = 100.0,
    merge_gap_ms: float = 200.0,
) -> list[tuple[int, int]]:
    frame_samples = ms_to_samples(frame_ms, sr)
    n_frames = len(audio) // frame_samples
    if n_frames == 0:
        return [(0, len(audio))]

    frame_rms = np.array([
        np.sqrt(np.mean(audio[i * frame_samples:(i + 1) * frame_samples] ** 2))
        for i in range(n_frames)
    ])

    noise_floor = np.percentile(frame_rms, 20)
    peak = frame_rms.max()
    threshold = noise_floor + energy_threshold * (peak - noise_floor)
    speech_mask = frame_rms > threshold

    merge_gap_frames = max(1, int(round(merge_gap_ms / frame_ms)))
    min_speech_frames = max(1, int(round(min_speech_ms / frame_ms)))

    regions: list[tuple[int, int]] = []
    in_speech = False
    start_frame = 0
    for i, active in enumerate(speech_mask):
        if active and not in_speech:
            start_frame, in_speech = i, True
        elif not active and in_speech:
            regions.append((start_frame, i))
            in_speech = False
    if in_speech:
        regions.append((start_frame, len(speech_mask)))

    # merge nearby regions
    merged: list[tuple[int, int]] = []
    for s, e in regions:
        if merged and s - merged[-1][1] <= merge_gap_frames:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    # drop very short fragments
    merged = [(s, e) for s, e in merged if e - s >= min_speech_frames]

    return [(s * frame_samples, min(e * frame_samples, len(audio))) for s, e in merged]


# ---------------------------------------------------------------------------
# Silence stripping
# ---------------------------------------------------------------------------

def strip_silence(
    audio: np.ndarray,
    sr: int,
    pad_ms: float = 80.0,
    gap_ms: float = 100.0,
) -> tuple[np.ndarray, list[tuple[int, int]], float]:
    """Return silence-stripped audio and the original speech regions.

    Each detected region is padded by *pad_ms* on both sides (clamped to
    array bounds) before concatenation. Regions are joined by *gap_ms* of
    silence so the result doesn't sound abruptly spliced.
    """
    regions = find_speech_regions(audio, sr)
    if not regions:
        return audio, [], float(len(audio) / sr)

    pad = ms_to_samples(pad_ms, sr)
    gap = np.zeros(ms_to_samples(gap_ms, sr), dtype=np.float32)

    chunks: list[np.ndarray] = []
    for i, (start, end) in enumerate(regions):
        s = max(0, start - pad)
        e = min(len(audio), end + pad)
        chunks.append(audio[s:e])
        if i < len(regions) - 1:
            chunks.append(gap)

    stripped = np.concatenate(chunks)
    original_speech_sec = sum(e - s for s, e in regions) / sr
    return stripped, regions, original_speech_sec


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strip inter-utterance silence from clean caller WAVs."
    )
    parser.add_argument("--input-dir", default="data/clean",
                        help="Directory of clean WAV files (modified in-place).")
    parser.add_argument("--pad-ms", type=float, default=80.0,
                        help="Padding around each speech region in ms (default: 80).")
    parser.add_argument("--gap-ms", type=float, default=100.0,
                        help="Silence gap inserted between regions in ms (default: 100).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    files = sorted(f for f in input_dir.glob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No WAV files found in {input_dir}")

    for path in files:
        audio, sr = load_mono(path)
        original_sec = len(audio) / sr
        stripped, regions, speech_sec = strip_silence(audio, sr, args.pad_ms, args.gap_ms)
        new_sec = len(stripped) / sr
        save_audio(path, stripped, sr)
        print(
            f"{path.name}  {original_sec:.1f}s → {new_sec:.1f}s  "
            f"({len(regions)} regions, {speech_sec:.1f}s speech)"
        )

    print(f"\nProcessed {len(files)} files in {input_dir} (in-place)")


if __name__ == "__main__":
    main()
