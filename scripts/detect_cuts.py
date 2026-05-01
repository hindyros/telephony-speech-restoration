#!/usr/bin/env python3
"""Detect cut points (silent gaps and/or hard splices) in an audio file.

Two complementary detectors are run and their results merged:

  1. **Silent-gap detector** — finds contiguous near-zero-amplitude regions.
     Best for the synthetic distorted files in this project, where cuts are
     zeroed-sample gaps of 1–200 ms inserted during distortion.

  2. **Spectral-flux detector** — computes the frame-to-frame change in the
     STFT magnitude spectrum and peaks above an adaptive threshold.  Catches
     hard splices where two different recordings are joined without silence.

Output JSON schema
------------------
{
  "file": "<path>",
  "duration_sec": <float>,
  "sample_rate": <int>,
  "detector": "<silence|spectral|both>",
  "cuts": [
    {
      "start_sec":   <float>,   // start of the cut / discontinuity
      "end_sec":     <float>,   // end  (== start for spectral-flux peaks)
      "duration_ms": <float>,
      "type":        "<silence|spectral>",
      "confidence":  <float>    // 0–1; 1.0 for definite silence
    },
    ...
  ]
}

Usage
-----
    python scripts/detect_cuts.py path/to/audio.wav
    python scripts/detect_cuts.py path/to/audio.wav --output cuts.json
    # For silent/zeroed gaps (this project's distorted files):
    python scripts/detect_cuts.py path/to/audio.wav --detector silence

    # For hard splices (content changes without silence):
    python scripts/detect_cuts.py path/to/audio.wav --detector spectral --peak-percentile 99.9

    python scripts/detect_cuts.py path/to/audio.wav --merge-window-ms 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------

def load_mono_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load any audio file as 32-bit float mono."""
    waveform, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    return waveform.mean(axis=1).astype(np.float32), sample_rate


# ---------------------------------------------------------------------------
# Detector 1 — silent-gap detector
# ---------------------------------------------------------------------------

def detect_silence_cuts(
    audio: np.ndarray,
    sample_rate: int,
    *,
    silence_threshold: float = 1e-4,
    min_duration_ms: float = 0.5,
    pre_context_ms: float = 5.0,
) -> list[dict[str, Any]]:
    """Return silent regions longer than *min_duration_ms*.

    Parameters
    ----------
    silence_threshold:
        Amplitude (absolute) below which a sample is considered silent.
        The default (1e-4) works well for 16-bit PCM stored as float32.
        Increase to ~5e-3 for very noisy signals.
    min_duration_ms:
        Ignore silent gaps shorter than this many milliseconds.
    pre_context_ms:
        Look-back window used to compute a local RMS so that we can skip
        regions that were already silent (e.g. pauses in speech) rather than
        artificially inserted cuts.  A sudden drop is more interesting than
        a continuation of existing silence.

    Returns a list of dicts with keys start_sec, end_sec, duration_ms, type,
    confidence.
    """
    min_samples = max(1, int(round(min_duration_ms * sample_rate / 1000.0)))
    context_samples = max(1, int(round(pre_context_ms * sample_rate / 1000.0)))

    silent_mask = np.abs(audio) <= silence_threshold

    cuts: list[dict[str, Any]] = []
    in_silence = False
    start_idx = 0

    for i, is_silent in enumerate(silent_mask):
        if is_silent and not in_silence:
            start_idx = i
            in_silence = True
        elif not is_silent and in_silence:
            duration_samples = i - start_idx
            if duration_samples >= min_samples:
                # Check whether the audio was already quiet before the cut
                # (genuine speech pause) vs. suddenly dropped to zero.
                pre_start = max(0, start_idx - context_samples)
                pre_rms = float(np.sqrt(np.mean(audio[pre_start:start_idx] ** 2)))
                # If the audio before the cut was already near-silent, skip it.
                if pre_rms > silence_threshold * 5:
                    confidence = min(1.0, pre_rms / 0.05)
                    cuts.append({
                        "start_sec":   round(start_idx / sample_rate, 4),
                        "end_sec":     round(i / sample_rate, 4),
                        "duration_ms": round(duration_samples * 1000 / sample_rate, 2),
                        "type":        "silence",
                        "confidence":  round(confidence, 4),
                    })
            in_silence = False

    # Handle trailing silence
    if in_silence:
        duration_samples = len(audio) - start_idx
        if duration_samples >= min_samples:
            pre_start = max(0, start_idx - context_samples)
            pre_rms = float(np.sqrt(np.mean(audio[pre_start:start_idx] ** 2)))
            if pre_rms > silence_threshold * 5:
                confidence = min(1.0, pre_rms / 0.05)
                cuts.append({
                    "start_sec":   round(start_idx / sample_rate, 4),
                    "end_sec":     round(len(audio) / sample_rate, 4),
                    "duration_ms": round(duration_samples * 1000 / sample_rate, 2),
                    "type":        "silence",
                    "confidence":  round(confidence, 4),
                })

    return cuts


# ---------------------------------------------------------------------------
# Detector 2 — spectral-flux detector
# ---------------------------------------------------------------------------

def _stft_magnitude(
    audio: np.ndarray,
    frame_size: int,
    hop_size: int,
) -> np.ndarray:
    """Return |STFT| matrix of shape (n_frames, n_bins) using a Hann window."""
    window = np.hanning(frame_size)
    n_frames = 1 + (len(audio) - frame_size) // hop_size
    mag = np.zeros((n_frames, frame_size // 2 + 1), dtype=np.float32)
    for k in range(n_frames):
        frame = audio[k * hop_size: k * hop_size + frame_size] * window
        mag[k] = np.abs(np.fft.rfft(frame))
    return mag


def detect_spectral_cuts(
    audio: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: float = 10.0,
    hop_ms: float = 2.0,
    peak_percentile: float = 97.0,
    min_gap_ms: float = 10.0,
) -> list[dict[str, Any]]:
    """Detect abrupt spectral changes via per-frame spectral flux.

    Spectral flux = sum of positive magnitude differences between consecutive
    STFT frames (half-wave rectified).  A spike in flux marks a hard splice.

    Parameters
    ----------
    frame_ms:
        STFT window length in ms.
    hop_ms:
        STFT hop in ms.  Smaller → finer time resolution, slower.
    peak_percentile:
        Flux peaks above this percentile are flagged as cut candidates.
    min_gap_ms:
        Merge candidate peaks closer than this (de-bounce).
    """
    frame_size = max(64, int(round(frame_ms * sample_rate / 1000.0)))
    hop_size   = max(1,  int(round(hop_ms   * sample_rate / 1000.0)))
    min_gap_frames = max(1, int(round(min_gap_ms / hop_ms)))

    if len(audio) < frame_size:
        return []

    mag = _stft_magnitude(audio, frame_size, hop_size)

    # Half-wave rectified spectral flux: only sudden increases count
    flux = np.sum(np.maximum(0.0, np.diff(mag, axis=0)), axis=1)

    # Adaptive threshold: percentile of non-zero flux values
    nonzero = flux[flux > 0]
    if len(nonzero) == 0:
        return []
    threshold = float(np.percentile(nonzero, peak_percentile))

    # Normalise flux to [0, 1] for confidence
    max_flux = float(flux.max()) if flux.max() > 0 else 1.0

    candidate_frames = np.where(flux > threshold)[0]

    # De-bounce: keep only the highest-flux frame within each min_gap window
    cuts: list[dict[str, Any]] = []
    i = 0
    while i < len(candidate_frames):
        cluster = [candidate_frames[i]]
        j = i + 1
        while j < len(candidate_frames) and candidate_frames[j] - candidate_frames[i] < min_gap_frames:
            cluster.append(candidate_frames[j])
            j += 1
        best_frame = max(cluster, key=lambda f: flux[f])
        # +1 because diff shrinks the array by one; frame k in diff corresponds
        # to the transition between frame k and k+1 in the original magnitude.
        time_sec = (best_frame + 1) * hop_size / sample_rate
        cuts.append({
            "start_sec":   round(time_sec, 4),
            "end_sec":     round(time_sec, 4),
            "duration_ms": 0.0,
            "type":        "spectral",
            "confidence":  round(float(flux[best_frame]) / max_flux, 4),
        })
        i = j

    return cuts


# ---------------------------------------------------------------------------
# Merge / deduplicate cuts from both detectors
# ---------------------------------------------------------------------------

def _merge_cuts(
    silence_cuts: list[dict[str, Any]],
    spectral_cuts: list[dict[str, Any]],
    merge_window_ms: float,
) -> list[dict[str, Any]]:
    """Combine and deduplicate cuts.

    Spectral-flux peaks that fall inside a known silent region are dropped
    (the silence detector already captured that event).  Remaining peaks that
    are within *merge_window_ms* of a silent cut are also dropped.
    """
    all_cuts = list(silence_cuts)
    window_sec = merge_window_ms / 1000.0

    for sc in spectral_cuts:
        t = sc["start_sec"]
        dominated = False
        for sil in silence_cuts:
            # inside the silent region or within the merge window
            if sil["start_sec"] - window_sec <= t <= sil["end_sec"] + window_sec:
                dominated = True
                break
        if not dominated:
            all_cuts.append(sc)

    all_cuts.sort(key=lambda c: c["start_sec"])
    return all_cuts


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def detect_cuts_in_file(
    audio_path: Path,
    output_path: Path | None,
    detector: str,
    silence_threshold: float,
    min_silence_ms: float,
    frame_ms: float,
    hop_ms: float,
    peak_percentile: float,
    merge_window_ms: float,
) -> dict[str, Any]:
    audio, sample_rate = load_mono_audio(audio_path)
    duration_sec = len(audio) / sample_rate

    silence_cuts: list[dict[str, Any]] = []
    spectral_cuts: list[dict[str, Any]] = []

    if detector in ("silence", "both"):
        silence_cuts = detect_silence_cuts(
            audio, sample_rate,
            silence_threshold=silence_threshold,
            min_duration_ms=min_silence_ms,
        )

    if detector in ("spectral", "both"):
        spectral_cuts = detect_spectral_cuts(
            audio, sample_rate,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
            peak_percentile=peak_percentile,
        )

    if detector == "both":
        cuts = _merge_cuts(silence_cuts, spectral_cuts, merge_window_ms)
    elif detector == "silence":
        cuts = silence_cuts
    else:
        cuts = spectral_cuts

    result: dict[str, Any] = {
        "file":         str(audio_path),
        "duration_sec": round(duration_sec, 4),
        "sample_rate":  sample_rate,
        "detector":     detector,
        "cuts":         cuts,
    }

    if output_path is None:
        output_path = audio_path.with_suffix(".cuts.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect silent gaps and/or hard splices in a WAV file."
    )
    parser.add_argument("input", help="Path to the input WAV file.")
    parser.add_argument(
        "--output", "-o",
        help="Path for the output JSON file. "
             "Defaults to <input>.cuts.json next to the input file.",
    )
    parser.add_argument(
        "--detector", choices=["silence", "spectral", "both"], default="both",
        help="Which detector(s) to run (default: both).",
    )

    # Silence-detector params
    sil = parser.add_argument_group("silence detector")
    sil.add_argument(
        "--silence-threshold", type=float, default=1e-4, metavar="THRESH",
        help="Amplitude below which a sample is 'silent' (default: 1e-4).",
    )
    sil.add_argument(
        "--min-silence-ms", type=float, default=0.5, metavar="MS",
        help="Ignore silent gaps shorter than this (default: 0.5 ms).",
    )

    # Spectral-flux params
    spec = parser.add_argument_group("spectral-flux detector")
    spec.add_argument(
        "--frame-ms", type=float, default=10.0, metavar="MS",
        help="STFT window size in ms (default: 10).",
    )
    spec.add_argument(
        "--hop-ms", type=float, default=2.0, metavar="MS",
        help="STFT hop size in ms (default: 2).",
    )
    spec.add_argument(
        "--peak-percentile", type=float, default=99.5, metavar="PCT",
        help="Spectral-flux peaks above this percentile are flagged (default: 99.5). "
             "Raise toward 99.9 to reduce false positives on noisy signals.",
    )

    # Merge params
    parser.add_argument(
        "--merge-window-ms", type=float, default=10.0, metavar="MS",
        help="Spectral peaks within this window of a silent cut are deduplicated "
             "(default: 10 ms).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_path = Path(args.input)

    if not audio_path.exists():
        raise FileNotFoundError(f"Input file not found: {audio_path}")

    output_path = Path(args.output) if args.output else None

    result = detect_cuts_in_file(
        audio_path=audio_path,
        output_path=output_path,
        detector=args.detector,
        silence_threshold=args.silence_threshold,
        min_silence_ms=args.min_silence_ms,
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        peak_percentile=args.peak_percentile,
        merge_window_ms=args.merge_window_ms,
    )

    n = len(result["cuts"])
    out_file = output_path or audio_path.with_suffix(".cuts.json")
    print(f"Detected {n} cut(s) in {audio_path.name}  ({result['duration_sec']:.2f}s)")
    for c in result["cuts"]:
        if c["type"] == "silence":
            print(f"  [{c['type']:8s}]  {c['start_sec']:.4f}s – {c['end_sec']:.4f}s  "
                  f"({c['duration_ms']:.1f} ms)  confidence={c['confidence']:.2f}")
        else:
            print(f"  [{c['type']:8s}]  {c['start_sec']:.4f}s               "
                  f"              confidence={c['confidence']:.2f}")
    print(f"Output -> {out_file}")


if __name__ == "__main__":
    main()
