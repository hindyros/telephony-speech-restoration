#!/usr/bin/env python3
"""Prepare data/clean/ from a local copy of the Harper Valley dataset.

Harper Valley layout (after cloning cricketclub/gridspace-stanford-harper-valley):

    data/
        audio/caller/<sid>.wav
        audio/agent/<sid>.wav
        metadata/<sid>.json
        transcript/<sid>.json

For each conversation we:
  1. Read metadata/<sid>.json to get tasks[0].task_type.
  2. Map it to a sandbox action label (see TASK_TYPE_TO_LABEL).
  3. Copy the caller-side audio to data/clean/<action_id>_<n>.wav,
     resampled to 16 kHz mono so the eval pipeline is consistent.
  4. Write a manifest CSV with sid, task_type, action_id, and source paths.

Example:
    python scripts/prepare_harper_valley.py \
        --source ~/datasets/gridspace-stanford-harper-valley \
        --max-per-task 20
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


TASK_TYPE_TO_LABEL = {
    "replace card": "replace_card",
    "transfer money": "transfer_money",
    "check balance": "check_balance",
    "order checks": "order_checks",
    "pay bill": "pay_bill",
    "reset password": "reset_password",
    "schedule appointment": "schedule_appointment",
    "get branch hours": "get_branch_hours",
}

LABEL_TO_ACTION_ID = {
    "replace_card": "B01",
    "transfer_money": "B02",
    "check_balance": "B03",
    "order_checks": "B04",
    "pay_bill": "B05",
    "reset_password": "B06",
    "schedule_appointment": "B07",
    "get_branch_hours": "B08",
}

DEFAULT_SAMPLE_RATE = 16_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build data/clean/ and a manifest from Harper Valley."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to the Harper Valley repo root (containing a 'data/' directory).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/clean",
        help="Destination for clean evaluation WAVs.",
    )
    parser.add_argument(
        "--manifest",
        default="data/clean/harper_valley_manifest.csv",
        help="CSV file summarizing chosen conversations.",
    )
    parser.add_argument(
        "--max-per-task",
        type=int,
        default=20,
        help="Cap on caller clips per task (balanced sampling).",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=4.0,
        help="Skip caller WAVs shorter than this many seconds.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=60.0,
        help="Skip caller WAVs longer than this many seconds.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate files even if they already exist.",
    )
    return parser.parse_args()


def read_task_type(metadata_path: Path) -> str | None:
    with metadata_path.open() as f:
        meta = json.load(f)
    tasks = meta.get("tasks") or []
    if not tasks:
        return None
    task_type = tasks[0].get("task_type")
    if isinstance(task_type, str):
        return task_type.strip().lower()
    return None


def probe_duration(wav_path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(wav_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return float(result.stdout.strip())


def convert_to_clean_wav(source_wav: Path, target_wav: Path) -> None:
    target_wav.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_wav),
        "-ac",
        "1",
        "-ar",
        str(DEFAULT_SAMPLE_RATE),
        str(target_wav),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


def main() -> None:
    args = parse_args()
    source_root = Path(args.source).expanduser().resolve()
    metadata_dir = source_root / "data" / "metadata"
    caller_dir = source_root / "data" / "audio" / "caller"

    if not metadata_dir.is_dir() or not caller_dir.is_dir():
        raise FileNotFoundError(
            f"Expected Harper Valley layout under {source_root}/data/. "
            "Make sure metadata/ and audio/caller/ exist."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    picked_per_label: dict[str, int] = {label: 0 for label in LABEL_TO_ACTION_ID}
    rows: list[dict[str, str | float]] = []
    skipped: dict[str, int] = {}

    metadata_files = sorted(metadata_dir.glob("*.json"))
    for metadata_path in metadata_files:
        sid = metadata_path.stem
        task_type = read_task_type(metadata_path)
        if task_type is None:
            skipped["no_task"] = skipped.get("no_task", 0) + 1
            continue
        label = TASK_TYPE_TO_LABEL.get(task_type)
        if label is None:
            skipped[f"unmapped:{task_type}"] = skipped.get(f"unmapped:{task_type}", 0) + 1
            continue

        caller_wav = caller_dir / f"{sid}.wav"
        if not caller_wav.exists():
            skipped["missing_caller_wav"] = skipped.get("missing_caller_wav", 0) + 1
            continue

        if picked_per_label[label] >= args.max_per_task:
            continue

        try:
            duration = probe_duration(caller_wav)
        except subprocess.CalledProcessError:
            skipped["probe_failed"] = skipped.get("probe_failed", 0) + 1
            continue

        if duration < args.min_duration or duration > args.max_duration:
            skipped["duration_filtered"] = skipped.get("duration_filtered", 0) + 1
            continue

        action_id = LABEL_TO_ACTION_ID[label]
        picked_per_label[label] += 1
        target_name = f"{action_id}_{picked_per_label[label]:02d}.wav"
        target_path = output_dir / target_name

        if target_path.exists() and not args.overwrite:
            continue

        convert_to_clean_wav(caller_wav, target_path)

        rows.append(
            {
                "filename": target_name,
                "action_id": action_id,
                "ground_truth_label": label,
                "task_type": task_type,
                "sid": sid,
                "source_caller_wav": str(caller_wav),
                "duration_seconds": round(duration, 3),
            }
        )
        print(f"Wrote {target_path} <- {caller_wav.name} ({task_type})")

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "action_id",
                "ground_truth_label",
                "task_type",
                "sid",
                "source_caller_wav",
                "duration_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSelected {len(rows)} clips")
    for label, count in picked_per_label.items():
        print(f"  {label:22s} {count}")
    if skipped:
        print("Skipped:")
        for reason, count in skipped.items():
            print(f"  {reason:22s} {count}")
    print(f"Manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
