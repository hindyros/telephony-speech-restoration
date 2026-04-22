# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Does

Tests the hypothesis that audio restoration of telephony-degraded speech measurably improves a banking-task agent's correct-action rate. Three conditions are compared—**clean** (ceiling), **distorted** (floor), **restored** (system under test)—over 8 banking actions drawn from the Gridspace Stanford Harper Valley dataset.

## Common Commands

```bash
# Populate data/clean/ from the real Harper Valley dataset
python scripts/prepare_harper_valley.py \
    --source data/raw/harper-valley \
    --max-per-task 20

# Generate distorted counterparts (works on the synthetic smoke-test set too)
python scripts/generate_distorted_audio.py
python scripts/generate_distorted_audio.py --overwrite   # force regeneration

# Run restoration notebook in Google Colab (see notebooks/COLAB_LINK.md),
# then download outputs to data/restored/

# Run evaluations (all conditions or a subset)
python evals/eval_runner.py --conditions clean distorted restored --output results/
python evals/eval_runner.py --conditions distorted --output results/

# Syntax check
python -m py_compile scripts/prepare_harper_valley.py
python -m py_compile scripts/generate_distorted_audio.py
python -m py_compile evals/eval_runner.py
```

No external Python packages are required — scripts use only stdlib plus `ffmpeg`/`ffprobe` (must be on PATH).

## Architecture

```
Harper Valley caller WAVs
        │
        ▼
scripts/prepare_harper_valley.py   ──►  data/clean/<action_id>_<n>.wav
                                         data/clean/harper_valley_manifest.csv
        │
        ▼
scripts/generate_distorted_audio.py ──►  data/distorted/<action_id>_<n>.wav
                                          data/distorted/distortion_manifest.csv
        │
        ▼
Colab restoration notebook          ──►  data/restored/
        │
        ▼
evals/eval_runner.py   ──►  results/raw_<timestamp>.csv
                             results/summary_<timestamp>.json
```

**`evals/eval_runner.py`** is the integration point. It iterates WAVs in each condition directory, parses the `<action_id>` from the filename (e.g. `B03_05.wav` → `B03`), calls `query_agent(audio_path)`, and compares the returned action label against `GROUND_TRUTH`.

**`query_agent()` is the only stub that needs implementation** (`evals/eval_runner.py:61`). Options:
- Transcribe audio → feed text to an LLM classifier
- Feed audio directly if the agent supports it
- Load a `.txt` sidecar written by the Colab notebook (restored-text path)

## Key Design Decisions

- **File naming** — `<action_id>_<n>.wav` (e.g. `B03_05.wav`). The prefix is the ground-truth key; no manifest lookup is needed at eval time.
- **16 kHz mono throughout** — both prep scripts normalize via ffmpeg; `data/restored/` is expected to match.
- **Deterministic distortion** — `generate_distorted_audio.py` seeds its RNG from `sha256(filename.stem)` so distortion parameters are reproducible without a manifest lookup.
- **Text vs. audio output from restoration** — decide before running evals (see README § Open Question). This determines what `query_agent()` reads from `data/restored/`.

## Data Directories

| Path | Contents |
|------|----------|
| `data/clean/` | 16 kHz mono caller WAVs + `harper_valley_manifest.csv` |
| `data/distorted/` | Telephony-degraded counterparts + `distortion_manifest.csv` |
| `data/restored/` | Output of Colab notebook (not committed) |
| `data/raw/harper-valley/` | Harper Valley clone (not committed) |
| `results/` | Timestamped eval CSVs and JSON summaries |
