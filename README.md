# Audio Restoration Evals

**Hypothesis:** An audio restoration pipeline applied to distorted banking-call voice input measurably improves agent correct-action rate compared to unrestored distorted audio.

## Experiment Overview

We evaluate three conditions over a fixed sandbox of 8 banking actions taken from the [Gridspace Stanford Harper Valley](https://github.com/cricketclub/gridspace-stanford-harper-valley) dataset. Every sample has a known ground-truth task label supplied by Harper Valley's metadata.

| Condition | Description | Expected Role |
|-----------|-------------|---------------|
| **Clean** | Original Harper Valley caller audio | Ceiling |
| **Distorted** | Telephony-style degraded version of the clean audio | Floor |
| **Restored** | Distorted audio passed through the restoration pipeline | System under test |

**Primary metric:** Correct action rate (did the agent identify the right banking task?)
**Secondary metrics:** Latency, awkward pauses (via VAD), roundtrip fidelity

## Pipeline

The restoration step is implemented in a Google Colab notebook:

> **[Restoration Pipeline Notebook →](notebooks/COLAB_LINK.md)**

After restoration, cleaned text or audio is fed directly to the agent. See [open question below](#open-question-text-vs-audio-output) for the rationale.

## Repository Structure

```
audio-restoration-evals/
├── README.md
├── notebooks/
│   └── COLAB_LINK.md                 # Link + instructions for the restoration notebook
├── sandbox/
│   └── actions.md                    # 8 banking actions (Harper Valley aligned)
├── data/
│   ├── raw/                          # Optional local copy of Harper Valley
│   ├── synthetic/                    # TTS smoke-test prompts (utterances.csv)
│   ├── clean/                        # Caller WAVs named <action_id>_<n>.wav
│   ├── distorted/                    # Telephony-style degraded counterparts
│   └── restored/                     # Output of the restoration notebook
├── evals/
│   └── eval_runner.py                # Scores all three conditions against GROUND_TRUTH
├── scripts/
│   ├── prepare_harper_valley.py      # Build data/clean/ from Harper Valley
│   └── generate_distorted_audio.py   # Build data/distorted/ from data/clean/
└── results/
    └── .gitkeep
```

## Sandbox Actions

The 8 actions match Harper Valley's `task_type` values. Full table in [`sandbox/actions.md`](sandbox/actions.md).

| ID | Harper Valley `task_type` | Label |
|----|---------------------------|-------|
| B01 | `replace card` | `replace_card` |
| B02 | `transfer money` | `transfer_money` |
| B03 | `check balance` | `check_balance` |
| B04 | `order checks` | `order_checks` |
| B05 | `pay bill` | `pay_bill` |
| B06 | `reset password` | `reset_password` |
| B07 | `schedule appointment` | `schedule_appointment` |
| B08 | `get branch hours` | `get_branch_hours` |

Filenames follow `<action_id>_<n>.wav`, e.g. `B03_05.wav`.

## Primary Dataset — Gridspace Stanford Harper Valley

- Real caller/agent WAVs split by speaker channel
- Ground-truth task type in `metadata/<sid>.json -> tasks[0].task_type`
- Repo: https://github.com/cricketclub/gridspace-stanford-harper-valley
- Clone it to any local path, then point the prep script at it:

```bash
git clone https://github.com/cricketclub/gridspace-stanford-harper-valley data/raw/harper-valley

python scripts/prepare_harper_valley.py \
    --source data/raw/harper-valley \
    --max-per-task 20
```

This populates `data/clean/` with caller-side WAVs renamed to the `B0X_NN.wav` scheme and writes `data/clean/harper_valley_manifest.csv` mapping each clip back to its `sid` and `task_type`.

## Smoke-Test Dataset — Synthetic TTS

A tiny TTS-generated set ships with the repo so the pipeline can be exercised before Harper Valley is downloaded. Prompts live in `data/synthetic/utterances.csv`. The committed `data/clean/` and `data/distorted/` files are this smoke-test set. They are intentionally short and artificial — use them for wiring, not for reporting results.

## Generating Distorted Audio

Once `data/clean/` is populated, generate the matched distorted condition with:

```bash
python scripts/generate_distorted_audio.py
```

This applies a telephony-style chain (band-limit, 8 kHz downsample, light compression, additive noise, short dropouts) and writes a `data/distorted/distortion_manifest.csv` with per-file parameters.

## Open Question: Text vs. Audio Output

**Decide this before running evals — it changes what you're testing.**

If the agent operates on text (or text + audio), feeding restored text directly avoids a lossy audio reconstruction step. If the agent requires audio input, you need to convert restored text back to speech. Current assumption: feed restored text directly to agent. Update `evals/eval_runner.py` accordingly.

## VAD Tooling

For pause/turn-taking metrics: [Picovoice Cobra](https://picovoice.ai/platform/cobra/) (free tier). Alternatives exist if needed.

## Running the Eval

```bash
# 1. Populate data/clean/
#    Option A (real): python scripts/prepare_harper_valley.py --source <harper-valley-root>
#    Option B (smoke test): use the synthetic set already in data/clean/

# 2. Generate distorted counterparts
python scripts/generate_distorted_audio.py

# 3. Run the restoration notebook on data/distorted/ -> data/restored/

# 4. Implement query_agent() in evals/eval_runner.py, then:
python evals/eval_runner.py --conditions clean distorted restored --output results/
```
