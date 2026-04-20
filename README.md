# Audio Restoration Evals

**Hypothesis:** An audio restoration pipeline applied to distorted voice input measurably improves agent correct-action rate compared to unrestored distorted audio.

## Experiment Overview

We test three conditions on a sandbox of ~10 discrete agent actions with known ground truth:

| Condition | Description | Expected Role |
|-----------|-------------|---------------|
| **Clean** | Original undistorted audio | Ceiling |
| **Distorted** | Artificially degraded audio | Floor |
| **Restored** | Distorted audio passed through restoration pipeline | System under test |

**Primary metric:** Correct action rate (did the agent take the right action?)
**Secondary metrics:** Latency, awkward pauses (via VAD), roundtrip fidelity

## Pipeline

The restoration step is implemented in a Google Colab notebook:

> **[Restoration Pipeline Notebook →](notebooks/COLAB_LINK.md)**

After restoration, cleaned text/audio is fed directly to the agent. See [open question below](#open-question-text-vs-audio-output) for the rationale.

## Repository Structure

```
audio-restoration-evals/
├── README.md
├── notebooks/
│   └── COLAB_LINK.md          # Link + instructions for the restoration notebook
├── sandbox/
│   └── actions.md             # ~10 discrete actions with ground-truth labels
├── data/
│   ├── raw/                   # Source audio (Harper Valley or CALLHOME)
│   ├── synthetic/             # TTS-generated samples (20 per teammate)
│   ├── distorted/             # Noise-augmented versions of raw/synthetic
│   └── restored/              # Output of restoration pipeline
├── evals/
│   └── eval_runner.py         # Script to run all three conditions and score actions
└── results/
    └── .gitkeep
```

## Datasets

### Primary — Gridspace Stanford Harper Valley
- Real caller/agent WAVs, pre-split by speaker
- Repo: https://github.com/cricketclub/gridspace-stanford-harper-valley
- Download to `data/raw/harper-valley/`

### Synthetic
- Generate with any TTS tool (API, open-source model, or app)
- Target: 20 samples per teammate
- Save to `data/synthetic/`
- Noise augmentation is trivial — apply to produce `data/distorted/` counterparts

### Optional — UPenn CALLHOME American English
- Useful for isolating restoration quality from agent accuracy
- Free for MIT students via LDC library access
- Catalog: https://catalog.ldc.upenn.edu/LDC97S42
- Download to `data/raw/callhome/`

## Open Question: Text vs. Audio Output

**Decide this before running evals — it changes what you're testing.**

If the agent operates on text (or text + audio), feeding restored text directly avoids a lossy audio reconstruction step. If the agent requires audio input, you need to convert restored text back to speech. Current assumption: feed restored text directly to agent. Update `evals/eval_runner.py` accordingly.

## Sandbox Actions

See [`sandbox/actions.md`](sandbox/actions.md) for the full list. Target ~10 discrete, unambiguous actions (e.g., "file a return request," "check order status") each with a clear ground-truth expected outcome.

## VAD Tooling

For pause/turn-taking metrics: [Picovoice Cobra](https://picovoice.ai/platform/cobra/) (free tier). Alternatives exist if needed.

## Running the Eval

```bash
# 1. Populate data/ directories (see Datasets above)
# 2. Run the restoration notebook on data/distorted/ → data/restored/
# 3. Run the evaluator
python evals/eval_runner.py --conditions clean distorted restored --output results/
```

## Team

- 20 synthetic samples per teammate
