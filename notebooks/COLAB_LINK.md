# Restoration Pipeline — Google Colab Notebook

**[ADD COLAB LINK HERE]**

> Replace this file with the shareable Colab URL once the notebook is finalized.

## What the notebook does

1. Accepts distorted audio from `data/distorted/`
2. Runs the restoration pipeline
3. Outputs cleaned audio/text to `data/restored/`

## Usage

1. Open the notebook in Google Colab
2. Mount your Google Drive or upload `data/distorted/` samples
3. Run all cells
4. Download outputs to `data/restored/` in this repo

## Notes

- The restored output (text or audio) feeds directly into the agent in the eval — no additional TTS step unless the agent requires audio-only input
- See the [open question in README](../README.md#open-question-text-vs-audio-output) for context on this design choice
