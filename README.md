# lerobot-dataset-atlas

Categorizes and visualizes every public [LeRobot](https://github.com/huggingface/lerobot)-format
dataset on the Hugging Face Hub, grouped by the robot platform that recorded it.

Live demo: https://huggingface.co/spaces/ravediamond/lerobot-dataset-atlas

## How it works

1. **Discover** (`pipeline/discover.py`) — list datasets tagged `LeRobot` on the Hub.
2. **Code analysis** (`pipeline/analyze.py`, no LLM) — pull `meta/info.json` per
   dataset: fps, episode count, action/state dims, camera count, robot type.
   Flags a quality tier (`clean` / `minor` / `broken`) from missing fps, zero
   cameras, non-standard fps, or tiny episode counts.
3. **Classification** (`pipeline/classify.py`) — normalize free-text task strings
   into a fixed category taxonomy (pick-place, manipulation, bimanual, assembly,
   navigation, other). `--mode heuristic` (default) is instant keyword matching;
   `--mode llm` calls HF Inference (`meta-llama/Llama-3.1-8B-Instruct`, concurrent
   requests) for higher-quality classification on ambiguous strings.
4. **Build** (`pipeline/build.py`) — merge into `space/data/datasets.json`, the
   file the static page fetches.

Runs weekly via `.github/workflows/refresh.yml` — needs one repo secret,
`HF_TOKEN` (write access to the Space), used for both Hub reads and HF Inference.

## Repo layout

- `space/` — the HF Space (static SDK): `index.html` + generated `data/datasets.json`.
- `pipeline/` — discover / analyze / classify / build scripts, run locally or in CI.

## Status

Pipeline is live end-to-end. Currently indexing a sample of the `LeRobot`-tagged
datasets on the Hub — bump the discover limit (or drop it) to cover the full tag.
