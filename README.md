# lerobot-dataset-atlas

Categorizes and visualizes every public [LeRobot](https://github.com/huggingface/lerobot)-format
dataset on the Hugging Face Hub, grouped by the robot platform that recorded it.

Live demo: https://huggingface.co/spaces/ravediamond/lerobot-dataset-atlas

## How it works

1. **Discover** — list datasets tagged `LeRobot` on the Hub.
2. **Code analysis** (no LLM) — pull `meta/info.json` + `episodes.jsonl` per dataset
   via `LeRobotDatasetMetadata`: fps, episode count, action/state dims, camera
   count, robot type. Flag missing fps, empty tasks, degenerate action ranges.
3. **Batched LLM pass** — normalize free-text task strings into a fixed category
   taxonomy (pick-place, manipulation, bimanual, assembly, navigation, other) via
   a batch API call. This is the only step that needs an LLM.
4. **Publish** — write results to JSON, served by a static HF Space.

## Repo layout

- `space/` — the HF Space (static SDK): `index.html` + generated `data/*.json`.
- `pipeline/` — offline dataset-analysis + batched LLM classification scripts.

## Status

Space UI is a working prototype on illustrative data. Pipeline (steps 1-3) not
yet wired up.
