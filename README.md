# lerobot-dataset-atlas

Indexes every public [LeRobot](https://github.com/huggingface/lerobot)-format dataset on the
Hugging Face Hub (~77k tagged repos), so you can filter by robot embodiment, cameras, task,
episode count, and a deterministic quality grade to find the right dataset for your use case.

Live demo: https://huggingface.co/spaces/ravediamond/lerobot-dataset-atlas

Architecture notes / research behind the quality heuristics: `LeRobot Dataset Atlas Architecture.md`.

## Pipeline (tiered, cost-ordered)

**Tier 0 — metadata only, zero network beyond `meta/info.json` + `meta/stats.json`:**

- `pipeline/discover.py` — list datasets tagged `LeRobot` on the Hub (one paginated call).
- `pipeline/analyze.py` — per dataset: fps, episodes, camera count/roles (wrist vs. exterior,
  from key-name patterns), action/state dims, embodiment (single/dual-arm, mobile, humanoid),
  action-space unit calibration (radians/degrees/normalized/raw-uncalibrated), and a 4-tier
  quality grade (`clean` / `warning` / `degraded` / `broken`) from `pipeline/quality.py`'s
  deterministic checks: dead action channels, camera/stats key sync, null feature names,
  frame-count invariants, timestamp sanity, black/overexposed camera feeds, task-string
  placeholder/entropy checks. No LLM calls.
- `pipeline/classify.py` — task-category taxonomy (pick-place, manipulation, bimanual, assembly,
  navigation, other). `--mode heuristic` (default) is instant keyword matching, no API. `--mode
  gemini` (Gemini free tier) or `--mode llm` (HF Inference, small paid cost) for ambiguous strings.

**Tier 1 — Hub API only, no downloads:**

- `pipeline/dedup.py` — fingerprints each dataset by hashing its sorted Git-LFS video-shard SHA-256
  object IDs (`list_repo_tree`, metadata only). Identical fingerprint = identical physical video
  content = re-upload/fork/mirror. Marks non-primary duplicates `is_duplicate_of` in the index.

**Tier 2 — small byte-range reads (opt-in, not in the daily job):**

- `pipeline/tier2_sample.py` — extracts one frame per dataset (ffmpeg reading the Hub URL
  directly) and scores it for blur (Laplacian variance) and information density (grayscale
  entropy), catching black/capped-lens/fully-out-of-focus cameras. Needs `ffmpeg` on PATH.
  Scoped down from the full design: does not do motion-conditioned camera-freeze detection
  (comparing frame pairs against proprioceptive state) — real additional engineering, deferred.

**Assembly:**

- `pipeline/build.py` — merges `pipeline/state/datasets.json` into `space/data/datasets.json`,
  the file the static page fetches.

## Running it

- `pipeline/backfill.py` — one-time (or occasional) full crawl, resumable (Ctrl+C-safe,
  checkpoints every 100 datasets). `--mode heuristic|llm|gemini` for classification.
- `pipeline/migrate_v2.py` — re-run Tier 0 only (no reclassification) over already-crawled
  datasets, to backfill new fields after a schema change like this one.
- `pipeline/update.py` — daily incremental: only (re)processes datasets that are new or whose
  `last_modified` changed. This is what CI runs.

Runs daily via `.github/workflows/refresh.yml` — needs repo secrets `HF_TOKEN` (write access to
the Space + HF dataset state repo) and `GEMINI_API_KEY` (classification, free tier).

## Repo layout

- `space/` — the HF Space (static SDK): `index.html` + generated `data/datasets.json`.
- `pipeline/` — the scripts above, run locally or in CI.
- `pipeline/state/datasets.json` — the full crawl state (~100MB+, gitignored — too large for
  GitHub; lives in the private HF dataset repo `ravediamond/lerobot-dataset-atlas-state`).

## Status

Full backfill complete (77k+ datasets discovered, ~22k usable after excluding tiny/broken ones).
Tier 0 v2 migration and Tier 1 dedup run as needed — check `pipeline/state/datasets.json` for
`quality_grade` / `is_duplicate_of` coverage. Tier 2 is opt-in, run manually.
