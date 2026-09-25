---
title: LeRobot Dataset Atlas
emoji: 🤖
colorFrom: yellow
colorTo: gray
sdk: static
pinned: false
license: apache-2.0
---

# LeRobot Dataset Atlas

Every public LeRobot-format dataset on the Hugging Face Hub (~22k after filtering out tiny/
broken ones), searchable by embodiment, cameras, task category, episode count, and a
deterministic quality grade — so you can actually find the right dataset for your use case
instead of guessing.

Data pipeline (offline, `pipeline/`): discover datasets tagged `LeRobot`, pull `meta/info.json`
+ `meta/stats.json` to compute embodiment, camera roles, action-space calibration, and a 4-tier
quality grade from zero-download deterministic checks (dead action channels, camera/stats sync,
black/overexposed feeds, task-string sanity), fingerprint via Hub LFS hashes to flag duplicate
uploads, classify task category (Gemini free tier), publish as static JSON (`data/datasets.json`)
this page reads. Refreshed daily by a GitHub Actions workflow.
