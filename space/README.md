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

Every public LeRobot-format dataset on the Hugging Face Hub, grouped by the robot
platform that recorded it. Click a platform card to see its task-category
repartition, episode quality distribution, and a dataset sample.

Data pipeline (offline): pull `meta/info.json` + `episodes.jsonl` per dataset via
`LeRobotDatasetMetadata` for hardware/fps/episode stats, batch task-string
classification through an LLM for category normalization, publish the result as
static JSON this page reads.

Currently showing illustrative placeholder data while the pipeline is wired up.
