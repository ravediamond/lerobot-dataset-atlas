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

Data pipeline (offline, `pipeline/`): discover datasets tagged `LeRobot`, pull
`meta/info.json` for hardware/fps/episode/camera stats, classify each task
string into a fixed category taxonomy via HF Inference, publish the result as
static JSON (`data/datasets.json`) this page reads. Refreshed weekly by a
GitHub Actions workflow.
