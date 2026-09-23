"""Pull meta/info.json (+ tasks, when present) for each discovered dataset and
compute the code-analysis metrics: fps, episodes, camera count, action/state
dims, robot type, quality flag. No LLM calls here.

Usage: python analyze.py --in out/discovered.json [--out out/analyzed.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
from tqdm import tqdm

VALID_FPS = {5, 10, 15, 20, 24, 25, 30, 50, 60}
MIN_EPISODES = 50

# ordered by specificity — first match wins
HARDWARE_KEYWORDS = [
    ("so101", "so101"), ("so100", "so100"), ("koch", "koch"), ("aloha", "aloha"),
    ("xarm", "xarm"), ("umi", "umi"), ("widowx", "widowx"), ("ur5e", "ur5e"),
    ("ur5", "ur5"), ("ur10", "ur10"), ("panda", "panda"), ("franka", "franka"),
    ("stretch", "stretch"), ("piper", "piper"), ("moss", "moss"), ("arx5", "arx5"),
    ("reachy", "reachy"), ("mycobot", "mycobot"), ("myarm", "myarm"), ("lekiwi", "lekiwi"),
]

# messy raw values seen in the wild -> canonical name
ROBOT_TYPE_ALIASES = {
    "so-100": "so100", "so100_ws": "so100", "so100_follower": "so100",
    "so101_follower": "so101", "aloha-stationary": "aloha", "so_follower": "so100",
}
SUFFIX_STRIP = ("_follower", "-follower", "_ws", "-ws")


def infer_robot_type_from_id(repo_id: str) -> str | None:
    name = repo_id.split("/", 1)[-1].lower()
    for kw, canon in HARDWARE_KEYWORDS:
        if kw in name:
            return canon
    return None


def normalize_robot_type(raw: str | None, repo_id: str) -> tuple[str, bool]:
    """Returns (canonical robot type, is_bimanual). Falls back to inferring the
    type from the dataset name when the metadata field is missing/unset."""
    val = (raw or "").strip().lower()
    if not val or val == "unknown":
        return (infer_robot_type_from_id(repo_id) or "unknown"), False

    bimanual = False
    if "bimanual" in val:
        bimanual = True
        val = val.replace("_bimanual", "").replace("-bimanual", "").replace("bimanual", "").strip("_- ")
    elif val.startswith("bi_"):
        bimanual = True
        val = val[3:]

    val = ROBOT_TYPE_ALIASES.get(val, val)
    for suf in SUFFIX_STRIP:
        if val.endswith(suf):
            val = val[: -len(suf)]
    val = ROBOT_TYPE_ALIASES.get(val, val)
    return val or "unknown", bimanual


def quality_flag(info: dict) -> str:
    fps = info.get("fps")
    episodes = info.get("total_episodes", 0)
    n_cams = info.get("_camera_count", 0)

    if not fps or episodes == 0 or n_cams == 0:
        return "broken"
    if fps not in VALID_FPS or episodes < 5:
        return "minor"
    return "clean"


def analyze_one(repo_id: str) -> dict | None:
    try:
        info_path = hf_hub_download(repo_id, "meta/info.json", repo_type="dataset")
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None
    info = json.loads(Path(info_path).read_text())

    episodes = info.get("total_episodes", 0)
    if episodes < MIN_EPISODES:
        # skip the tasks.jsonl fetch and LLM classify call — too small to be useful
        return {"id": repo_id, "excluded": f"episodes<{MIN_EPISODES}", "episodes": episodes}

    features = info.get("features", {})
    camera_keys = [k for k, v in features.items() if v.get("dtype") in ("video", "image")]
    action = features.get("action", {})
    state = features.get("observation.state", {})

    task_strings: list[str] = []
    try:
        tasks_path = hf_hub_download(repo_id, "meta/tasks.jsonl", repo_type="dataset")
        for line in Path(tasks_path).read_text().splitlines():
            if line.strip():
                task_strings.append(json.loads(line).get("task", ""))
    except (EntryNotFoundError, RepositoryNotFoundError):
        pass

    info["_camera_count"] = len(camera_keys)
    robot_type, bimanual = normalize_robot_type(info.get("robot_type"), repo_id)

    return {
        "id": repo_id,
        "robot_type": robot_type,
        "bimanual": bimanual,
        "fps": info.get("fps"),
        "episodes": info.get("total_episodes", 0),
        "frames": info.get("total_frames", 0),
        "cameras": len(camera_keys),
        "camera_keys": camera_keys,
        "action_dim": (action.get("shape") or [None])[0],
        "state_dim": (state.get("shape") or [None])[0],
        "task_strings": task_strings,
        "quality": quality_flag(info),
        "codebase_version": info.get("codebase_version"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", type=Path, default=Path(__file__).parent / "out" / "discovered.json")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "out" / "analyzed.json")
    args = parser.parse_args()

    discovered = json.loads(args.inp.read_text())
    results = []
    skipped = []
    for row in tqdm(discovered, desc="analyzing"):
        try:
            analyzed = analyze_one(row["id"])
        except Exception as e:  # noqa: BLE001 - keep the pipeline moving on a bad repo
            skipped.append({"id": row["id"], "error": str(e)})
            continue
        if analyzed is None:
            skipped.append({"id": row["id"], "error": "no meta/info.json"})
            continue
        analyzed["downloads"] = row.get("downloads")
        analyzed["last_modified"] = row.get("last_modified")
        results.append(analyzed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"analyzed {len(results)} datasets, skipped {len(skipped)} -> {args.out}")
    if skipped:
        (args.out.parent / "skipped.json").write_text(json.dumps(skipped, indent=2))


if __name__ == "__main__":
    main()
