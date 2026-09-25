"""Pull meta/info.json + meta/stats.json (+ tasks, when present) for each
discovered dataset and compute code-analysis metrics: fps, episodes, camera
count, action/state dims, robot type, embodiment, and a 4-tier quality grade
(clean/warning/degraded/broken) from deterministic Tier-0 checks in quality.py.
No LLM calls, no video download.

Usage: python analyze.py --in out/discovered.json [--out out/analyzed.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
from tqdm import tqdm

from quality import camera_role, classify_embodiment, compute_quality, task_string_quality

MIN_EPISODES = 50

# ordered by specificity — first match wins
HARDWARE_KEYWORDS = [
    ("so101", "so101"), ("so100", "so100"), ("koch", "koch"), ("aloha", "aloha"),
    ("xarm", "xarm"), ("umi", "umi"), ("widowx", "widowx"), ("ur5e", "ur5e"),
    ("ur5", "ur5"), ("ur10", "ur10"), ("panda", "panda"), ("franka", "franka"),
    ("stretch", "stretch"), ("piper", "piper"), ("moss", "moss"), ("arx5", "arx5"),
    ("reachy", "reachy"), ("mycobot", "mycobot"), ("myarm", "myarm"), ("lekiwi", "lekiwi"),
    # academic / OpenX-style converted datasets — robot named in the repo slug
    ("kuka", "kuka"), ("fanuc", "fanuc"), ("utokyo_pr2", "pr2"), ("_pr2_", "pr2"),
    ("sawyer", "sawyer"), ("jaco", "jaco"), ("edan", "edan"), ("dlr_sara", "sara"),
    ("berkeley_gnm", "gnm"), ("baxter", "baxter"), ("locobot", "locobot"),
    ("hsr_", "hsr"), ("tiago", "tiago"), ("roarm", "roarm"), ("spot", "spot"),
    ("conq", "spot"), ("pusht", "pusht"), ("droid", "franka"),
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


def infer_robot_type_from_tags(tags: list[str] | None) -> str | None:
    """Hub tags are usually generic taxonomy (license:, format:, ...) but a few
    datasets carry a bare hardware-name tag (e.g. 'panda'). Cheap fallback."""
    if not tags:
        return None
    tagset = {t.lower() for t in tags}
    for kw, canon in HARDWARE_KEYWORDS:
        if kw in tagset:
            return canon
    return None


def normalize_robot_type(raw: str | None, repo_id: str, tags: list[str] | None = None) -> tuple[str, bool]:
    """Returns (canonical robot type, is_bimanual). Falls back to the dataset's
    Hub tags, then to inferring from the repo name, when the info.json field is
    missing/unset."""
    val = (raw or "").strip().lower()
    if not val or val == "unknown":
        inferred = infer_robot_type_from_tags(tags) or infer_robot_type_from_id(repo_id)
        return inferred or "unknown", False

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


def _download_json(repo_id: str, path: str) -> dict | None:
    try:
        p = hf_hub_download(repo_id, path, repo_type="dataset")
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None
    return json.loads(Path(p).read_text())


def analyze_one(repo_id: str, tags: list[str] | None = None) -> dict | None:
    info = _download_json(repo_id, "meta/info.json")
    if info is None:
        return None

    episodes = info.get("total_episodes", 0)
    if episodes < MIN_EPISODES:
        # skip stats.json, tasks, and the LLM classify call — too small to be useful
        return {"id": repo_id, "excluded": f"episodes<{MIN_EPISODES}", "episodes": episodes}

    features = info.get("features", {})
    camera_keys = [k for k, v in features.items() if v.get("dtype") in ("video", "image")]
    action = features.get("action", {})
    state = features.get("observation.state", {})
    action_dim = (action.get("shape") or [None])[0]

    task_strings: list[str] = []
    try:
        tasks_path = hf_hub_download(repo_id, "meta/tasks.jsonl", repo_type="dataset")
        for line in Path(tasks_path).read_text().splitlines():
            if line.strip():
                task_strings.append(json.loads(line).get("task", ""))
    except (EntryNotFoundError, RepositoryNotFoundError):
        pass

    stats = _download_json(repo_id, "meta/stats.json") or {}
    action_stats = stats.get("action") or {}

    robot_type, bimanual = normalize_robot_type(info.get("robot_type"), repo_id, tags)
    cameras_roles = {k: camera_role(k) for k in camera_keys}
    embodiment = classify_embodiment(action_dim, robot_type, bimanual)
    tstr_quality = task_string_quality(task_strings)
    q = compute_quality(
        fps=info.get("fps"),
        episodes=episodes,
        cameras=len(camera_keys),
        camera_keys=camera_keys,
        stats=stats,
        features=features,
        action_stats=action_stats,
        total_frames=info.get("total_frames", 0),
    )

    return {
        "id": repo_id,
        "robot_type": robot_type,
        "bimanual": bimanual,
        "embodiment": embodiment,
        "fps": info.get("fps"),
        "episodes": episodes,
        "frames": info.get("total_frames", 0),
        "cameras": len(camera_keys),
        "camera_keys": camera_keys,
        "camera_roles": cameras_roles,
        "action_dim": action_dim,
        "state_dim": (state.get("shape") or [None])[0],
        "action_units": q["action_units"],
        "task_strings": task_strings,
        "task_string_quality": tstr_quality,
        "quality": q["grade"],  # kept for backward compat with old 3-tier consumers
        "quality_grade": q["grade"],
        "quality_flags": q["flags"],
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
            analyzed = analyze_one(row["id"], tags=row.get("tags"))
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
