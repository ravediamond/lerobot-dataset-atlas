"""Merge pipeline/state/datasets.json into the JSON shape space/index.html fetches.

Usage: python build.py [--in state/datasets.json] [--out ../space/data/datasets.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CAT_META = {
    "pick_place": {"label": "Pick & place", "color": "#3987e5"},
    "manipulation": {"label": "Fine manipulation", "color": "#d95926"},
    "bimanual": {"label": "Bimanual coordination", "color": "#199e70"},
    "assembly": {"label": "Assembly / insertion", "color": "#c98500"},
    "navigation": {"label": "Navigation", "color": "#d55181"},
    "other": {"label": "Other / unlabeled", "color": "#9085e9"},
}
# 4-tier quality grade. Old records (pre-v2 migration) only have the 3-tier
# 'quality' field (clean/minor/broken) — mapped onto the new grades below.
QUALITY_META = {
    "clean": {"label": "Clean", "color": "var(--good)"},
    "warning": {"label": "Warning", "color": "var(--warning)"},
    "degraded": {"label": "Degraded", "color": "var(--serious)"},
    "broken": {"label": "Broken", "color": "var(--critical)"},
}
LEGACY_QUALITY_MAP = {"clean": "clean", "minor": "warning", "broken": "broken"}


def to_row(r: dict) -> dict:
    cat = CAT_META[r.get("category", "other")]
    grade = r.get("quality_grade") or LEGACY_QUALITY_MAP.get(r.get("quality"), "warning")
    q = QUALITY_META[grade]
    owner, name = (r["id"].split("/", 1) + [r["id"]])[:2]
    embodiment = r.get("embodiment") or {}
    return {
        "id": r["id"],
        "name": name,
        "owner": owner,
        "robotType": r.get("robot_type") or "unknown",
        "bimanual": bool(r.get("bimanual")),
        "hardwareType": embodiment.get("hardware_type"),
        "cat": {"key": r.get("category", "other"), **cat},
        "episodes": r.get("episodes", 0),
        "cameras": r.get("cameras", 0),
        "cameraRoles": r.get("camera_roles") or {},
        "fps": r.get("fps"),
        "actionDim": r.get("action_dim"),
        "stateDim": r.get("state_dim"),
        "actionUnits": r.get("action_units"),
        "q": {"key": grade, **q},
        "qualityFlags": r.get("quality_flags") or [],
        "lastUpload": (r.get("last_modified") or "")[:10],
        "taskString": (r.get("task_strings") or [""])[0],
        "taskStringQuality": r.get("task_string_quality") or {},
        "downloads": r.get("downloads"),
        "hubUrl": f"https://huggingface.co/datasets/{r['id']}",
        "isDuplicateOf": r.get("is_duplicate_of"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", type=Path, default=Path(__file__).parent / "state" / "datasets.json")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent.parent / "space" / "data" / "datasets.json")
    args = parser.parse_args()

    state = json.loads(args.inp.read_text())
    records = [r for r in state.values() if "error" not in r and "excluded" not in r]
    rows = [to_row(r) for r in records]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"generated": True, "count": len(rows), "datasets": rows}, indent=2))
    print(f"built {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
