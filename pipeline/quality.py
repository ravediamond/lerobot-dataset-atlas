"""Tier-0 deterministic quality/embodiment heuristics — zero network cost beyond
the meta/info.json + meta/stats.json already fetched by analyze.py. No LLM,
no video download. See LeRobot Dataset Atlas Architecture.md for the research
behind these checks.
"""

from __future__ import annotations

import math
import re

DEAD_STD_EPS = 1e-6

WRIST_PATTERNS = ("wrist", "hand", "gripper_cam", "eye_in_hand", "eef")
EXTERIOR_PATTERNS = (
    "top", "overhead", "front", "side", "external", "third_person",
    "base", "room", "laptop", "main", "fixed", "table", "scene", "phone",
)

PLACEHOLDER_RE = re.compile(r"^(task|robot|episode|demo|test|none|null|na|tbd)[\s_-]*\d*$", re.IGNORECASE)


def camera_role(key: str) -> str:
    k = key.lower()
    if any(p in k for p in WRIST_PATTERNS):
        return "wrist"
    if any(p in k for p in EXTERIOR_PATTERNS):
        return "exterior"
    return "unknown"


def _flatten(nested) -> list[float]:
    """Image channel stats come back as e.g. [[[0.38]], [[0.39]], [[0.40]]] —
    one nested singleton per channel. Flatten to one float per channel."""
    out = []
    for x in nested or []:
        while isinstance(x, list):
            x = x[0] if x else 0.0
        out.append(float(x))
    return out


def dead_action_channels(action_stats: dict) -> list[int]:
    std = action_stats.get("std") or []
    return [i for i, s in enumerate(std) if abs(s) < DEAD_STD_EPS]


def guess_unit_space(min_arr: list[float], max_arr: list[float]) -> str:
    vals = [v for v in (min_arr or []) + (max_arr or []) if v is not None]
    if not vals:
        return "unknown"
    lo, hi = min(vals), max(vals)
    if -1.05 <= lo and hi <= 1.05:
        return "normalized"
    if -3.3 <= lo and hi <= 3.3:
        return "radians"
    if -190 <= lo and hi <= 370:
        return "degrees"
    if abs(lo) > 1000 or abs(hi) > 1000:
        return "raw_ticks"
    return "unknown"


def image_health(mean_channels: list[float]) -> str:
    if not mean_channels:
        return "unknown"
    if all(m < 0.03 for m in mean_channels):
        return "black"
    if all(m > 0.97 for m in mean_channels):
        return "overexposed"
    return "healthy"


def has_quantile_stats(field_stats: dict) -> bool:
    return "q01" in field_stats or "q99" in field_stats


def shannon_entropy(s: str) -> float:
    s = s.strip()
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def is_placeholder_task(s: str) -> bool:
    s = s.strip()
    return not s or bool(PLACEHOLDER_RE.match(s)) or len(s) < 3


def task_string_quality(task_strings: list[str]) -> dict:
    strings = [t for t in task_strings if t is not None]
    if not strings:
        return {"total": 0, "placeholder_count": 0, "avg_entropy": 0.0}
    placeholder = sum(1 for t in strings if is_placeholder_task(t))
    avg_entropy = sum(shannon_entropy(t) for t in strings) / len(strings)
    return {"total": len(strings), "placeholder_count": placeholder, "avg_entropy": round(avg_entropy, 2)}


def classify_embodiment(action_dim: int | None, robot_type: str, bimanual: bool) -> dict:
    if bimanual:
        hw = "dual_arm"
    elif robot_type == "lekiwi":
        hw = "mobile_manipulator"
    elif robot_type == "reachy":
        hw = "humanoid"
    else:
        hw = "single_arm"
    return {"hardware_type": hw, "dof": action_dim}


def compute_quality(
    *,
    fps: int | None,
    episodes: int,
    cameras: int,
    camera_keys: list[str],
    stats: dict,
    features: dict,
    action_stats: dict,
    total_frames: int,
) -> dict:
    """Returns {"grade": clean|warning|degraded|broken, "flags": [...]}"""
    broken: list[str] = []
    degraded: list[str] = []
    warning: list[str] = []

    if not fps or episodes == 0 or cameras == 0:
        broken.append("missing_core_fields")

    # camera key <-> stats.json sync. Distinguish "stats.json absent entirely"
    # (common, not fatal — dataset is still usable, just lacks precomputed
    # normalization stats) from "stats.json exists but dropped a declared
    # camera key" (the real migration-tool bug: guaranteed KeyError at
    # training time when the ImageNet normalization pass runs).
    if not stats:
        warning.append("no_stats_file")
    else:
        missing_stats = [k for k in camera_keys if k not in stats]
        if missing_stats == camera_keys:
            # stats.json exists (has tabular/state stats) but skipped image
            # stats for every camera entirely — common (large video corpora
            # often skip per-pixel stats deliberately), dataset is still
            # usable with self-computed normalization. Not fatal.
            degraded.append("camera_stats_missing")
        elif missing_stats:
            # a SUBSET of declared cameras dropped from an otherwise-populated
            # stats.json — the real migration-tool bug: guaranteed KeyError
            # for that specific camera at training time.
            broken.append("camera_stats_missing_partial")

    # null feature names on video/image features
    for k in camera_keys:
        names = (features.get(k) or {}).get("names")
        if not names:
            broken.append("null_feature_names")
            break

    # frame/episode count invariant vs stats-reported sample count
    count = (action_stats.get("count") or [None])[0]
    if count is not None and total_frames and abs(count - total_frames) > max(5, 0.01 * total_frames):
        broken.append("frame_count_mismatch")

    # action channel degeneracy
    dead = dead_action_channels(action_stats)
    n_dims = len(action_stats.get("std") or [])
    if n_dims and len(dead) == n_dims:
        broken.append("all_action_channels_dead")
    elif dead:
        degraded.append("some_dead_action_channels")

    # image health per camera
    any_bad_image = False
    for k in camera_keys:
        cam_stats = stats.get(k) or {}
        mean = _flatten(cam_stats.get("mean"))
        health = image_health(mean)
        if health in ("black", "overexposed"):
            any_bad_image = True
    if any_bad_image:
        degraded.append("bad_camera_feed")

    # timestamp sanity — episode-relative seconds should be small; large values
    # usually mean raw epoch time leaked into the field
    ts = stats.get("timestamp") or {}
    ts_min = (ts.get("min") or [0])[0]
    ts_max = (ts.get("max") or [0])[0]
    if ts_min is not None and ts_min < -0.01:
        degraded.append("negative_timestamp")
    if ts_max is not None and ts_max > 100_000:
        degraded.append("timestamp_scale_anomaly")

    # unit-space / calibration sanity on action + state
    action_units = guess_unit_space(action_stats.get("min"), action_stats.get("max"))
    if action_units in ("raw_ticks", "unknown") and n_dims:
        warning.append("uncalibrated_or_unknown_units")

    # fps standardness
    if fps and fps not in (5, 10, 15, 20, 24, 25, 30, 50, 60):
        warning.append("nonstandard_fps")
    if episodes and episodes < 5:
        warning.append("very_few_episodes")

    # quantile stats (needed for flow-matching policies like pi0)
    if not has_quantile_stats(action_stats):
        warning.append("no_quantile_stats")

    if broken:
        grade = "broken"
    elif degraded:
        grade = "degraded"
    elif warning:
        grade = "warning"
    else:
        grade = "clean"

    return {"grade": grade, "flags": broken + degraded + warning, "action_units": action_units}
