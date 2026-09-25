"""Tier 2: single-frame visual health micro-sampling.

For each dataset, extracts exactly one frame from its first camera's first
video shard (via ffmpeg reading the Hub URL directly — no manual download)
and scores it for blur (Laplacian variance) and information density (grayscale
entropy). Flags fully black/blank feeds and severely out-of-focus cameras.

Scoped-down vs. the full Tier 2 design: this does NOT implement
motion-conditioned camera-freeze detection (comparing a frame against
proprioceptive state to catch a frozen feed during active motion) — that
needs paired frame+state sampling at two timestamps and is real additional
engineering. Single-frame blur/entropy is the cheap, high-value 80% case
(catches black frames, capped lenses, fully out-of-focus cameras).

Requires: ffmpeg on PATH. Not run as part of the daily incremental update —
opt-in, separate pass, since it's slower and has an extra system dependency.

Usage: python tier2_sample.py [--workers 12] [--limit N]
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_url
from PIL import Image
from tqdm import tqdm

from backfill import load_state, save_state

BLUR_THRESHOLD = 15.0  # Laplacian variance below this: likely out of focus / blank
ENTROPY_THRESHOLD = 1.0  # grayscale Shannon entropy below this: near-uniform frame


def first_video_path(repo_id: str, camera_key: str) -> str | None:
    api = HfApi()
    prefix = f"videos/{camera_key}/"
    try:
        for entry in api.list_repo_tree(repo_id, repo_type="dataset", path_in_repo=f"videos/{camera_key}", recursive=True):
            if entry.path.startswith(prefix) and entry.path.endswith((".mp4", ".webm", ".mkv")):
                return entry.path
    except Exception:  # noqa: BLE001
        return None
    return None


def extract_frame(repo_id: str, video_path: str) -> np.ndarray | None:
    url = hf_hub_url(repo_id, video_path, repo_type="dataset")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "frame.jpg"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", url, "-frames:v", "1", "-f", "image2", str(out)],
                timeout=25, check=True, capture_output=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None
        if not out.exists():
            return None
        return np.array(Image.open(out).convert("L"))  # grayscale


def laplacian_variance(gray: np.ndarray) -> float:
    """4-neighbor discrete Laplacian, computed directly (no scipy dependency)."""
    g = gray.astype(np.float32)
    center = g[1:-1, 1:-1]
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * center
    return float(lap.var())


def grayscale_entropy(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray, bins=256, range=(0, 255), density=True)
    hist = hist[hist > 0]
    return float(-(hist * np.log2(hist)).sum())


def sample_one(repo_id: str, camera_keys: list[str]) -> tuple[str, dict]:
    if not camera_keys:
        return repo_id, {"tier2_status": "no_cameras"}
    video_path = first_video_path(repo_id, camera_keys[0])
    if not video_path:
        return repo_id, {"tier2_status": "video_not_found"}
    gray = extract_frame(repo_id, video_path)
    if gray is None:
        return repo_id, {"tier2_status": "extract_failed"}
    blur = laplacian_variance(gray)
    entropy = grayscale_entropy(gray)
    flags = []
    if blur < BLUR_THRESHOLD:
        flags.append("blurry_or_blank_frame")
    if entropy < ENTROPY_THRESHOLD:
        flags.append("low_entropy_frame")
    return repo_id, {
        "tier2_status": "ok",
        "tier2_laplacian_var": round(blur, 2),
        "tier2_entropy": round(entropy, 2),
        "tier2_flags": flags,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    state = load_state()
    candidates = [
        (k, v.get("camera_keys") or [])
        for k, v in state.items()
        if "error" not in v and "excluded" not in v
    ]
    if args.limit:
        candidates = candidates[: args.limit]
    print(f"sampling {len(candidates)} datasets")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(sample_one, repo_id, cams) for repo_id, cams in candidates]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="tier2 sampling"):
            repo_id, result = fut.result()
            state[repo_id].update(result)

    save_state(state)
    ok = sum(1 for k, _ in candidates if state[k].get("tier2_status") == "ok")
    flagged = sum(1 for k, _ in candidates if state[k].get("tier2_flags"))
    print(f"sampled {ok} ok, {flagged} flagged (blur/low-entropy)")


if __name__ == "__main__":
    main()
