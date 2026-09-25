"""Tier 1: exact-duplicate detection via Hub Git-LFS content hashes.

Lists each dataset's file tree (metadata only, zero download) and fingerprints
it by hashing the sorted set of its video-shard and parquet-shard LFS SHA-256
object IDs. Two repos with the same video fingerprint contain byte-identical
visual trajectories — almost always a re-upload, fork, or mirror. The repo
with more downloads (tie-break: earliest last_modified) is kept as primary;
the rest are marked is_duplicate_of in state.

This is a separate pass from analyze.py because dedup is inherently a global
comparison across the whole catalog, not a per-repo computation.

Usage: python dedup.py [--workers 20]
"""

from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import HfApi
from tqdm import tqdm

from backfill import load_state, save_state


def fingerprint_one(repo_id: str) -> tuple[str, str | None, str | None]:
    """Returns (repo_id, video_hash, tabular_hash), either None on failure."""
    api = HfApi()
    try:
        tree = api.list_repo_tree(repo_id, repo_type="dataset", recursive=True)
        video_oids, tabular_oids = [], []
        for entry in tree:
            lfs = getattr(entry, "lfs", None)
            if not lfs:
                continue
            path = entry.path.lower()
            if path.startswith("videos/") or path.endswith((".mp4", ".mkv", ".webm")):
                video_oids.append(lfs.sha256)
            elif path.startswith("data/") or path.endswith(".parquet"):
                tabular_oids.append(lfs.sha256)
        video_hash = hashlib.sha256(",".join(sorted(video_oids)).encode()).hexdigest() if video_oids else None
        tabular_hash = hashlib.sha256(",".join(sorted(tabular_oids)).encode()).hexdigest() if tabular_oids else None
        return repo_id, video_hash, tabular_hash
    except Exception:  # noqa: BLE001 - dedup is best-effort, never block the pipeline
        return repo_id, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    state = load_state()
    candidates = {k: v for k, v in state.items() if "error" not in v and "excluded" not in v}
    print(f"fingerprinting {len(candidates)} datasets")

    fingerprints: dict[str, tuple[str | None, str | None]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fingerprint_one, repo_id) for repo_id in candidates]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="fingerprinting"):
            repo_id, video_hash, tabular_hash = fut.result()
            fingerprints[repo_id] = (video_hash, tabular_hash)
            state[repo_id]["video_lfs_hash"] = video_hash
            state[repo_id]["tabular_lfs_hash"] = tabular_hash

    # group by video fingerprint, keep the most-downloaded as primary
    groups: dict[str, list[str]] = {}
    for repo_id, (video_hash, _) in fingerprints.items():
        if video_hash:
            groups.setdefault(video_hash, []).append(repo_id)

    dup_count = 0
    for video_hash, repo_ids in groups.items():
        if len(repo_ids) < 2:
            continue
        ranked = sorted(
            repo_ids,
            key=lambda r: (-(state[r].get("downloads") or 0), state[r].get("last_modified") or ""),
        )
        primary = ranked[0]
        for repo_id in ranked[1:]:
            state[repo_id]["is_duplicate_of"] = primary
            dup_count += 1

    save_state(state)
    print(f"found {dup_count} duplicates across {sum(1 for g in groups.values() if len(g) > 1)} groups")


if __name__ == "__main__":
    main()
