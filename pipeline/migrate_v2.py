"""One-time migration: re-run Tier-0 analysis (info.json + stats.json, no LLM)
over every already-crawled 'ok' record in state, to backfill the v2 fields
(embodiment, camera_roles, quality_grade, quality_flags, action_units,
task_string_quality) that didn't exist when it was first analyzed. Preserves
the existing 'category' field (no reclassification, no LLM calls).

Resumable like backfill.py: checkpoints every CHECKPOINT_EVERY, safe to Ctrl+C.

Usage: python migrate_v2.py [--workers 20] [--force]
"""

from __future__ import annotations

import argparse
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from analyze import analyze_one
from backfill import CHECKPOINT_EVERY, load_state, save_state


def migrate_one(repo_id: str, old_rec: dict) -> tuple[str, dict]:
    try:
        new_rec = analyze_one(repo_id)
    except Exception as e:  # noqa: BLE001
        return repo_id, old_rec  # keep the old record rather than lose data on a transient error
    if new_rec is None or "excluded" in new_rec:
        return repo_id, old_rec
    new_rec["category"] = old_rec.get("category", "other")
    new_rec["downloads"] = old_rec.get("downloads")
    new_rec["last_modified"] = old_rec.get("last_modified")
    new_rec["license"] = old_rec.get("license")
    return repo_id, new_rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--force", action="store_true", help="re-migrate records that already have quality_grade")
    args = parser.parse_args()

    state = load_state()
    todo = {
        k: v for k, v in state.items()
        if "error" not in v and "excluded" not in v and (args.force or "quality_grade" not in v)
    }
    print(f"{len(state)} total, {len(todo)} need v2 migration")
    if not todo:
        print("nothing to do")
        return

    stop_requested = False

    def handle_sigint(signum, frame):  # noqa: ARG001
        nonlocal stop_requested
        if stop_requested:
            sys.exit(1)
        print("\nstopping, checkpointing... (Ctrl+C again to force)")
        stop_requested = True

    signal.signal(signal.SIGINT, handle_sigint)

    completed_since_checkpoint = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(migrate_one, k, v): k for k, v in todo.items()}
        pbar = tqdm(as_completed(futures), total=len(futures), desc="migrating v2")
        for fut in pbar:
            if stop_requested:
                for f in futures:
                    f.cancel()
                break
            repo_id, rec = fut.result()
            state[repo_id] = rec
            completed_since_checkpoint += 1
            if completed_since_checkpoint >= CHECKPOINT_EVERY:
                save_state(state)
                completed_since_checkpoint = 0
                pbar.set_postfix(saved=len(state))

    save_state(state)
    print(f"migration done, {len(state)} total in state")


if __name__ == "__main__":
    main()
