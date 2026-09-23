"""Daily incremental refresh: only (re)process datasets that are new or whose
last_modified changed since the last run. Reads/writes the same
pipeline/state/datasets.json that backfill.py produces.

Usage: python update.py [--mode heuristic|llm|gemini] [--workers 20]
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from backfill import STATE_PATH, load_state, process_one, save_state
from classify import QuotaExceededError
from discover import discover


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["heuristic", "llm", "gemini"], default="llm")
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    print("discovering current dataset list...")
    discovered = discover()
    print(f"{len(discovered)} datasets tagged LeRobot on the Hub")

    state = load_state()
    if not state:
        print("no existing state — run backfill.py first for the initial full crawl")
        return

    todo = [
        row for row in discovered
        if row["id"] not in state or state[row["id"]].get("last_modified") != row.get("last_modified")
    ]
    print(f"{len(todo)} new or changed datasets to (re)process")

    removed = set(state) - {row["id"] for row in discovered}
    for repo_id in removed:
        del state[repo_id]
    if removed:
        print(f"dropped {len(removed)} datasets no longer tagged LeRobot")

    if not todo:
        save_state(state)
        print("no changes; state re-saved for the removals above")
        return

    client = None
    if args.mode == "llm":
        from huggingface_hub import InferenceClient

        client = InferenceClient()
    elif args.mode == "gemini":
        from google import genai

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        args.workers = min(args.workers, 4)  # respect free-tier rate limits

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_one, row, args.mode, client) for row in todo]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"update ({args.mode})"):
            try:
                repo_id, rec = fut.result()
            except QuotaExceededError as e:
                print(f"\n{args.mode} quota/rate limit hit: {e}\nstopping — checkpointing and exiting.")
                for f in futures:
                    f.cancel()
                break
            state[repo_id] = rec

    save_state(state)
    ok = sum(1 for r in state.values() if "error" not in r)
    print(f"state now has {len(state)} datasets ({ok} ok) -> {STATE_PATH}")


if __name__ == "__main__":
    main()
