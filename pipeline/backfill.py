"""One-time (or occasional) full crawl of every LeRobot-tagged dataset on the
Hub. Safe to Ctrl+C at any point and re-run later: already-done datasets are
skipped on resume, and progress is checkpointed to disk every CHECKPOINT_EVERY
completions, not just at the end.

State lives in pipeline/state/datasets.json, a dict keyed by repo_id. This is
the file update.py (the daily incremental job) and build.py (site JSON) both
read.

Usage:
  python backfill.py                    # full crawl, resumable, LLM classification
  python backfill.py --mode heuristic    # no HF Inference calls, just keyword match
  python backfill.py --limit 500         # cap for a test run
  python backfill.py --workers 32        # tune concurrency
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from analyze import analyze_one
from classify import QuotaExceededError, classify_heuristic, classify_llm_single, text_for
from discover import discover

STATE_PATH = Path(__file__).parent / "state" / "datasets.json"
CHECKPOINT_EVERY = 100


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE_PATH)  # atomic on POSIX + Windows


def process_one(row: dict, mode: str, llm_client) -> tuple[str, dict]:
    repo_id = row["id"]
    try:
        rec = analyze_one(repo_id, tags=row.get("tags"))
    except Exception as e:  # noqa: BLE001 - never let one bad repo kill the crawl
        return repo_id, {"id": repo_id, "error": str(e)}
    if rec is None:
        return repo_id, {"id": repo_id, "error": "no meta/info.json"}
    if "excluded" in rec:
        return repo_id, rec

    rec["downloads"] = row.get("downloads")
    rec["last_modified"] = row.get("last_modified")

    if mode == "llm":
        _, cat = classify_llm_single(llm_client, rec)
    else:
        cat = classify_heuristic(text_for(rec))
    rec["category"] = cat
    return repo_id, rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["heuristic", "llm"], default="llm")
    parser.add_argument("--limit", type=int, default=None, help="cap discovery, for test runs")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--force", action="store_true", help="re-process datasets already in state")
    args = parser.parse_args()

    print("discovering...")
    discovered = discover(limit=args.limit)
    print(f"{len(discovered)} datasets tagged LeRobot on the Hub")

    state = load_state()
    todo = discovered if args.force else [d for d in discovered if d["id"] not in state]
    print(f"{len(state)} already in state, {len(todo)} to process this run")

    if not todo:
        print("nothing to do")
        return

    llm_client = None
    if args.mode == "llm":
        from huggingface_hub import InferenceClient

        llm_client = InferenceClient()

    completed_since_checkpoint = 0
    stop_requested = False

    def handle_sigint(signum, frame):  # noqa: ARG001
        nonlocal stop_requested
        if stop_requested:
            sys.exit(1)  # second Ctrl+C: hard exit
        print("\nstopping after in-flight requests finish, checkpointing... (Ctrl+C again to force)")
        stop_requested = True

    signal.signal(signal.SIGINT, handle_sigint)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, row, args.mode, llm_client): row["id"] for row in todo}
        pbar = tqdm(as_completed(futures), total=len(futures), desc=f"backfill ({args.mode})")
        for fut in pbar:
            if stop_requested:
                for f in futures:
                    f.cancel()
                break
            try:
                repo_id, rec = fut.result()
            except QuotaExceededError as e:
                print(f"\nHF Inference quota hit: {e}\nstopping — checkpointing and exiting.")
                for f in futures:
                    f.cancel()
                stop_requested = True
                break
            state[repo_id] = rec
            completed_since_checkpoint += 1
            if completed_since_checkpoint >= CHECKPOINT_EVERY:
                save_state(state)
                completed_since_checkpoint = 0
                pbar.set_postfix(saved=len(state))

    save_state(state)
    ok = sum(1 for r in state.values() if "error" not in r)
    err = len(state) - ok
    print(f"\nstate now has {len(state)} datasets ({ok} ok, {err} errored) -> {STATE_PATH}")
    if stop_requested:
        print("stopped early — re-run the same command to resume from here")


if __name__ == "__main__":
    main()
