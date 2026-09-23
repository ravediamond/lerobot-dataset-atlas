"""List every public dataset tagged LeRobot on the HF Hub.

Usage: python discover.py [--limit N] [--out discovered.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


def discover(limit: int | None = None) -> list[dict]:
    api = HfApi()
    results = []
    for ds in api.list_datasets(filter="LeRobot", full=True, limit=limit):
        results.append(
            {
                "id": ds.id,
                "downloads": getattr(ds, "downloads", None),
                "last_modified": str(ds.last_modified) if ds.last_modified else None,
                "private": ds.private,
                "tags": list(ds.tags or []),
            }
        )
        if limit and len(results) >= limit:
            break
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "out" / "discovered.json")
    args = parser.parse_args()

    results = discover(limit=args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"discovered {len(results)} datasets -> {args.out}")


if __name__ == "__main__":
    main()
