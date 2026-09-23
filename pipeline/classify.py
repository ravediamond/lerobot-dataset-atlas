"""Normalize each dataset's task string(s) into a fixed category taxonomy.

Three modes:
  --mode heuristic   keyword match, no API key needed, runs instantly (default)
  --mode llm         HF Inference calls (concurrent), needs HF_TOKEN, small paid cost
  --mode gemini       Google AI Studio Gemini API, needs GEMINI_API_KEY, free tier

Usage: python classify.py --in out/analyzed.json [--out out/classified.json] [--mode heuristic|llm|gemini]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

CATEGORIES = ["pick_place", "manipulation", "bimanual", "assembly", "navigation", "other"]

KEYWORDS = {
    "bimanual": ["both arms", "bimanual", "two arms", "hand off", "hand-off", "transfer between"],
    "navigation": ["navigate", "drive to", "waypoint", "dock", "move to the", "explore"],
    "assembly": ["assemble", "screw", "insert the", "stack", "snap", "attach", "build"],
    "manipulation": ["rotate", "twist", "thread", "fold", "wipe", "pour", "open the drawer", "close the drawer", "push", "pull", "slide"],
    "pick_place": ["pick", "place", "put ", "grasp", "sort", "move the", "grab"],
}


def text_for(record: dict) -> str:
    if record.get("task_strings"):
        return " ; ".join(t for t in record["task_strings"] if t)
    return record["id"].split("/", 1)[-1].replace("_", " ").replace("-", " ")


def classify_heuristic(text: str) -> str:
    t = text.lower()
    for cat in ["bimanual", "navigation", "assembly", "manipulation", "pick_place"]:
        if any(kw in t for kw in KEYWORDS[cat]):
            return cat
    return "other"


LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PROMPT_TMPL = (
    "Classify this robot dataset's task into exactly one category: "
    f"{', '.join(CATEGORIES)}.\n"
    "Reply with only the category key, nothing else.\n\nTask: {text}"
)

QUOTA_MARKERS = (
    "exceeded your monthly included credits",
    "payment required",
    "quota",
    "exceeded your",
    "resource_exhausted",
    "resource exhausted",
    "rate limit",
)


class QuotaExceededError(RuntimeError):
    """Raised when HF Inference reports the account's usage limit is hit — the
    caller should stop issuing requests, not fall back to 'other' per record."""


def _is_quota_error(e: Exception) -> bool:
    status = getattr(getattr(e, "response", None), "status_code", None)
    if status in (402, 429):
        return True
    return any(marker in str(e).lower() for marker in QUOTA_MARKERS)


def classify_llm_single(client, record: dict) -> tuple[str, str]:
    try:
        resp = client.chat_completion(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": PROMPT_TMPL.format(text=text_for(record)[:400])}],
            max_tokens=10,
        )
        text = resp.choices[0].message.content.strip().lower()
        cleaned = re.sub(r"[^a-z_]", "", text)
        return record["id"], (cleaned if cleaned in CATEGORIES else "other")
    except Exception as e:
        if _is_quota_error(e):
            raise QuotaExceededError(str(e)) from e
        return record["id"], "other"  # transient/other error: fall back, don't kill the run


def classify_llm(records: list[dict], max_workers: int = 8) -> dict[str, str]:
    from huggingface_hub import InferenceClient

    client = InferenceClient()  # uses HF_TOKEN / cached `hf auth login` credential
    out: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(classify_llm_single, client, r) for r in records]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="classifying (hf inference)"):
            repo_id, cat = fut.result()
            out[repo_id] = cat
    return out


GEMINI_MODEL = "gemini-3.5-flash-lite"


def classify_gemini_single(client, record: dict) -> tuple[str, str]:
    try:
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=PROMPT_TMPL.format(text=text_for(record)[:400]),
        )
        text = (resp.text or "").strip().lower()
        cleaned = re.sub(r"[^a-z_]", "", text)
        return record["id"], (cleaned if cleaned in CATEGORIES else "other")
    except Exception as e:
        if _is_quota_error(e):
            raise QuotaExceededError(str(e)) from e
        return record["id"], "other"  # transient/other error: fall back, don't kill the run


def classify_gemini(records: list[dict], max_workers: int = 4) -> dict[str, str]:
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    out: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(classify_gemini_single, client, r) for r in records]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="classifying (gemini)"):
            repo_id, cat = fut.result()
            out[repo_id] = cat
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", type=Path, default=Path(__file__).parent / "out" / "analyzed.json")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "out" / "classified.json")
    parser.add_argument("--mode", choices=["heuristic", "llm", "gemini"], default="heuristic")
    args = parser.parse_args()

    records = json.loads(args.inp.read_text())

    if args.mode == "llm":
        cat_by_id = classify_llm(records)
        for r in records:
            r["category"] = cat_by_id.get(r["id"], "other")
    elif args.mode == "gemini":
        cat_by_id = classify_gemini(records)
        for r in records:
            r["category"] = cat_by_id.get(r["id"], "other")
    else:
        for r in records:
            r["category"] = classify_heuristic(text_for(r))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=2))
    print(f"classified {len(records)} datasets ({args.mode}) -> {args.out}")


if __name__ == "__main__":
    main()
