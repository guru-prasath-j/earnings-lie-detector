#!/usr/bin/env python3
"""
LLM-scored guidance features from earnings press releases.

The rule-based ``guidance_features`` know the *direction* of a guidance change but not its size, whether it
is about EPS or revenue, or the tone around it.  Here an LLM reads only the outlook passages of each release
(sentences mentioning guidance / outlook / forecast, plus two neighbours each side) and fills a fixed JSON schema.

Uses NVIDIA's OpenAI-compatible API (build.nvidia.com).  The key is read from ``NVIDIA_API_KEY`` or from
``~/personal/.secrets/nvidia.env`` (a line ``export NVIDIA_API_KEY=...``) and is never printed.

Results are cached one JSON line per release in ``data/processed/llm_guidance.jsonl``; reruns skip releases
already scored, so an interrupted run just resumes.  Releases with no outlook passage are not sent.

Usage:
    python src/models/llm_guidance.py --limit 20 --show      # quality check on 20 releases
    python src/models/llm_guidance.py                        # all releases (~1,400 calls; free tier queues, 16-250 s each → hours)

Failed releases are cached as ``{"error": "failed"}``; ``--retry-failed`` drops them from the cache and scores them again.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from text_features import _BOILERPLATE_RE, _OUTLOOK_RE, _SENT_RE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("llm_guidance")
logging.getLogger("httpx").setLevel(logging.WARNING)

BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "openai/gpt-oss-20b"  # llama-3.3-70b was retired (HTTP 410); probed 2026-10-28
KEY_FILE = Path.home() / "personal" / ".secrets" / "nvidia.env"
CACHE = Path("data/processed/llm_guidance.jsonl")
MAX_PASSAGE_CHARS = 6000

DIRECTIONS = ["raised", "lowered", "maintained", "initiated", "given", "withdrawn", "not_mentioned"]
SYSTEM_PROMPT = f"""You read the outlook section of a company's quarterly earnings press release and extract facts.
Answer with ONE JSON object and nothing else, with exactly these keys:
  "eps_guidance": one of {DIRECTIONS} -- what the company did to its EPS / earnings-per-share guidance in THIS release
  "eps_change_pct": number or null -- change of the EPS guidance midpoint vs the prior guidance: % change for a dollar
      figure (e.g. 2.5 or -4); percentage-point change if the guidance is a growth rate (6% -> 8% is 2); null if not stated
  "revenue_guidance": one of {DIRECTIONS} -- same, for revenue / sales guidance (organic sales counts)
  "revenue_change_pct": number or null -- same, for the revenue guidance midpoint
  "horizon": one of ["full_year", "next_quarter", "both", "none"] -- period the guidance covers
  "outlook_tone": integer from -2 (very cautious) to 2 (very confident) -- management's tone about the coming period
  "headwinds": true or false -- does the outlook cite specific headwinds or risks (macro, FX, tariffs, demand, costs)?
  "specificity": 0, 1 or 2 -- 0 no forward guidance, 1 qualitative only, 2 numeric ranges or targets
Rules: "raised" / "lowered" only if the text says the guidance changed versus a previous figure. "maintained" only
if the text says it is reaffirmed / unchanged. "initiated" only if the text says the outlook is new (first outlook
for a new year, reinstated after a suspension). "given" when figures are stated but the text does not say how they
compare with earlier guidance. Guidance for a single product, segment, expense line, margin, cash flow or tax rate
does not count as EPS or revenue guidance."""


def load_api_key() -> str:
    key = os.environ.get("NVIDIA_API_KEY")
    if not key and KEY_FILE.exists():
        m = re.search(r"NVIDIA_API_KEY\s*=\s*['\"]?([^'\"\s]+)", KEY_FILE.read_text())
        key = m.group(1) if m else None
    if not key:
        raise SystemExit(f"NVIDIA_API_KEY not set and not found in {KEY_FILE}")
    return key


def outlook_passage(text: str) -> str:
    """Sentences mentioning guidance/outlook/forecast plus two neighbours each side, in order."""
    sents = _SENT_RE.split(text)
    keep = set()
    for i, s in enumerate(sents):
        if len(s) <= 800 and _OUTLOOK_RE.search(s) and not _BOILERPLATE_RE.search(s):
            keep.update(range(i - 2, i + 3))
    passage = " ".join(sents[i] for i in sorted(keep) if 0 <= i < len(sents))
    return passage[:MAX_PASSAGE_CHARS]


def parse_answer(raw: str) -> Optional[dict]:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        ans = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if ans.get("eps_guidance") not in DIRECTIONS or ans.get("revenue_guidance") not in DIRECTIONS:
        return None
    return ans


class RateLimiter:
    """At most ``rpm`` calls per minute across threads."""

    def __init__(self, rpm: float):
        self.interval, self.next_t, self.lock = 60.0 / rpm, 0.0, threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next_t)
            self.next_t = t + self.interval
        time.sleep(max(0.0, t - now))


def score(client, model: str, limiter: RateLimiter, ticker: str, date: str, passage: str) -> dict:
    user = f"Company: {ticker}\nRelease date: {date}\n\nOutlook passages:\n{passage}"
    for attempt in range(5):
        limiter.wait()
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0, max_tokens=1000,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}])
            ans = parse_answer(resp.choices[0].message.content or "")
            if ans is not None:
                return ans
            logger.warning("%s %s: unparseable answer (attempt %d)", ticker, date, attempt + 1)
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            logger.warning("%s %s: %s %s (attempt %d)", ticker, date, type(exc).__name__, status or "", attempt + 1)
            if status is not None and 400 <= status < 500 and status != 429:
                break  # permanent (bad request, model gone, auth) → don't retry
            time.sleep(min(60, 5 * 2 ** attempt))  # rate limit / timeout / server error → back off
    return {"error": "failed"}


def load_cache(path: Path) -> Dict[tuple, dict]:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            out[(r["ticker"], r["date"])] = r
    return out


# ---- features for the models ----------------------------------------------------------
LLM_COLS = ["llm_has_guidance", "llm_eps_dir", "llm_eps_withdrawn", "llm_eps_change", "llm_rev_dir",
            "llm_rev_change", "llm_tone", "llm_headwinds", "llm_specificity"]
_DIR_SIGN = {"raised": 1.0, "lowered": -1.0}


def llm_features(path: Path = CACHE) -> pd.DataFrame:
    """Numeric features per (ticker, release date); releases never sent (no outlook) are all zeros."""
    rows = []
    for (ticker, date), r in load_cache(path).items():
        a = r.get("answer") or {}

        def pct(k):
            v = a.get(k)
            return float(np.clip(v, -20, 20)) if isinstance(v, (int, float)) else 0.0
        rows.append({
            "ticker": ticker, "date": date,
            "llm_has_guidance": float(a.get("specificity", 0) or 0) > 0,
            "llm_eps_dir": _DIR_SIGN.get(a.get("eps_guidance"), 0.0),
            "llm_eps_withdrawn": float(a.get("eps_guidance") == "withdrawn"),
            "llm_eps_change": pct("eps_change_pct"),
            "llm_rev_dir": _DIR_SIGN.get(a.get("revenue_guidance"), 0.0),
            "llm_rev_change": pct("revenue_change_pct"),
            "llm_tone": float(a.get("outlook_tone", 0) or 0),
            "llm_headwinds": float(bool(a.get("headwinds"))),
            "llm_specificity": float(a.get("specificity", 0) or 0),
        })
    return pd.DataFrame(rows, columns=["ticker", "date"] + LLM_COLS).astype({c: float for c in LLM_COLS}) \
        .set_index(["ticker", "date"])


def main() -> None:
    p = argparse.ArgumentParser(description="Score press-release outlook passages with an LLM (NVIDIA API)")
    p.add_argument("--releases-dir", default="data/raw/press_releases")
    p.add_argument("--cache", default=str(CACHE))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--limit", type=int, default=None, help="score at most N new releases (random sample)")
    p.add_argument("--rpm", type=float, default=35, help="requests per minute (free tier ~40)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--timeout", type=float, default=300,
                   help="seconds per call; the free tier queues requests, a 120 s cut-off wasted ~25%% of calls")
    p.add_argument("--show", action="store_true", help="print passage and answer for each scored release")
    p.add_argument("--retry-failed", action="store_true", help="re-score releases cached as failed")
    args = p.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=BASE_URL, api_key=load_api_key(), timeout=args.timeout, max_retries=0)

    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_cache(cache_path)
    if args.retry_failed:
        done = {k: r for k, r in done.items() if "error" not in (r.get("answer") or {})}
        cache_path.write_text("".join(json.dumps(r) + "\n" for r in done.values()))
    releases = [(r["company"], r["date"], r["text"]) for f in sorted(glob.glob(f"{args.releases_dir}/*.json"))
                for r in json.load(open(f))]
    todo = [x for x in releases if (x[0], x[1]) not in done]
    if args.limit:
        rng = np.random.default_rng(0)
        todo = [todo[i] for i in rng.choice(len(todo), min(args.limit, len(todo)), replace=False)]
    logger.info("%d releases, %d cached, scoring %d with %s", len(releases), len(done), len(todo), args.model)

    limiter, write_lock = RateLimiter(args.rpm), threading.Lock()
    counts = {"scored": 0, "no_outlook": 0, "failed": 0}

    def work(item):
        ticker, date, text = item
        passage = outlook_passage(text)
        answer = score(client, args.model, limiter, ticker, date, passage) if passage else None
        rec = {"ticker": ticker, "date": date, "model": args.model, "passage_chars": len(passage),
               "answer": answer}
        with write_lock:
            with open(cache_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            counts["no_outlook" if answer is None else "failed" if "error" in answer else "scored"] += 1
            n = sum(counts.values())
            if args.show:
                print(f"\n=== {ticker} {date} ({len(passage)} chars)\n{passage[:1500]}\n--> {json.dumps(answer)}")
            elif n % 50 == 0:
                logger.info("%d/%d %s", n, len(todo), counts)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, todo))
    logger.info("Done: %s → %s", counts, cache_path)


if __name__ == "__main__":
    main()
