"""Shared pieces: the API runner (with a spending cap), answer parsers and statistics.

Every test is a list of calls. A call holds either `prompt` (one user message) or `messages` (a full conversation),
plus metadata used for scoring. Identical requests are sent once: a request is identified by the SHA-256 of its
text, plus a suffix for repeated trials (`_r<n>`), for the identical rerun of a CALM prompt (`_rerun`) and for a set
reasoning effort (`_e<effort>`). Answers are appended to a log as they arrive, so an interrupted run resumes.
"""
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "raw")
# USD per million tokens, standard tier (OpenAI pricing, October 2026).
PRICES = {"gpt-6.1-sol": {"input": 2.00, "cached_input": 0.10, "output": 10.00},
          "gpt-6-luna": {"input": 0.10, "cached_input": 0.01, "output": 0.50}}


def messages_of(c):
    return c["messages"] if "messages" in c else [{"role": "user", "content": c["prompt"]}]


def request_key(c, effort=None):
    text = c["prompt"] if "messages" not in c else json.dumps(c["messages"], ensure_ascii=False, sort_keys=True)
    key = hashlib.sha256(text.encode()).hexdigest()[:20] + ("_rerun" if c["variant"] == "orig_rerun" else "")
    key += f"_r{c['replicate']}" if "replicate" in c else ""
    return key + (f"_e{effort}" if effort else "")


def cost(u, price):
    cached = u.get("cached_input_tokens", 0)
    return ((u["input_tokens"] - cached) * price["input"] + cached * price["cached_input"]
            + u["output_tokens"] * price["output"]) / 1e6


def estimate(reqs, price, out_tokens):
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    tin = sum(len(enc.encode(m["content"])) + 4 for msgs in reqs.values() for m in msgs)
    return (tin * price["input"] + len(reqs) * out_tokens * price["output"]) / 1e6


def run(calls, model, tag, effort=None, cap=1.0, out_tokens=150, workers=8, reuse=()):
    """Send every distinct request not yet answered; stop before the estimate or the spend passes `cap` (USD).
    Writes results/raw/<model>_<tag>.jsonl: one record per call, its metadata and answer, without the prompt text."""
    import openai
    price = PRICES[model]
    log = os.path.join(RAW, f"{model}_{tag}_answers.jsonl")
    answers, own = {}, set()
    for path in [os.path.join(RAW, f) for f in reuse] + [log]:
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                r = json.loads(line)
                if r["error"] is None:
                    answers[r["key"]] = r
                    own.add(r["key"]) if path == log else None
    reqs = {}
    for c in calls:
        k = request_key(c, effort)
        if k not in answers:
            reqs.setdefault(k, messages_of(c))
    spent = sum(cost(answers[k]["usage"], price) for k in own)
    est = estimate(reqs, price, out_tokens)
    print(f"{len(reqs)} requests to send; ${spent:.4f} spent; estimated ${est:.2f} more (cap ${cap:.2f})", flush=True)
    if spent + est > cap:
        sys.exit("estimate exceeds cap; nothing sent")
    client = openai.OpenAI(timeout=600, max_retries=2)  # reads OPENAI_API_KEY; the key is never printed

    def one(k, msgs):
        body = {"model": model, "max_completion_tokens": 16000, "messages": msgs}
        if effort:
            body["reasoning_effort"] = effort
        try:
            r = client.chat.completions.create(**body).model_dump()
            u = r["usage"]
            usage = {"input_tokens": u["prompt_tokens"], "output_tokens": u["completion_tokens"],
                     "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0,
                     "cached_input_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0}
            return {"key": k, "response": r["choices"][0]["message"]["content"], "usage": usage,
                    "returned_model": r["model"], "finish_reason": r["choices"][0]["finish_reason"], "error": None}
        except Exception as e:
            return {"key": k, "response": None, "error": f"{type(e).__name__}: {e}"[:500]}

    stop = False
    os.makedirs(RAW, exist_ok=True)
    with ThreadPoolExecutor(workers) as ex, open(log, "a", encoding="utf-8") as f:
        futs = [ex.submit(one, k, m) for k, m in reqs.items()]
        for fut in as_completed(futs):
            if fut.cancelled():
                continue
            r = fut.result()
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            if r["error"] is None:
                answers[r["key"]] = r
                spent += cost(r["usage"], price)
            if not stop and (spent > cap or (r["error"] and ("insufficient_quota" in r["error"] or "billing" in r["error"]))):
                print("cap or billing limit reached; cancelling the remaining requests", flush=True)
                stop = True
                for x in futs:
                    x.cancel()
    write_records(calls, answers, model, tag, effort, price)
    print(f"{sum(request_key(c, effort) in answers for c in calls)}/{len(calls)} calls answered; ${spent:.4f} spent")


def write_records(calls, answers, model, tag, effort, price):
    billed = set()
    with open(os.path.join(RAW, f"{model}_{tag}.jsonl"), "w", encoding="utf-8") as f:
        for c in calls:
            k = request_key(c, effort)
            a = {x: v for x, v in answers.get(k, {"response": None, "error": "no response"}).items() if x != "key"}
            rec = {x: v for x, v in c.items() if x not in ("prompt", "messages")}
            rec.update(request_key=k, requested_model=model, reasoning_effort=effort or "default", **a)
            if a.get("usage") and k not in billed:
                rec["cost_usd"] = round(cost(a["usage"], price), 6)
                billed.add(k)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load(model, tag):
    p = os.path.join(RAW, f"{model}_{tag}.jsonl")
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


# --- parsers ---------------------------------------------------------------------------------------------------

VERDICT = re.compile(r"\[\[\s*([ABC])\s*\]\]")


def verdict(text):
    """Last [[A]], [[B]] or [[C]] in a reply (the MT-Bench / CALM pairwise format)."""
    found = VERDICT.findall(text or "")
    return found[-1] if found else None


def bracket_rating(text):
    """Last [[n]] in a reply (the MT-Bench single-rating format)."""
    m = re.findall(r"\[\[\s*(\d+(?:\.\d+)?)\s*\]\]", text or "")
    return float(m[-1]) if m else None


def leading_number(text, lo=1, hi=10):
    """A reply that starts with a number in [lo, hi] ("7", "3/10 - provisional"); anything else is None."""
    m = re.match(r"^\s*\**\s*(\d+(?:\.\d+)?)\s*(?:/\s*10)?", text or "")
    return float(m.group(1)) if m and lo <= float(m.group(1)) <= hi else None


def yes_no(text):
    m = re.findall(r"final answer\s*[:：]\s*\**\s*(yes|no)\b", text or "", re.I)
    return m[-1].lower() if m else None


def letter(text, options=("A", "B", "C")):
    """A reply that is a bare option letter, or starts with one followed by punctuation ("A.", "(A)", "A:")."""
    t = (text or "").strip().strip("*").strip()
    if t in options:
        return t
    m = re.match(r"^(?:\(([A-C])\)|([A-C])[.):])(?:\s|$)", t)
    v = m and (m.group(1) or m.group(2))
    return v if v in options else None


# --- statistics ------------------------------------------------------------------------------------------------

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def rnd(x, k=3):
    return None if x is None else round(x, k)


def sign_test(k, n):
    """Two-sided exact binomial test of k successes in n trials at p = 0.5."""
    if n == 0:
        return float("nan")
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n)


def fisher(a, b, c, d):
    """Two-sided Fisher exact test for the 2x2 table [[a, b], [c, d]]."""
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d

    def p(x):
        return math.comb(r1, x) * math.comb(r2, c1 - x) / math.comb(n, c1)
    p0 = p(a)
    return min(1.0, sum(p(x) for x in range(max(0, c1 - r2), min(r1, c1) + 1) if p(x) <= p0 * (1 + 1e-9)))


def perm_paired(diffs, n=20000, seed=0):
    """Two-sided sign-flip permutation test that the mean of paired differences is 0."""
    diffs = [d for d in diffs if d is not None]
    rng = random.Random(seed)
    obs = abs(sum(diffs) / len(diffs))
    hits = sum(abs(sum(d if rng.random() < 0.5 else -d for d in diffs) / len(diffs)) >= obs - 1e-12 for _ in range(n))
    return (hits + 1) / (n + 1)


def perm_two(a, b, n=20000, seed=0):
    """Two-sided permutation test of a difference in means between two independent samples."""
    rng = random.Random(seed)
    obs = abs(sum(a) / len(a) - sum(b) / len(b))
    pool, k, hits = a + b, len(a), 0
    for _ in range(n):
        rng.shuffle(pool)
        hits += abs(sum(pool[:k]) / k - sum(pool[k:]) / len(pool[k:])) >= obs - 1e-12
    return (hits + 1) / (n + 1)


def by_key(records, *keys):
    out = defaultdict(list)
    for r in records:
        out[tuple(r[k] for k in keys)].append(r)
    return out
