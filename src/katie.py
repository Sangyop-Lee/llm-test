"""The Katie series: one NoLiMa-Hard question, asked at six input lengths with the fact always at 50% depth.

NoLiMa (Modarressi et al., ICML 2025; github.com/adobe-research/NoLiMa @ cb14780b249fecf2851127b2101a062c1b2c6430)
hides one sentence in a long novel and asks a question that shares no words with it. This script builds one of its
hard items exactly as the authors' evaluation code does, at the lengths of the series:

  needle    "The European Central Bank is next to where Katie lives."
  question  "Which character has been to Germany?"   (answer: Katie)

- Needle placement by the authors' BookHaystack (src/book_haystack.py, unchanged), lengths in o200k tokens.
- Character drawn as in the authors' async_evaluate.py: np.random.seed(42 + int(id[:4]) + haystack_no) with
  haystack_no = 0, one np.random.choice per needle depth over np.linspace(0, 100, 5); at 50% this gives Katie.
- System prompt "You are a helpful assistant", the needle set's task template, Chat Completions.
- Scored "contains": correct if the response contains "Katie".
- Haystacks: up to 32K the authors' short book, up to 280K their long book 1; beyond that, their long books joined
  end to end (1+2 up to 580K, 1-4 beyond), because one book holds only ~290K tokens. Run ./fetch_data.sh first.

  python src/katie.py build                     # build all six prompts, print their token counts and sha256
  python src/katie.py submit --repeats 3 --cap 25    # OpenAI Batch API, 3 calls per length
  python src/katie.py collect                   # append scored answers to results/<model>_katie.jsonl
  python src/katie.py direct --repeats 3 --cap 25    # or: standard API, full price, only the calls still missing
  python src/katie.py summary                   # the results table
  (add --model gpt-6-luna for GPT-6 Luna; the default is gpt-6.1-sol)

The API key is read from OPENAI_API_KEY by the openai package; it is never printed or logged.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from collections import defaultdict
from copy import copy

import numpy as np
import tiktoken

sys.path.insert(0, os.path.dirname(__file__))
from book_haystack import BookHaystack  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEEDLES = os.path.join(ROOT, "data", "needlesets", "needle_set_hard.json")
HAY = os.path.join(ROOT, "data", "haystack")
MODEL = "gpt-6.1-sol"  # set by --model
TEST = "0401Inv_T15_C02_twohop"
DEPTH = 50
LENGTHS = [1000, 32000, 270000, 500000, 750000, 900000]
SYSTEM_PROMPT = "You are a helpful assistant"
# USD per million tokens (OpenAI pricing, October 2026). Above 272K input tokens the whole request costs 2x input
# and 1.5x output; the Batch API halves the price.
PRICES = {"gpt-6.1-sol": {"input": 2.00, "cached_input": 0.10, "output": 10.00},
          "gpt-6-luna": {"input": 0.10, "cached_input": 0.01, "output": 0.50}}
LONG = 272_000


def book_for(length):
    if length <= 32000:
        return os.path.join(HAY, "rand_shuffle", "rand_book_1.txt")
    if length <= 280000:
        return os.path.join(HAY, "rand_shuffle_long", "rand_book_1.txt")
    return os.path.join(HAY, "joined_long", "rand_book_1+2.txt" if length <= 580000 else "rand_book_1-4.txt")


def the_test():
    """The authors' test construction (evaluation/run_tests.py), for the one item used here."""
    for exp in json.load(open(NEEDLES, encoding="utf-8")):
        for qtype, question in exp["questions"].items():
            for test_id, test in exp["tests"].items():
                if f"{exp['id']}_{test_id}_{qtype}" != TEST:
                    continue
                needle, full_question = exp["needle"], copy(question)
                for n, arg in enumerate(test["input_args"]):
                    ph = "{" + str(n + 1) + "}"
                    if ph in question:
                        full_question = question.replace(ph, arg)
                    if ph in needle:
                        needle = needle.replace(ph, arg)
                return {"needle": needle, "question": full_question, "character_set": exp["character_set"],
                        "task_template": exp["task_template"], "seed": 42 + int(exp["id"][:4])}
    raise KeyError(TEST)


def build(length, enc=tiktoken.get_encoding("o200k_base")):
    t = the_test()
    np.random.seed(t["seed"])
    for depth in np.linspace(0, 100, 5):
        char = str(np.random.choice(t["character_set"]))  # one draw per depth, as in the authors' code
        if int(round(depth)) == DEPTH:
            break
    placement = BookHaystack(book_for(length)).generate_w_needle_placement(
        needle=t["needle"].replace("{CHAR}", char), token_count_func=lambda s: len(enc.encode(s)),
        encoding_func=enc.encode, decoding_func=enc.decode, context_length=length, depth=DEPTH / 100, shift=0,
        static_depth=-1, distractor=None)
    prompt = t["task_template"].format(haystack=placement["text"], question=t["question"])
    return {"length": length, "character": char, "question": t["question"], "haystack": os.path.relpath(book_for(length), ROOT),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "prompt": prompt}


def results_path():
    return os.path.join(ROOT, "results", f"{MODEL}_katie.jsonl")


def state_path():
    return os.path.join(ROOT, "results", f"{MODEL}_batch_state.json")


def cost(u, batch):
    PRICE = PRICES[MODEL]
    m_in, m_out = (2.0, 1.5) if u["input_tokens"] > LONG else (1.0, 1.0)
    c = ((u["input_tokens"] - u["cached_input_tokens"]) * PRICE["input"] * m_in
         + u["cached_input_tokens"] * PRICE["cached_input"] * m_in + u["output_tokens"] * PRICE["output"] * m_out) / 1e6
    return c * (0.5 if batch else 1)


def client():
    import openai
    return openai.OpenAI(timeout=600)


def submit(lengths, repeats, cap):
    """Send enough calls to bring every length to `repeats` answered calls in this model's results file."""
    have = defaultdict(int)
    if os.path.exists(results_path()):
        for line in open(results_path(), encoding="utf-8"):
            have[json.loads(line)["length"]] += 1
    built = [build(n) for n in lengths]
    todo = {b["length"]: max(0, repeats - have[b["length"]]) for b in built}
    est = sum(b["length"] * PRICES[MODEL]["input"] * (2 if b["length"] > LONG else 1) * todo[b["length"]] for b in built) / 1e6 * 0.5
    print("new calls per length:", todo)
    print(f"{sum(todo.values())} requests; estimated input cost ${est:.2f} (cap ${cap:.2f})")
    if est > cap:
        sys.exit("estimate exceeds cap; nothing sent")
    req_path = os.path.join(ROOT, "results", "batch_requests.jsonl")
    meta = {}
    with open(req_path, "w", encoding="utf-8") as f:
        for b in built:
            for k in range(have[b["length"]], repeats):
                cid = f"L{b['length']}_r{k + 1}"
                meta[cid] = {x: v for x, v in b.items() if x != "prompt"}
                f.write(json.dumps({"custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
                                    "body": {"model": MODEL, "max_completion_tokens": 16000,
                                             "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                                          {"role": "user", "content": b["prompt"]}]}}) + "\n")
    c = client()
    up = c.files.create(file=open(req_path, "rb"), purpose="batch")
    batch = c.batches.create(input_file_id=up.id, endpoint="/v1/chat/completions", completion_window="24h")
    json.dump({"batch_id": batch.id, "submitted_utc": dt.datetime.now(dt.UTC).isoformat(), "meta": meta},
              open(state_path(), "w"), indent=1)
    os.remove(req_path)  # holds the haystack text, which is not redistributed
    print("batch", batch.id, batch.status)


def collect():
    """Append the batch's answers. A cancelled batch keeps whatever it finished; `direct` then sends the rest."""
    st = json.load(open(state_path()))
    c = client()
    b = c.batches.retrieve(st["batch_id"])
    if b.status not in ("completed", "cancelled") or not b.output_file_id:
        sys.exit(f"batch is {b.status}: {b.request_counts}")
    with open(results_path(), "a", encoding="utf-8") as f:
        for line in c.files.content(b.output_file_id).text.splitlines():
            r = json.loads(line)
            body = r["response"]["body"]
            u = body["usage"]
            usage = {"input_tokens": u["prompt_tokens"], "output_tokens": u["completion_tokens"],
                     "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
                     "cached_input_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)}
            ans = body["choices"][0]["message"]["content"] or ""
            m = st["meta"][r["custom_id"]]
            f.write(json.dumps({**m, "depth_percent": DEPTH, "requested_model": MODEL, "returned_model": body["model"],
                                "api": "batch", "batch_id": b.id, "request_utc": st["submitted_utc"], "response": ans,
                                "correct": int(m["character"] in ans), "usage": usage, "cost_usd": round(cost(usage, True), 4)},
                               ensure_ascii=False) + "\n")
    print("appended to", results_path())


def direct(lengths, repeats, cap):
    """Standard API, one call at a time: bring every length to `repeats` answered calls (full price)."""
    have = defaultdict(int)
    if os.path.exists(results_path()):
        for line in open(results_path(), encoding="utf-8"):
            have[json.loads(line)["length"]] += 1
    todo = {n: max(0, repeats - have[n]) for n in lengths}
    est = sum(n * PRICES[MODEL]["input"] * (2 if n > LONG else 1) * k for n, k in todo.items()) / 1e6
    print("new calls per length:", todo, f"estimated input cost ${est:.2f} (cap ${cap:.2f})")
    if est > cap:
        sys.exit("estimate exceeds cap; nothing sent")
    c, spent = client(), 0.0
    for n, k in todo.items():
        if not k:
            continue
        b = build(n)
        for _ in range(k):
            when = dt.datetime.now(dt.UTC).isoformat()
            r = c.chat.completions.create(model=MODEL, max_completion_tokens=16000,
                                          messages=[{"role": "system", "content": SYSTEM_PROMPT},
                                                    {"role": "user", "content": b["prompt"]}]).model_dump()
            u = r["usage"]
            usage = {"input_tokens": u["prompt_tokens"], "output_tokens": u["completion_tokens"],
                     "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
                     "cached_input_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0}
            ans = r["choices"][0]["message"]["content"] or ""
            rec = {**{x: v for x, v in b.items() if x != "prompt"}, "depth_percent": DEPTH, "requested_model": MODEL,
                   "returned_model": r["model"], "api": "standard", "batch_id": None, "request_utc": when,
                   "response": ans, "correct": int(b["character"] in ans), "usage": usage,
                   "cost_usd": round(cost(usage, False), 4)}
            with open(results_path(), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            spent += rec["cost_usd"]
            print(n, repr(ans), f"${spent:.4f}", flush=True)
            if spent > cap:
                sys.exit("cap reached")


def summary():
    rows = [json.loads(l) for l in open(results_path(), encoding="utf-8")]
    by = defaultdict(list)
    for x in rows:
        by[x["length"]].append(x)
    print(f"{'length':>8}  correct  answers")
    for n in sorted(by):
        xs = by[n]
        print(f"{n:>8}  {sum(x['correct'] for x in xs)}/{len(xs)}      " + " | ".join(x["response"].strip() for x in xs))
    print(f"cost of these calls: ${sum(x['cost_usd'] for x in rows):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["build", "submit", "collect", "direct", "summary"])
    ap.add_argument("--lengths", default=",".join(map(str, LENGTHS)))
    ap.add_argument("--model", default="gpt-6.1-sol", choices=["gpt-6.1-sol", "gpt-6-luna"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cap", type=float, default=25.0, help="USD; submit refuses if the estimate is higher")
    a = ap.parse_args()
    MODEL = a.model
    lengths = [int(x) for x in a.lengths.split(",")]
    if a.action == "build":
        for n in lengths:
            b = build(n)
            print(n, b["character"], b["haystack"], b["prompt_sha256"])
    elif a.action == "submit":
        submit(lengths, a.repeats, a.cap)
    elif a.action == "collect":
        collect()
    elif a.action == "direct":
        direct(lengths, a.repeats, a.cap)
    else:
        summary()
