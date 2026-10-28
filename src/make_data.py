"""How the data files in data/ were made. They are included, so nothing here needs to be run to repeat the
tests; this file documents them and can rebuild them.

  python src/make_data.py truthfulqa <generation-validation.parquet>    -> data/truthfulqa_200.jsonl
  python src/make_data.py mtbench <human-judgments.parquet>             -> data/mtbench_30.json
  python src/make_data.py material dossiers --cap 0.3                   -> data/dossiers.jsonl

truthfulqa_200  TruthfulQA (Lin et al., ACL 2022), generation config, validation split, Apache-2.0:
                huggingface.co/datasets/truthfulqa/truthful_qa, generation/validation-00000-of-00001.parquet,
                sha256 dfb1004b8ab83b22e8e476c76d5ac6074ff35c43946a724e810de3d83c3e21a5. 200 questions drawn with
                random.Random(0); every second one is paired with its best true answer, the others with one of its
                false answers drawn by the same generator.
mtbench_30      MT-Bench human judgments (Zheng et al., NeurIPS 2023), CC BY 4.0:
                huggingface.co/datasets/lmsys/mt_bench_human_judgments, human split
                (human-00000-of-00001-25f4910818759289.parquet), sha256
                4877bc46a40929f4082c3c79593700fb897b1d6c7f4c473032694a01322f5769. First turn only. Questions ranked
                by how much the human experts disagreed (answer pairs that got different winners from different
                experts, plus the share of tie votes); the 30 most disputed questions that have answers from all of
                gpt-4, claude-v1, gpt-3.5-turbo, vicuna-13b-v1.2 and alpaca-13b, with those answers and the experts'
                votes among them.
celebrity_parents  Berglund et al., "The Reversal Curse" (ICLR 2024), MIT: huggingface.co/datasets/lberglund/reversal_curse,
                celebrity_relations/parent_child_pairs.csv, copied unchanged, sha256
                59d181610f1477a3cfb8a63428bb8cc037b7d73c9cdf5616fdd8d8ba90debb5d.
summeval_20     SummEval (Fabbri et al., TACL 2021), MIT: huggingface.co/datasets/mteb/summeval,
                data/test-00000-of-00001-35901af5f6649399.parquet, sha256
                7ff6f3f4d044223bd6922c5434d27bf28473950346fcfc2a3b7352c19163690e. 20 articles drawn with
                random.Random(2026), and for each, 5 of its 16 machine summaries drawn with the same generator; their
                human coherence ratings are kept.
dossiers        written for these tests by GPT-6 Luna (gpt-6-luna) with the prompt below, one per topic in turn.
                Every document in them is invented ("fabricated": true in each record). Luna had written words such
                as "fictional" into some texts; every "fictional", "fictitious", "invented", "imaginary" or
                "made-up" followed by a space was then removed from the text fields shown to the judge
                (document_name, dossier).
"""
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, PRICES, cost  # noqa: E402

DATA = os.path.join(ROOT, "data")

TOPICS = ["early Buddhist monastic law", "Chinese Buddhist translation history", "Tang dynasty poetry",
          "medieval European chronicles", "Song dynasty printing", "Japanese temple records", "Byzantine hagiography",
          "Korean Buddhist epigraphy", "Islamic manuscript culture", "Renaissance letter collections",
          "Silk Road trade documents", "Daoist ritual texts", "Mughal court histories", "Tibetan monastic biographies",
          "Ottoman legal records", "Ming dynasty local gazetteers", "Heian court diaries", "Carolingian charters",
          "Pali commentarial literature", "Yuan dynasty drama"]

PROMPTS = {
    "dossiers": """Write material for a study of how readers assess historical documents, in the field of {topic}. Everything must be FICTIONAL.

Write a "dossier" of about 150 words describing a fictional document (what it is, where it was found, its script, contents and physical features) whose evidence for authenticity is genuinely MIXED: some features support authenticity and some raise doubts, roughly balanced. Do not state any conclusion and do not mention what anyone else thinks of it. Also give "document_name", a short name for it.
Reply with JSON only: {{"document_name": "...", "dossier": "..."}}"""
}
FIELDS = {"dossiers": ("document_name", "dossier")}
PREFIX = {"dossiers": "anchoring"}  # record ids as used in the results
CLEAN = re.compile(r"\b(?:fictional|fictitious|invented|imaginary|made-up)\s+", re.I)


def truthfulqa(parquet):
    import pandas as pd
    df = pd.read_parquet(parquet)
    rows = [{"question": x["question"], "best_answer": x["best_answer"], "incorrect": list(x["incorrect_answers"]),
             "category": x["category"]} for _, x in df.iterrows()]
    rng = random.Random(0)
    out = []
    for j, i in enumerate(rng.sample(range(len(rows)), 200)):
        r, truthful = rows[i], j % 2 == 0
        out.append({"id": f"tqa{i:03d}", "question": r["question"],
                    "answer": r["best_answer"] if truthful else rng.choice(r["incorrect"]),
                    "answer_is_truthful": truthful, "category": r["category"]})
    with open(os.path.join(DATA, "truthfulqa_200.jsonl"), "w", encoding="utf-8") as f:
        f.write("".join(json.dumps(o) + "\n" for o in out))


def mtbench(parquet):
    import pandas as pd
    t1 = pd.read_parquet(parquet)
    t1 = t1[t1.turn == 1]
    answers, votes = defaultdict(dict), defaultdict(list)
    for _, r in t1.iterrows():
        for side in ("a", "b"):
            conv = r[f"conversation_{side}"]
            answers[r.question_id][r[f"model_{side}"]] = (conv[0]["content"], conv[1]["content"])
        votes[r.question_id].append((r.model_a, r.model_b, r.winner))

    def disagreement(vs):
        by = defaultdict(list)
        for a, b, w in vs:
            by[tuple(sorted((a, b)))].append("tie" if w.startswith("tie") else (a if w == "model_a" else b))
        return sum(len(set(v)) > 1 for v in by.values()) + sum(w == "tie" for v in by.values() for w in v) / max(1, len(vs))
    models = ["gpt-4", "claude-v1", "gpt-3.5-turbo", "vicuna-13b-v1.2", "alpaca-13b"]
    out = {"questions": [], "source": "lmsys/mt_bench_human_judgments human split, turn 1, CC BY 4.0"}
    for q in sorted(answers, key=lambda q: -disagreement(votes[q])):
        if all(m in answers[q] for m in models):
            out["questions"].append({"question_id": int(q), "question": answers[q][models[0]][0],
                                     "answers": {m: answers[q][m][1] for m in models},
                                     "human_votes": [{"model_a": a, "model_b": b, "winner": w}
                                                     for a, b, w in votes[q] if a in models and b in models]})
        if len(out["questions"]) == 30:
            break
    json.dump(out, open(os.path.join(DATA, "mtbench_30.json"), "w"), ensure_ascii=False, indent=0)


def material(kind, cap):
    import openai
    from concurrent.futures import ThreadPoolExecutor
    client, model = openai.OpenAI(timeout=600), "gpt-6-luna"

    def one(n):
        topic = TOPICS[n % len(TOPICS)]
        r = client.chat.completions.create(model=model, max_completion_tokens=16000, response_format={"type": "json_object"},
                                           messages=[{"role": "user", "content": PROMPTS[kind].format(topic=topic)}]).model_dump()
        rec = {"id": f"{PREFIX[kind]}{n:02d}", "topic": topic, "generator": r["model"], "fabricated": True}
        rec.update({k: CLEAN.sub("", v) for k, v in json.loads(r["choices"][0]["message"]["content"]).items() if k in FIELDS[kind]})
        u = r["usage"]
        return rec, {"input_tokens": u["prompt_tokens"], "output_tokens": u["completion_tokens"], "cached_input_tokens": 0}
    spent = 0.0
    with ThreadPoolExecutor(8) as ex, open(os.path.join(DATA, f"{kind}.jsonl"), "w", encoding="utf-8") as f:
        for rec, usage in ex.map(one, range(40)):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            spent += cost(usage, PRICES[model])
            if spent > cap:
                sys.exit("cap reached")


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "truthfulqa":
        truthfulqa(sys.argv[2])
    elif what == "mtbench":
        mtbench(sys.argv[2])
    else:
        material(sys.argv[2], float(sys.argv[sys.argv.index("--cap") + 1]) if "--cap" in sys.argv else 0.3)
