"""How the data files in data/ were made. They are included, so nothing here needs to be run to repeat the
tests; this file documents them and can rebuild them.

  python src/make_data.py truthfulqa <generation-validation.parquet>    -> data/truthfulqa_200.jsonl
  python src/make_data.py mtbench <human-judgments.parquet>             -> data/mtbench_80.json

truthfulqa_200  TruthfulQA (Lin et al., ACL 2022), generation config, validation split, Apache-2.0:
                huggingface.co/datasets/truthfulqa/truthful_qa, generation/validation-00000-of-00001.parquet,
                sha256 dfb1004b8ab83b22e8e476c76d5ac6074ff35c43946a724e810de3d83c3e21a5. 200 questions drawn with
                random.Random(0); every second one is paired with its best true answer, the others with one of its
                false answers drawn by the same generator.
mtbench_80      MT-Bench human judgments (Zheng et al., NeurIPS 2023), CC BY 4.0:
                huggingface.co/datasets/lmsys/mt_bench_human_judgments, human split
                (human-00000-of-00001-25f4910818759289.parquet), sha256
                4877bc46a40929f4082c3c79593700fb897b1d6c7f4c473032694a01322f5769. First turn only. All 80 questions, each with the
                answers of gpt-4, claude-v1, gpt-3.5-turbo, vicuna-13b-v1.2 and alpaca-13b and the experts' votes
                among them, ordered by how much the experts disagreed (answer pairs that got different winners from
                different experts, plus the share of tie votes).
celebrity_parents  Berglund et al., "The Reversal Curse" (ICLR 2024), MIT: huggingface.co/datasets/lberglund/reversal_curse,
                celebrity_relations/parent_child_pairs.csv, copied unchanged, sha256
                59d181610f1477a3cfb8a63428bb8cc037b7d73c9cdf5616fdd8d8ba90debb5d.
summeval_20     SummEval (Fabbri et al., TACL 2021), MIT: huggingface.co/datasets/mteb/summeval,
                data/test-00000-of-00001-35901af5f6649399.parquet, sha256
                7ff6f3f4d044223bd6922c5434d27bf28473950346fcfc2a3b7352c19163690e. 20 articles drawn with
                random.Random(2026), and for each, 5 of its 16 machine summaries drawn with the same generator; their
                human coherence ratings are kept.
"""
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT  # noqa: E402

DATA = os.path.join(ROOT, "data")

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
    json.dump(out, open(os.path.join(DATA, "mtbench_80.json"), "w"), ensure_ascii=False, indent=0)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "truthfulqa":
        truthfulqa(sys.argv[2])
    elif what == "mtbench":
        mtbench(sys.argv[2])
