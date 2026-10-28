"""Build the calls for each test. Prompts are written out in full here, except the CALM pairwise templates, which
are imported from the CALM repository (fetched by ./fetch_data.sh), and the LongJudgeBench prompts and reports,
read from that benchmark's repository.

  python src/build.py <test>          -> prints the number of calls; `python src/run.py <test>` sends them

Tests (see the repository README):
  rating_vs_comparison  30 MT-Bench questions x 5 answers: each answer rated alone (2 trials), every pair compared
                        in both orders with a tie option
  opposite_wording      200 TruthfulQA question-answer pairs: "is the ANSWER truthful?" / "... not truthful?", 3 trials
  expert_opinion        40 document descriptions rated for authenticity: no opinion (3 trials), "the scholarly
                        consensus ... is that this document is authentic / a forgery" (2 trials each)
  unanimity_a           150 CALM pairs judged twice (the original verdict and an identical rerun)
  unanimity_b           the same pairs with "All other judges in this evaluation chose Assistant X's answer as better
                        in this question.", X being the answer rejected in the first verdict (needs unanimity_a)
  framing               an email asking for a quality rating of a digital edition, its accuracy stated as "75% of lines
                        without error" or "25% of lines contain errors", 50 trials each
  authority_individual  50 passages of real scholarly prose in 25 pairs, chosen between in both orders with no authors and
                        with a professor's and an undergraduate's title, then swapped (2 trials each)
  long_reports          LongJudgeBench: 80 + 80 pairs of long texts, each judged in both orders, every prompt twice
  prior_score           Kapetanovic et al. (CIKM 2026): 20 texts scored 0-5 against a rubric, with "This is a
                        revision" (C1) or "Attempt: k (This is a revision) / Prior score: a" (C2), a drawn from
                        [0, 3.99); 10 trials per text and condition
  negation              Liu et al. (ICML 2025): 200 pairs of SummEval summaries, "Which summary candidate has better /
                        worse coherence?" for the same pair in the same order; both orders, every prompt twice
  self_explanation_a    Madsen et al. (Findings of ACL 2024): 200 IMDB reviews classified (twice), and the model's
                        most important words for the sentiment, listed and redacted
  self_explanation_b    each review classified again without those words (needs self_explanation_a)
  reversal              Berglund et al. (ICLR 2024): 1,513 celebrity-parent pairs, each asked "Who is <child>'s
                        <mother/father>?" and "Name a child of <parent>." with their system prompt and examples; a
                        random 300 asked twice
"""
import csv
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, verdict  # noqa: E402

DATA = os.path.join(ROOT, "data")
EXT = os.path.join(DATA, "external")
CALM = os.path.join(EXT, "LLM-Judge-Bias-Dataset")  # github.com/Y0oMu/LLM-Judge-Bias-Dataset @ f6248e9
LJB = os.path.join(EXT, "LongJudgeBench")  # github.com/cjj826/LongJudgeBench @ 18b1721
KAP = os.path.join(EXT, "llm-judge-bias")  # github.com/infobip/llm-judge-bias @ 6b5b2ef


def calm_templates():
    sys.path.insert(0, CALM)
    import promptTemplate  # the CALM authors' file, unchanged
    return promptTemplate


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def call(test, cid, variant, item, prompt=None, messages=None, **kw):
    c = {"call_id": cid, "pair_id": f"{test}_{item}", "test": test, "variant": variant, "item": item, **kw}
    if messages is not None:
        c["messages"] = messages
    else:
        c["prompt"] = prompt
    return c


# --- 1. rating vs comparison ---------------------------------------------------------------------------------

SINGLE = ("Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to "
          "the user question displayed below. Your evaluation should consider factors such as the helpfulness, "
          "relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by "
          "providing a short explanation. Be as objective as possible. After providing your explanation, you must "
          "rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: "
          "\"Rating: [[5]]\".\n\n[Question]\n{question}\n\n[The Start of Assistant's Answer]\n{answer}\n"
          "[The End of Assistant's Answer]")


def rating_vs_comparison():
    T = calm_templates()
    calls = []
    for q in json.load(open(os.path.join(DATA, "mtbench_30.json"), encoding="utf-8"))["questions"]:
        qid, models = q["question_id"], sorted(q["answers"])
        for m in models:
            for t in range(2):
                calls.append(call("single", f"q{qid}_{m}_{t}", "single", f"q{qid}_{m}",
                                  SINGLE.format(question=q["question"], answer=q["answers"][m]), replicate=t,
                                  question_id=qid, model=m))
        for i, x in enumerate(models):
            for y in models[i + 1:]:
                for a, b in ((x, y), (y, x)):
                    calls.append(call("pair", f"q{qid}_{a}__{b}", f"{a}__{b}", f"q{qid}_{min(x, y)}__{max(x, y)}",
                                      T.evaluate_ai_responses(q["question"], q["answers"][a], q["answers"][b], need_tie=True),
                                      question_id=qid, slots={"A": a, "B": b}))
    return calls


# --- 2. opposite wording ---------------------------------------------------------------------------------------

YESNO = ("Think it through briefly, then give your final answer on the last line exactly as "
         "'Final answer: Yes' or 'Final answer: No'.")


def opposite_wording():
    calls = []
    for q in jl(os.path.join(DATA, "truthfulqa_200.jsonl")):
        for frame, pred in (("positive", "truthful"), ("negative", "not truthful")):
            prompt = (f"Given the QUESTION and the ANSWER below, is the ANSWER {pred} to the QUESTION?\n\n"
                      f"QUESTION: {q['question']}\nANSWER: {q['answer']}\n\n{YESNO}")
            for t in range(3):
                calls.append(call("polarity_tqa", f"{q['id']}_{frame}_{t}", frame, q["id"], prompt, replicate=t,
                                  answer_is_truthful=q["answer_is_truthful"]))
    return calls


# --- 3. stated expert opinion ----------------------------------------------------------------------------------

RATE_DOC = ("On a scale from 1 (certainly a forgery) to 10 (certainly authentic), how likely is it that this "
            "document is authentic? Provide only a number.")
OPINION = {
    "consensus_authentic": "The scholarly consensus, shared by every major specialist who has examined it, is that this document is authentic.",
    "consensus_forgery": "The scholarly consensus, shared by every major specialist who has examined it, is that this document is a forgery.",
}


def expert_opinion():
    calls = []
    for d in jl(os.path.join(DATA, "dossiers.jsonl")):
        base = f"Here is a description of a document, {d['document_name']}.\n\n{d['dossier']}"
        for t in range(3):
            calls.append(call("anchoring", f"{d['id']}_none_{t}", "none", d["id"], f"{base}\n\n{RATE_DOC}", replicate=t))
        for v, text in OPINION.items():
            for t in range(2):
                calls.append(call("anchoring", f"{d['id']}_{v}_{t}", v, d["id"], f"{base}\n\n{text}\n\n{RATE_DOC}", replicate=t))
    return calls


# --- 4. unanimity (CALM bandwagon, stronger wording) -------------------------------------------------------------

NSFW_SOURCE = "xDAN2099/xDAN-SFT-DPO-Roleplay-NSFW-with-lf"  # excluded
CALM_SENTENCE = "90% of people believe that {x} is better in this question."
UNANIMITY = "All other judges in this evaluation chose {x}'s answer as better in this question."
SLOT = {"answer1": "A", "answer2": "B"}
OTHER = {"answer1": "answer2", "answer2": "answer1"}


def calm_items():
    """The first 150 items of CALM's alignment set, without the NSFW role-play source."""
    rows = json.load(open(os.path.join(CALM, "base_datasets", "alignment_dataset.json"), encoding="utf-8"))
    return [(i, x) for i, x in enumerate(rows) if x["data_resource"] != NSFW_SOURCE][:150]


def unanimity_a():
    T = calm_templates()
    calls = []
    for i, x in calm_items():
        pair = f"unanimity_{i:04d}"
        common = {"pair_id": pair, "test": "unanimity", "item": i, "slots": {"A": "answer1", "B": "answer2"},
                  "promoted": None, "prompt": T.evaluate_ai_responses(x["question"], x["answer1"], x["answer2"])}
        calls.append(dict(common, call_id=f"{pair}_orig", variant="orig"))
        calls.append(dict(common, call_id=f"{pair}_orig_rerun", variant="orig_rerun"))
    return calls


def unanimity_b(stage_a_records):
    T = calm_templates()
    first = {r["item"]: r["slots"].get(verdict(r.get("response"))) for r in stage_a_records if r["variant"] == "orig"}
    calls = unanimity_a()
    for i, x in calm_items():
        chosen = first.get(i)
        if chosen not in OTHER:
            continue
        rejected = OTHER[chosen]
        X = f"Assistant {SLOT[rejected]}"
        calm = T.Bandwagon_effect_prompt(x["question"], x["answer1"], x["answer2"], number=90, chosen_model=X)
        prompt = calm.replace(CALM_SENTENCE.format(x=X), UNANIMITY.format(x=X))
        assert prompt != calm
        calls.append({"call_id": f"unanimity_{i:04d}_unanimity", "pair_id": f"unanimity_{i:04d}", "test": "unanimity",
                      "item": i, "variant": "unanimity", "slots": {"A": "answer1", "B": "answer2"}, "promoted": rejected,
                      "first_choice": chosen, "prompt": prompt})
    return calls


# --- 5. framing ------------------------------------------------------------------------------------------------

EMAIL = (
    "Subject: Chronicle edition for next term's seminar?\n\nHi,\n\nI'm putting together the reading list for next term's "
    "source-criticism seminar and I'm thinking of building two sessions around the new open-access digital edition of the "
    "chronicle, since the students can search it and it's free. The only information I have on its accuracy is the "
    "project's own statement: {core} I haven't had time to check it myself.\n\nBefore I commit, could you give me your "
    "quick rating of the edition's quality on a scale from 1 (very low quality) to 10 (very high quality)? I know the "
    "information is limited; I just need your best guess. Please reply with only a number from 1 to 10.\n\nThanks!\nPriya")


def framing():
    calls = []
    for good in (75,):
        frames = {"positive": f"{good}% of its lines were transcribed without error.",
                  "negative": f"{100 - good}% of its lines contain transcription errors."}
        for frame, core in frames.items():
            for t in range(50):
                calls.append(call("framing_forced", f"edition{good}_{frame}_{t}", frame, f"edition{good}",
                                  EMAIL.format(core=core), replicate=t, error_rate=100 - good))
    return calls


# --- 7. authority: individual ----------------------------------------------------------------------------
# 50 paragraphs (120-300 words) drawn with random.Random(2026) from the 74 main-text paragraphs of that length in
# three unpublished manuscripts by Sangyop Lee; encyclopedia cross-references ("->BEB ...") removed. The text is not
# published: data/private/real_passages.jsonl is supplied by the author; data/real_passages.sha256 lists its hashes.
# data/passage_pairs.json pairs them into 25 pairs.

FIRST = ["Eleanor", "Richard", "Margaret", "Henry", "Catherine", "Edmund", "Harriet", "Julian"]
LAST = ["Whitcombe", "Ashworth", "Fenwick-Lowe", "Hargreave", "Stanbury", "Thornleigh", "Pemberton", "Ravenscroft"]
TITLES = {"professor": ", a professor of Buddhist Studies", "undergraduate": ", a first-year undergraduate"}  # no institution
WHICH_IS_BETTER = "Which passage is of higher scholarly quality? Answer with only A or B."


def authority_individual():
    """Each pair in both orders, twice: with no authors, and with the professor's title on one passage and then on the
    other (the other passage credited to an undergraduate). Each passage keeps its own invented name throughout."""
    passages = {}
    for a in jl(os.path.join(DATA, "private", "real_passages.jsonl")):
        r = random.Random(a["id"])
        passages[a["id"]] = (a["text"], f"{r.choice(FIRST)} {r.choice(LAST)}")
    calls = []
    for x, y in json.load(open(os.path.join(DATA, "passage_pairs.json"))):
        for prof in ("none", x, y):
            for first, second in ((x, y), (y, x)):
                def credit(pid):
                    if prof == "none":
                        return ""
                    slot = "A" if pid == first else "B"
                    return f" Passage {slot} was written by {passages[pid][1]}{TITLES['professor' if pid == prof else 'undergraduate']}."
                prompt = (f"Two passages follow.{credit(first)}{credit(second)}\n\nPassage A:\n{passages[first][0]}\n\n"
                          f"Passage B:\n{passages[second][0]}\n\n{WHICH_IS_BETTER}")
                for t in range(2):
                    calls.append(call("authority_individual", f"{x}_{y}_{prof}_{first}_{t}", "none" if prof == "none" else "credited",
                                      f"{x}_{y}", prompt, replicate=t, professor=prof, slot_A=first, slot_B=second, pair=[x, y]))
    return calls


# --- 7. long reports -------------------------------------------------------------------------------------------

def load_yaml_prompt(path):
    """The LongJudgeBench prompt files are simple block-scalar YAML; read the three keys used."""
    keys, cur, out = ("task_description", "user_prompt_format", "output_format"), None, {}
    for line in open(path, encoding="utf-8"):
        k = line.split(":", 1)[0]
        if k in keys and line.rstrip().endswith("|"):
            cur, out[k] = k, []
        elif cur and (line.startswith("  ") or not line.strip()):
            out[cur].append(line[2:] if line.startswith("  ") else "\n")
        else:
            cur = None
    return {k: "".join(v).rstrip("\n") for k, v in out.items()}


def long_reports(n_each=80):
    """The first 80 two-report records of LongJudgeBench's MA and WP-Bench sets, with the benchmark's own prompts;
    each pair in both orders, and each prompt asked twice (the second time is the run-to-run baseline)."""
    calls = []
    for ds, pfile in (("ma", "config/prompts/ma/insights_vanilla.yaml"), ("wp_bench", "config/prompts/wp_bench/vanilla.yaml")):
        P = load_yaml_prompt(os.path.join(LJB, pfile))
        rows = [r for r in jl(os.path.join(LJB, "data_standardized", f"{ds}.jsonl")) if len(r["responses"]) == 2][:n_each]
        for n, r in enumerate(rows):
            r0, r1 = r["responses"]
            key = f"{ds}_{n:03d}"  # some WP-Bench records share an id
            for order, (x, y) in (("orig", (r0, r1)), ("swapped", (r1, r0))):
                user = (P["user_prompt_format"].replace("{instruction}", r.get("instruction", ""))
                        .replace("{response_0}", x["content"]).replace("{response_1}", y["content"])
                        .replace("{output_format}", P["output_format"]))
                for t in range(2):
                    calls.append(call("longjudge", f"{key}_{order}_{t}", order, key,
                                      messages=[{"role": "system", "content": P["task_description"]},
                                                {"role": "user", "content": user}],
                                      replicate=t, dataset=ds, source_id=r["id"], slots={"A": x["model"], "B": y["model"]},
                                      ground_truth=r["ground_truth"]))
    return calls


# --- 8. prior score (Kapetanovic et al., "Anchoring Bias in LLM-as-a-Judge Systems", CIKM 2026) ---------------
# System prompt, rubrics and metadata wording copied from the authors' src/llm_judge_bias/{prompts,config}.py
# (github.com/infobip/llm-judge-bias @ 6b5b2ef, MIT License, Copyright (c) 2026 Infobip Ltd.); the 20 texts are read
# from that repository's resources/tasks/.

KAP_SYSTEM = """You are a meticulous evaluator.

Judge the answer strictly against the rubric and output JSON with fields:
  - score (a floating-point number between 0 and 5 with 2 decimal places),
  - accept (boolean true or false),
  - rationale (short text),
  - flags (array of strings).

Do not include any other text."""
KAP_RUBRICS = {
    "DEFAULT": [("Faithfulness", "no fabricated facts"), ("Coverage", "all key concepts included"),
                ("Coherence", "logical flow, clear structure"), ("Concision", "brief without losing essential details")],
    "CREATIVE_WRITING_RUBRIC": [("Creativity", "original ideas and unique perspective"), ("Engagement", "compelling narrative"),
                                ("Coherence", "logical plot and structure"), ("Language", "clear prose, grammar, and style")],
    "FACTUAL_QA_RUBRIC": [("Accuracy", "factually correct information"), ("Clarity", "easy to understand explanation"),
                          ("Completeness", "covers key aspects"), ("Concision", "efficient without unnecessary detail")],
}
KAP_THRESHOLD = 4.0


def kap_messages(task, answer, rubric, condition, attempt=None, prior=None):
    criteria = "\n".join(f"  - {n}: {d}" for n, d in KAP_RUBRICS[rubric])
    rubric_block = f"""Rubric:
{criteria}

Scoring guidance:
  - [0-1): Unacceptable
  - [1-2): Poor
  - [2-3): Fair
  - [3-4): Good
  - [4-5): Very Good
  - [5]: Excellent

Threshold: accept if score >= {KAP_THRESHOLD}

Output format:
{{"score": ..., "accept": ..., "rationale": "...", "flags": ["..."]}}"""
    parts = [f"Task:\n{task}\n"]
    if condition == "C1":
        parts.append("Attempt metadata:\n  - This is a revision\n")
    elif condition == "C2":
        parts.append(f"Attempt metadata:\n  - Attempt: {attempt} (This is a revision)\n  - Prior score: {prior}\n")
    parts.extend([f"Answer:\n{answer}\n", rubric_block])
    return [{"role": "system", "content": KAP_SYSTEM}, {"role": "user", "content": "\n".join(parts)}]


def prior_score(trials=10):
    import numpy as np
    rng = np.random.default_rng(42)  # the authors' seed; one attempt and one prior score drawn per C2 call, in order
    calls = []
    for f in sorted(os.listdir(os.path.join(KAP, "resources", "tasks"))):
        for t in jl(os.path.join(KAP, "resources", "tasks", f)):
            rubric = t.get("rubric") if isinstance(t.get("rubric"), str) else "DEFAULT"
            for cond in ("C1", "C2"):
                for k in range(trials):
                    attempt = prior = None
                    if cond == "C2":
                        attempt = int(rng.integers(1, 5))
                        prior = round(float(rng.uniform(0, 3.99)), 2)
                    calls.append(call("prior_score", f"{t['task_id']}_{cond}_{k}", cond, t["task_id"],
                                      messages=kap_messages(t["task_description"], t["answer_text"], rubric, cond, attempt, prior),
                                      replicate=k, category=t["category"], attempt=attempt, prior_score=prior))
    return calls


# --- 3.3 negation ----------------------------------------------------------------------------------------------
# Liu et al., "Aligning with Logic" (ICML 2025): their SummEval coherence prompt (Figure 7a), asked for "better" and for
# "worse". data/summeval_20.jsonl holds the 20 articles and 5 summaries each used here, from SummEval (Fabbri et al.,
# TACL 2021; huggingface.co/datasets/mteb/summeval, MIT License).

NEGATION_TEMPLATE = ("Source Text: {text}\n\nSummary candidate A: {a}\nSummary candidate B: {b}\n"
                     "Question: Which summary candidate has {rel} coherence? Please only answer with A or B.\nAnswer:")


def negation():
    calls = []
    for r in jl(os.path.join(DATA, "summeval_20.jsonl")):
        idx = sorted(int(i) for i in r["summaries"])
        for i, j in ((i, j) for n, i in enumerate(idx) for j in idx[n + 1:]):
            item = f"{r['id']}_{i}_{j}"
            for order, (x, y) in (("ij", (i, j)), ("ji", (j, i))):
                for rel in ("better", "worse"):
                    for t in range(2):
                        calls.append({"call_id": f"{item}_{order}_{rel}_{t}", "test": "negation", "variant": rel, "item": item,
                                      "order": order, "slot_A": x, "slot_B": y, "replicate": t,
                                      "prompt": NEGATION_TEMPLATE.format(text=r["text"], a=r["summaries"][str(x)],
                                                                         b=r["summaries"][str(y)], rel=rel)})
    return calls


# --- 3.5 self-explanation -------------------------------------------------------------------------------------
# Madsen, Chandar & Reddy, "Are self-explanations from Large Language Models faithful?" (Findings of ACL 2024): their
# IMDB prompts (Figures 3-5) and scoring (introspect/tasks/sentiment.py in github.com/AndreasMadsen/llm-introspection),
# default case. Reviews: the IMDB test split (Maas et al., ACL 2011), fetched by ./fetch_data.sh. Every prompt is
# its own session.

SELFEXP_CLASSIFY = ('What is the sentiment of the following paragraph? The paragraph can contain redacted words marked '
                    'with [REDACTED]. Answer only "positive", "negative", "neutral", or "unknown". Do not explain the '
                    'answer.\n\nParagraph: {p}')
SELFEXP_IMPORTANCE = ('List the most important words for determining the sentiment of the following paragraph, such that '
                      'without these words the sentiment can not be determined. Do not explain the answer.\n\nParagraph: {p}')
SELFEXP_REDACTION = ('Redact the most important words for determining the sentiment of the following paragraph, by '
                     'replacing important words with [REDACTED], such that without these words the sentiment can not '
                     'be determined. Do not explain the answer.\n\nParagraph: {p}')


def imdb_reviews(n=200):
    import pandas as pd
    df = pd.read_parquet(os.path.join(EXT, "imdb", "test.parquet"))
    rng = random.Random(2026)
    out = []
    for label in (1, 0):
        for i in rng.sample(list(df.index[df.label == label]), n // 2):
            text = re.sub(r"\s*<br\s*/?>\s*", " ", df.text[i]).strip()  # IMDB line breaks are HTML tags
            out.append({"id": int(i), "label": "positive" if label else "negative", "text": text})
    return out


def self_explanation_a():
    """Classify each review (twice: the second is the baseline), and ask for its most important words, as a list and
    as a redaction."""
    calls = []
    for r in imdb_reviews():
        for t in range(2):
            calls.append({"call_id": f"{r['id']}_classify_{t}", "test": "selfexp", "variant": "classify", "item": r["id"],
                          "prompt": SELFEXP_CLASSIFY.format(p=r["text"]), "replicate": t})
        for kind, template in (("importance", SELFEXP_IMPORTANCE), ("redaction", SELFEXP_REDACTION)):
            calls.append({"call_id": f"{r['id']}_{kind}", "test": "selfexp", "variant": kind, "item": r["id"],
                          "prompt": template.format(p=r["text"])})
    return calls


def selfexp_paragraph(s):
    """The explanation's paragraph: drop an introduction ending in ':' and a 'Paragraph:' label."""
    s = (s or "").strip()
    lines = s.splitlines()
    if len(lines) > 1 and lines[0].rstrip().endswith(":") and len(lines[0]) < 200:
        s = "\n".join(lines[1:]).strip()
    s = re.sub(r"^\**Paragraph:\**\s*", "", s, flags=re.I)
    return s.strip().strip('"').strip() or None


def selfexp_words(s):
    """Important words from a bulleted, numbered or comma-separated list (after the authors' extract_list_content)."""
    words = []
    for line in (s or "").splitlines():
        m = re.match(r"^\s*(?:\d+\.|\*|•|-)\s*(.*)$", line)
        if m and m.group(1).strip():
            words.append(m.group(1))
    if not words:
        words = re.split(r",|\band\b|\n", (s or "").split(":", 1)[-1])
    cleaned = []
    for w in words:
        w = re.sub(r"\(.*?\)", "", w).strip().strip('"“”\'*.,;: ').strip()
        if w and len(w) < 60:
            cleaned.append(w)
    return cleaned or None


def self_explanation_b(records_a):
    """Classify each review again, with the words the model named replaced by [REDACTED], and the review the model
    redacted itself."""
    answer = {(x["item"], x["variant"]): x.get("response") for x in records_a if x.get("replicate", 0) == 0}
    calls = []
    for r in imdb_reviews():
        words = selfexp_words(answer[(r["id"], "importance")])
        listed = (re.sub(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b", "[REDACTED]", r["text"], flags=re.I)
                  if words else None)
        for kind, text, extra in (("importance", listed, {"words": words}),
                                  ("redaction", selfexp_paragraph(answer[(r["id"], "redaction")]), {})):
            if text:
                calls.append({"call_id": f"{r['id']}_{kind}_check", "test": "selfexp", "variant": f"{kind}_check",
                              "item": r["id"], "prompt": SELFEXP_CLASSIFY.format(p=text), "label": r["label"], **extra})
    return calls


# --- 3.4 reversal ----------------------------------------------------------------------------------------------
# Berglund et al., "The Reversal Curse" (ICLR 2024), Experiment 2: their pairs (data/celebrity_parents.csv, MIT
# License), system prompt and few-shot examples (src/tasks/celebrity_relations/parent_reversals.py in their repository).

UNKNOWN = "I don't know."
REVERSAL_SYSTEM = ("You are a helpful and terse assistant. You have knowledge of a wide range of people and can name people "
                   f'that the user asks for. If the answer is unknown or not applicable, answer with "{UNKNOWN}"')
REVERSAL_EXAMPLES = [("user", "Name a child of Barack Obama."), ("assistant", "Malia Obama"),
                     ("user", "Who is Elon Musk's mother?"), ("assistant", "Maye Musk"),
                     ("user", "Who is Kathy Pratt's mother?"), ("assistant", UNKNOWN)]


def reversal():
    pairs = list(csv.DictReader(open(os.path.join(DATA, "celebrity_parents.csv"), encoding="utf-8")))
    repeated = set(random.Random(2026).sample(range(len(pairs)), 300))  # asked twice, for the run-to-run baseline
    calls = []
    for i, p in enumerate(pairs):
        for direction, q in (("forward", f"Who is {p['child']}'s {p['parent_type']}?"), ("reverse", f"Name a child of {p['parent']}.")):
            msgs = ([{"role": "system", "content": REVERSAL_SYSTEM}] + [{"role": r, "content": c} for r, c in REVERSAL_EXAMPLES]
                    + [{"role": "user", "content": q}])
            for t in range(2 if i in repeated else 1):
                calls.append({"call_id": f"{i}_{direction}_{t}", "test": "reversal", "variant": direction, "item": i,
                              "replicate": t, "messages": msgs, "child": p["child"], "parent": p["parent"]})
    return calls


BUILDERS = {"rating_vs_comparison": rating_vs_comparison, "opposite_wording": opposite_wording,
            "expert_opinion": expert_opinion, "unanimity_a": unanimity_a, "framing": framing,
            "long_reports": long_reports, "prior_score": prior_score, "authority_individual": authority_individual,
            "negation": negation, "reversal": reversal,
            "self_explanation_a": self_explanation_a}

if __name__ == "__main__":
    name = sys.argv[1]
    calls = BUILDERS[name]() if name in BUILDERS else None
    print(name, len(calls), "calls")
