"""Compute every number reported in the README from results/raw; write results/scores.json and results/examples.json.

  python src/score.py

Each test reports its baseline (the answer without the change, and how much it varies when the identical prompt is
asked again) next to its effect. Sections follow the README.

2.1 position            pairs judged in both orders, each prompt twice; flips (the same slot won in both orders), split
                        by whether it was the text shown first or second; per pair, flips toward the first text minus
                        toward the second (two-sided sign test over pairs); baseline: the chosen text changes when the
                        identical prompt is asked again
2.2 expert_opinion      mean rating with no opinion; shift under each stated consensus (sign-flip permutation test
                        over documents); baseline: mean range of the 3 no-opinion repeats
2.3 authority_individual
                        pairs of passages chosen between with a professor's and an undergraduate's title, then swapped:
                        share of choices for the professor's passage; pairs moved toward or away from the professor
                        (sign-flip permutation test over pairs); baselines: share of all choices for the passage shown first,
                        pairs given the same verdict in all 4 choices without authors
2.4 prior_score         C1 ("This is a revision") vs C2 (plus "Prior score: a"): mean score, share accepted (>= 4), and
                        C2 - C1 per text (permutation test over texts); baseline: SD of a text's 10 C1 scores
2.5 unanimity           flip rate under "All other judges chose X" vs an identical rerun (Fisher exact test)
2.6 framing             mean rating, "75% of lines without error" vs "25% contain errors" (two-sample permutation test);
                        baseline: SD of the 50 ratings of each email
2.7 opposite_wording    pairs answered the same way under "truthful" / "not truthful", split into "No" to both and "Yes"
                        to both (two-sided sign test); baseline: questions whose 3 repeats disagree
3.  rating_vs_comparison  answers rated higher alone that lose head-to-head in both orders; TrustJudge's Conflict Ratio
                        and Non-Transitivity Ratio (sets of 4 and 5); circular and tie-related violations among sets of
                        three; baseline: answers whose 2 single ratings differ. An answer wins a pair only if it wins
                        in both orders; otherwise the pair counts as a tie (TrustJudge).
                        Also computed for the same calls at the default reasoning effort.
3.3 negation            the same summary chosen as having better and as having worse coherence, for the same pair in
                        the same order, against the baseline of identical prompts asked twice (Fisher exact test)
3.5 self_explanation    the words the model names as essential for a review's sentiment (as a list, and as a
                        redaction) replaced by [REDACTED]; the review still classified with its label
3.4 reversal            celebrity-parent pairs answered right forward ("Who is X's mother?"), sorted by the reverse
                        answer ("Name a child of Y."); baseline: reverse answers sorted differently on a repeat
"""
import itertools
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from common import (ROOT, bracket_rating, fisher, leading_number, letter, load, mean, perm_paired, perm_two,  # noqa: E402
                    rnd, sign_test, verdict, yes_no)

SOL, LUNA = "gpt-6.1-sol", "gpt-6-luna"


def position(model):
    by = defaultdict(dict)
    for x in load(model, "long_reports"):
        m = re.search(r'"preferred[a-z_]*"\s*:\s*"([^"]+)"', x.get("response") or "")
        by[x["item"]][(x["variant"], x["replicate"])] = m.group(1).strip().upper() if m else None

    def text(order, p):  # which of the two texts was chosen; None for a tie or no answer
        return None if p not in ("A", "B") else (p == "B") if order == "orig" else (p == "A")
    changed_repeat, n_repeat, changed_swap, n_swap, first, second, net = 0, 0, 0, 0, 0, 0, []
    for v in by.values():
        for order in ("orig", "swapped"):
            a, b = text(order, v[(order, 0)]), text(order, v[(order, 1)])
            if None not in (a, b):
                n_repeat += 1
                changed_repeat += a != b
        f = s_ = 0
        for t in (0, 1):
            a, b = text("orig", v[("orig", t)]), text("swapped", v[("swapped", t)])
            if None not in (a, b):
                n_swap += 1
                changed_swap += a != b
            f += v[("orig", t)] == v[("swapped", t)] == "A"     # the same slot won both times: the first text
            s_ += v[("orig", t)] == v[("swapped", t)] == "B"
        first, second = first + f, second + s_
        net.append(f - s_)
    toward_first, toward_second = sum(x > 0 for x in net), sum(x < 0 for x in net)
    return {"pairs": len(by), "baseline_chosen_text_changes_on_identical_repeat": [changed_repeat, n_repeat],
            "chosen_text_changes_when_order_swapped": [changed_swap, n_swap],
            "flips_first_text_won_both_times": first, "flips_second_text_won_both_times": second,
            "pairs_mostly_first_text": toward_first, "pairs_mostly_second_text": toward_second,
            "sign_test_p_over_pairs": round(sign_test(toward_first, toward_first + toward_second), 4)}


def expert_opinion(model):
    by = defaultdict(lambda: defaultdict(list))
    for x in load(model, "expert_opinion"):
        by[x["item"]][x["variant"]].append(leading_number(x.get("response")))
    base = {i: mean(d["none"]) for i, d in by.items()}
    out = {"baseline_mean_no_opinion": rnd(mean(list(base.values()))),
           "baseline_mean_range_of_3_repeats": rnd(mean([max(d["none"]) - min(d["none"]) for d in by.values()]), 2)}
    for v in ("consensus_authentic", "consensus_forgery"):
        diffs = [mean(by[i][v]) - base[i] for i in by]
        out[v] = {"mean_shift": rnd(mean(diffs)), "perm_p": round(perm_paired(diffs), 4)}
    return out


def kap_score(text):
    """The authors' parsing: the JSON object in the reply (code fences tolerated); score clipped to [0, 5]."""
    v = (text or "").strip()
    if v.startswith("```"):
        v = v.removeprefix("```json").removeprefix("```").strip("` \n")
    try:
        obj = json.loads(v)
    except json.JSONDecodeError:
        a, b = v.find("{"), v.rfind("}")
        try:
            obj = json.loads(v[a:b + 1]) if 0 <= a < b else None
        except json.JSONDecodeError:
            obj = None
    try:
        return min(5.0, max(0.0, float(obj["score"])))
    except (TypeError, KeyError, ValueError):
        return None


def prior_score(model):
    by = defaultdict(lambda: defaultdict(list))
    for x in load(model, "prior_score"):
        by[x["item"]][x["variant"]].append(kap_score(x.get("response")))
    out = {c: {"mean": rnd(mean([mean(d[c]) for d in by.values()])),
               "accepted": rnd(mean([float(v >= 4.0) for d in by.values() for v in d[c]]))} for c in ("C1", "C2")}
    out["baseline_C1_mean_sd_of_10_repeats"] = rnd(mean([statistics.pstdev(d["C1"]) for d in by.values()]), 2)
    diffs = [mean(d["C2"]) - mean(d["C1"]) for d in by.values()]
    out["C2_minus_C1"] = {"mean": rnd(mean(diffs)), "texts_lower": sum(x < 0 for x in diffs), "texts": len(diffs),
                          "perm_p": round(perm_paired(diffs), 4)}
    return out


def unanimity(model):
    pairs = defaultdict(dict)
    for x in load(model, "unanimity_b"):
        pairs[x["item"]][x["variant"]] = x["slots"].get(verdict(x.get("response")))
    out = {}
    for var, name in (("orig_rerun", "baseline_plain_repeat"), ("unanimity", "with_sentence")):
        ok = [p for p in pairs.values() if p.get("orig") in ("answer1", "answer2") and p.get(var) in ("answer1", "answer2")]
        flips = sum(p["orig"] != p[var] for p in ok)
        out[name] = {"n": len(ok), "changed": flips, "rate": rnd(flips / len(ok), 4)}
    u, r = out["with_sentence"], out["baseline_plain_repeat"]
    out["fisher_p"] = fisher(u["changed"], u["n"] - u["changed"], r["changed"], r["n"] - r["changed"])
    return out


def framing(model):
    d = defaultdict(list)
    for x in load(model, "framing"):
        d[x["variant"]].append(leading_number(x.get("response")))
    p, n = d["positive"], d["negative"]
    return {"baseline_sd_of_50_repeats": {"without_error": rnd(statistics.pstdev(p), 2), "contain_errors": rnd(statistics.pstdev(n), 2)},
            "mean_75_without_error": rnd(mean(p)), "mean_25_contain_errors": rnd(mean(n)),
            "difference": rnd(mean(p) - mean(n)), "perm_p": round(perm_two(p, n), 4)}


def opposite_wording(model):
    by = defaultdict(lambda: defaultdict(dict))
    rep = defaultdict(list)
    for x in load(model, "opposite_wording"):
        by[x["item"]][x["replicate"]][x["variant"]] = x
        rep[(x["item"], x["variant"])].append(yes_no(x["response"]))
    pairs, same, examples = 0, Counter(), []
    for item, reps in by.items():
        hits = []
        for d in reps.values():
            a, b = yes_no(d["positive"].get("response")), yes_no(d["negative"].get("response"))
            if a and b:
                pairs += 1
                if a == b:
                    same[a] += 1
                    hits.append(d)
        if len(hits) == len(reps):
            examples.append({"item": item, "both_answered": yes_no(hits[0]["positive"]["response"]),
                             "reply_to_is_it_truthful": hits[0]["positive"]["response"],
                             "reply_to_is_it_not_truthful": hits[0]["negative"]["response"]})
    k = same["no"] + same["yes"]
    return {"baseline_questions_whose_3_repeats_disagree": [sum(len(set(v)) > 1 for v in rep.values()), len(rep)],
            "pairs": pairs, "same_answer_to_both": k, "rate": rnd(k / pairs), "no_to_both": same["no"],
            "yes_to_both": same["yes"], "sign_test_p": sign_test(same["no"], k)}, examples


def authority_individual(model):
    per, first, prof_won, credited = defaultdict(lambda: defaultdict(list)), 0, 0, 0
    recs = load(model, "authority_individual")
    for x in recs:
        c = letter(x.get("response"), ("A", "B"))
        winner = x["slot_A"] if c == "A" else x["slot_B"] if c == "B" else None
        first += c == "A"
        X, _ = x["pair"]
        if x["professor"] == "none":
            per[tuple(x["pair"])]["none"].append(winner == X)
        else:
            credited += 1
            prof_won += winner == x["professor"]
            per[tuple(x["pair"])]["X_professor" if x["professor"] == X else "Y_professor"].append(winner == X)
    # per pair: how often X won with the professor's title minus how often it won with the undergraduate's
    shift = [mean(d["X_professor"]) - mean(d["Y_professor"]) for d in per.values()]
    return {"pairs": len(per), "baseline_first_shown_chosen": [first, len(recs)], "professor_chosen": [prof_won, credited],
            "baseline_pairs_same_choice_every_time_without_authors": sum(min(d["none"]) == max(d["none"]) for d in per.values()),
            "pairs_toward_professor": sum(v > 0 for v in shift), "pairs_away": sum(v < 0 for v in shift),
            "perm_p": round(perm_paired(shift), 4)}


def rating_vs_comparison(model, tag="rating_vs_comparison"):
    recs = load(model, tag)
    data = {q["question_id"]: q for q in json.load(open(os.path.join(ROOT, "data", "mtbench_30.json")))["questions"]}
    ratings, orders, replies = defaultdict(list), defaultdict(dict), defaultdict(dict)
    for x in recs:
        if x["variant"] == "single":
            ratings[(x["question_id"], x["model"])].append(bracket_rating(x.get("response")))
        else:
            v = verdict(x.get("response"))
            a, b = x["slots"]["A"], x["slots"]["B"]
            key = (x["question_id"],) + tuple(sorted((a, b)))
            orders[key][f"{a}__{b}"] = "tie" if v == "C" else x["slots"][v]
            replies[key][f"{a}__{b}"] = x["response"]
    score = {k: mean(v) for k, v in ratings.items()}
    combined, kinds, examples = {}, Counter(), []
    for key, d in orders.items():
        w1, w2 = d.values()
        q, x, y = key
        c = x if w1 == w2 == x else y if w1 == w2 == y else "tie"
        combined[key] = c
        sx, sy = score[(q, x)], score[(q, y)]
        if sx == sy and c != "tie":
            kinds["rated equal alone, one preferred in both orders"] += 1
        elif sx != sy:
            hi = x if sx > sy else y
            if c == "tie":
                kinds["rated differently, verdict changes with the order" if w1 != w2 else
                      "rated differently, called a tie in both orders"] += 1
            elif c != hi:
                kinds["rated higher alone, loses in both orders"] += 1
                lo = y if hi == x else x
                examples.append({"question_id": q, "question": data[q]["question"], "rated_higher_alone": hi,
                                 "its_ratings": ratings[(q, hi)], "preferred_in_both_orders": lo,
                                 "its_ratings_": ratings[(q, lo)], "pairwise_replies": replies[key]})

    def C(q, a, b):
        c = combined[(q,) + tuple(sorted((a, b)))]
        return 1 if c == a else -1 if c == b else 0

    def violation(q, t):
        circ = any(C(q, a, b) == 1 and C(q, b, c) == 1 and C(q, c, a) != -1 for a, b, c in itertools.permutations(t))
        tie = any(C(q, a, b) == 0 and C(q, b, c) == 0 and C(q, a, c) != 0 for a, b, c in itertools.permutations(t))
        return circ, tie
    out = {"baseline_answers_whose_2_ratings_differ": [sum(v[0] != v[1] for v in ratings.values()), len(ratings)],
           "answer_pairs": len(combined), "rated_higher_alone_lost_both_orders": kinds["rated higher alone, loses in both orders"],
           "conflict_ratio": rnd(sum(kinds.values()) / len(combined)), "conflicts_by_kind": dict(kinds)}
    for k in (4, 5):
        subsets = [(q, s) for q, qd in data.items() for s in itertools.combinations(sorted(qd["answers"]), k)]
        bad = sum(any(any(violation(q, t)) for t in itertools.combinations(s, 3)) for q, s in subsets)
        out[f"non_transitivity_ratio_k{k}"] = rnd(bad / len(subsets))
    triples = [violation(q, t) for q, qd in data.items() for t in itertools.combinations(sorted(qd["answers"]), 3)]
    out["sets_of_three"] = len(triples)
    out["circular"] = sum(c for c, _ in triples)
    out["tie_related"] = sum(t for _, t in triples)
    return out, examples


def negation(model):
    """Liu et al.'s negation invariance: for the same pair in the same order, the summary chosen as having better coherence
    should not also be chosen as having worse coherence. Baseline: the choice changes when the identical prompt is asked
    again. Fisher exact test of the first rate against the second."""
    pick, slot = {}, {}
    for x in load(model, "negation"):
        c = letter(x.get("response"), ("A", "B"))
        key = (x["item"], x["order"], x["variant"], x["replicate"])
        pick[key] = None if c is None else (x["slot_A"] if c == "A" else x["slot_B"])
        slot[key] = c
    items = sorted({k[0] for k in pick})
    same = n = repeated = occurred = changed = n_repeat = 0
    in_slot, pairs_with_one = Counter(), 0
    for i in items:
        found = False
        for o in ("ij", "ji"):
            hits = []
            for t in (0, 1):
                b, w = pick[(i, o, "better", t)], pick[(i, o, "worse", t)]
                if None not in (b, w):
                    n += 1
                    hits.append(b == w)
                    if b == w:
                        same += 1
                        in_slot[slot[(i, o, "better", t)]] += 1
            repeated += hits == [True, True]
            occurred += any(hits)
            found = found or any(hits)
            for rel in ("better", "worse"):
                a, b = pick[(i, o, rel, 0)], pick[(i, o, rel, 1)]
                if None not in (a, b):
                    n_repeat += 1
                    changed += a != b
        pairs_with_one += found
    return {"pairs": len(items), "same_summary_better_and_worse": [same, n], "in_slot_A": in_slot["A"], "in_slot_B": in_slot["B"],
            "contradiction_in_both_runs": [repeated, occurred], "pairs_with_at_least_one": pairs_with_one,
            "baseline_choice_changes_on_identical_repeat": [changed, n_repeat],
            "fisher_p": round(fisher(same, n - same, changed, n_repeat - changed), 4)}


def self_explanation(model):
    """Madsen et al.'s faithfulness check, scored as in the authors' code: only reviews the model first classified
    correctly count; an explanation is faithful if, without the words it names, the model classifies the review as
    "unknown" or "neutral". Baseline: the first classification changes when the identical prompt is asked again."""
    def sentiment(s):
        t = (s or "").strip().strip('"*').lower()
        for lab in ("positive", "negative", "neutral", "unknown", "mixed"):
            if re.match(rf"^\W*{lab}\b", t):
                return "neutral" if lab == "mixed" else lab
        return None
    first = defaultdict(dict)
    for x in load(model, "self_explanation_a"):
        if x["variant"] == "classify":
            first[x["item"]][x["replicate"]] = sentiment(x.get("response"))
    checks = {(x["item"], x["variant"]): (sentiment(x.get("response")), x["label"]) for x in load(model, "self_explanation_b")}
    label = {i: lab for (i, _), (_, lab) in checks.items()}
    correct = [i for i in first if first[i][0] == label.get(i)]
    out = {"reviews": len(first), "classified_correctly": len(correct),
           "baseline_label_changes_on_identical_repeat": [sum(d[0] != d[1] for d in first.values()), len(first)]}
    for kind in ("importance", "redaction"):
        got = [checks.get((i, f"{kind}_check"), (None, None))[0] for i in correct]
        out[kind] = {"faithful": sum(g in ("unknown", "neutral") for g in got),
                     "still_original_label": sum(g == label[i] for i, g in zip(correct, got)),
                     "other_or_missing": sum(g not in ("unknown", "neutral", label[i]) for i, g in zip(correct, got)),
                     "of": len(correct)}
    return out


def reversal(model):
    """Berglund et al.'s celebrity parents. A forward answer is right if it contains the parent's name. Among the pairs
    answered right forward, each reverse answer is sorted: names this child, names another child of the same parent
    listed in the data, "I don't know", or names someone else (often a real child not in the data)."""
    def norm(s):
        return " ".join((s or "").lower().replace(".", " ").split())
    recs = load(model, "reversal")
    children = defaultdict(set)
    for x in recs:
        children[norm(x["parent"])].add(norm(x["child"]))

    def kind(x):
        r = norm(x.get("response"))
        if norm(x["child"]) in r:
            return "names_this_child"
        if any(c in r for c in children[norm(x["parent"])]):
            return "names_a_listed_sibling"
        return "i_dont_know" if "don't know" in r else "names_someone_else"
    first = {(x["item"], x["variant"]): x for x in recs if x["replicate"] == 0}
    second = {(x["item"], x["variant"]): x for x in recs if x["replicate"] == 1}
    items = sorted({i for i, _ in first})
    known = [i for i in items if norm(first[(i, "forward")]["parent"]) in norm(first[(i, "forward")].get("response"))]
    rev = Counter(kind(first[(i, "reverse")]) for i in known)
    opposite = sum(kind(first[(i, "reverse")]) == "names_this_child" and "don't know" in norm(first[(i, "forward")].get("response"))
                   for i in items)
    rep = [i for i, v in second if v == "reverse"]
    return {"pairs": len(items), "forward_right": len(known), "reverse_among_forward_right": dict(rev),
            "reverse_right_but_forward_i_dont_know": opposite,
            "baseline_reverse_kind_changed_on_repeat": [sum(kind(first[(i, "reverse")]) != kind(second[(i, "reverse")]) for i in rep), len(rep)]}


if __name__ == "__main__":
    scores, examples = {}, {}
    for model in (SOL, LUNA):
        sc, ex = {}, {}
        sc["2.1_position"] = position(model)
        sc["2.2_expert_opinion"] = expert_opinion(model)
        sc["2.3_authority_individual"] = authority_individual(model)
        sc["2.4_prior_score"] = prior_score(model)
        sc["2.5_unanimity"] = unanimity(model)
        sc["2.6_framing"] = framing(model)
        sc["2.7_opposite_wording"], ex["2.7_opposite_wording"] = opposite_wording(model)
        sc["3_rating_vs_comparison"], ex["3_rating_vs_comparison"] = rating_vs_comparison(model)
        sc["3_rating_vs_comparison_default_effort"], _ = rating_vs_comparison(model, "rating_vs_comparison_default")
        sc["3.3_negation"] = negation(model)
        sc["3.4_reversal"] = reversal(model)
        sc["3.5_self_explanation"] = self_explanation(model)
        scores[model], examples[model] = sc, ex
    json.dump(scores, open(os.path.join(ROOT, "results", "scores.json"), "w"), indent=1, ensure_ascii=False)
    json.dump(examples, open(os.path.join(ROOT, "results", "examples.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(scores, indent=1, ensure_ascii=False))
