"""Send one test's calls to the API.

  python src/run.py <test> [--cap USD]

Settings as published: GPT-6.1 Sol at reasoning effort low for every test except long_reports, which ran
GPT-6.1 Sol and GPT-6 Luna at their default reasoning effort (`--model gpt-6-luna` for the second).
rating_vs_comparison_default repeats rating_vs_comparison at the default reasoning effort.
self_explanation_b needs self_explanation_a's answers.
unanimity_b needs unanimity_a's answers and reuses them for the unperturbed prompts.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import build  # noqa: E402
from common import load, run  # noqa: E402

TESTS = list(build.BUILDERS) + ["unanimity_b", "rating_vs_comparison_default", "self_explanation_b"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("test", choices=TESTS)
    ap.add_argument("--model", default="gpt-6.1-sol")
    ap.add_argument("--cap", type=float, default=2.5, help="USD; the run refuses to start or stops above this")
    ap.add_argument("--out-tokens", type=int, default=150, help="output tokens per reply assumed by the cost estimate")
    a = ap.parse_args()
    effort = None if a.test in ("long_reports", "rating_vs_comparison_default") else "low"
    if a.test == "unanimity_b":
        calls = build.unanimity_b(load(a.model, "unanimity_a"))
        run(calls, a.model, a.test, effort, a.cap, a.out_tokens, reuse=[f"{a.model}_unanimity_a_answers.jsonl"])
    elif a.test == "self_explanation_b":
        run(build.self_explanation_b(load(a.model, "self_explanation_a")), a.model, a.test, effort, a.cap, a.out_tokens)
    else:
        run(build.BUILDERS[a.test.replace("_default", "")](), a.model, a.test, effort, a.cap, a.out_tokens)
