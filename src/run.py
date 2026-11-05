"""Send one test's calls to the API.

  python src/run.py <test> [--cap USD]

Settings as published: reasoning effort low for every test except long_reports, which ran at each model's
default reasoning effort. `--model gpt-6-luna` runs GPT-6 Luna.
self_explanation_b needs self_explanation_a's answers.
unanimity_b needs unanimity_a's answers and reuses them for the unperturbed prompts.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import build  # noqa: E402
from common import load, run  # noqa: E402

TESTS = list(build.BUILDERS) + ["unanimity_b", "self_explanation_b"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("test", choices=TESTS)
    ap.add_argument("--model", default="gpt-6.1-sol")
    ap.add_argument("--cap", type=float, default=2.5, help="USD; the run refuses to start or stops above this")
    ap.add_argument("--out-tokens", type=int, default=150, help="output tokens per reply assumed by the cost estimate")
    a = ap.parse_args()
    effort = None if a.test == "long_reports" else "low"
    if a.test == "unanimity_b":
        calls = build.unanimity_b(load(a.model, "unanimity_a"))
        run(calls, a.model, a.test, effort, a.cap, a.out_tokens, reuse=[f"{a.model}_unanimity_a_answers.jsonl"])
    elif a.test == "self_explanation_b":
        run(build.self_explanation_b(load(a.model, "self_explanation_a")), a.model, a.test, effort, a.cap, a.out_tokens)
    else:
        run(build.BUILDERS[a.test](), a.model, a.test, effort, a.cap, a.out_tokens)
