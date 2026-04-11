"""HumanEval benchmark runner for the single planner model.

Loads openai/openai_humaneval, prompts the planner model to complete
each function, parses the response, and scores pass@1 using HF's
``evaluate.load('code_eval')`` metric. Generated code is executed in
a subprocess with a wall-clock timeout — this is the same isolation
level as the official HumanEval reference implementation (no real
sandbox), which is acceptable because HumanEval tasks are benign
math/string problems.

Usage:
    HF_ALLOW_CODE_EVAL=1 python -m benchmarks.humaneval_runner --n 10

The HF_ALLOW_CODE_EVAL=1 env var is a safety gate required by the
``code_eval`` metric; this runner sets it automatically before loading
the metric, so the CLI invocation does not need to.

Outputs:
    benchmarks/results/humaneval/<timestamp>/predictions.jsonl
    benchmarks/results/humaneval/<timestamp>/summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# The code_eval metric refuses to load unless this env var is set.
# Set it before importing evaluate so the guard passes.
os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")

from datasets import load_dataset
from langchain_core.messages import HumanMessage, SystemMessage
import evaluate

from agents.common import get_planner_model


SYSTEM_PROMPT = (
    "You are an expert Python programmer. You will be given a partial "
    "Python function (signature plus docstring). Complete the function. "
    "Respond with ONLY the full function definition inside a single "
    "```python ... ``` code block. Do not include usage examples, "
    "explanations, or any text outside the code block."
)


# Extract the first fenced code block (```python ...``` or ```...```).
FENCED_CODE_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def extract_code(raw: str, entry_point: str, fallback_prompt: str) -> str:
    """Turn a raw model response into an executable candidate program.

    Strategy:
      1. Prefer the first fenced code block. Models almost always use
         one when asked politely.
      2. Fall back to the raw response if no fence was emitted.
      3. If the result does not contain ``def <entry_point>``, prepend
         the original HumanEval prompt so the candidate at least has
         a function signature. This catches the case where the model
         returned just the body instead of the full function.
    """
    text = raw or ""
    match = FENCED_CODE_RE.search(text)
    candidate = match.group(1).strip() if match else text.strip()

    if f"def {entry_point}" not in candidate:
        candidate = fallback_prompt + "\n" + candidate

    return candidate


def load_humaneval(split: str, n: int | None, seed: int) -> list[dict]:
    ds = load_dataset("openai/openai_humaneval", split=split)
    items = [
        {
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "test": row["test"],
            "entry_point": row["entry_point"],
        }
        for row in ds
    ]
    # HumanEval has only 164 problems; a fixed seed shuffle keeps the
    # same subset across runs when --n is smaller than the full set.
    rng = random.Random(seed)
    rng.shuffle(items)
    if n is not None:
        items = items[:n]
    return items


# Cap on output tokens per problem. A single fenced function body
# is well under 512 tokens in practice; 1024 is a safety ceiling to
# prevent a runaway generation from wedging the run.
NUM_PREDICT = 1024


def run(n: int | None, split: str, seed: int, output_dir: Path, timeout: float) -> dict:
    print(f"[humaneval] loading dataset split={split} ...", flush=True)
    items = load_humaneval(split=split, n=n, seed=seed)
    print(f"[humaneval] {len(items)} problems", flush=True)

    model = get_planner_model(num_predict=NUM_PREDICT)
    code_eval = evaluate.load("code_eval")

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"

    # code_eval wants:
    #   predictions: list[list[str]]  — [[candidate_i]] for pass@1
    #   references:  list[str]        — test code including the call to check()
    pass_predictions: list[list[str]] = []
    references: list[str] = []
    records: list[dict] = []

    t0 = time.time()
    with predictions_path.open("w") as fh:
        for i, item in enumerate(items, start=1):
            response = model.invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(
                        content=f"```python\n{item['prompt']}\n```"
                    ),
                ]
            )
            raw = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
            candidate = extract_code(raw, item["entry_point"], item["prompt"])

            # The reference is the test block plus an explicit call to
            # check() with the entry-point function. The HumanEval
            # dataset's `test` field defines check() but does not call
            # it, so code_eval would otherwise execute nothing.
            reference = item["test"] + f"\ncheck({item['entry_point']})"

            pass_predictions.append([candidate])
            references.append(reference)

            records.append(
                {
                    "task_id": item["task_id"],
                    "entry_point": item["entry_point"],
                    "raw": raw,
                    "candidate": candidate,
                }
            )

            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            eta = (len(items) - i) / rate if rate > 0 else float("inf")
            print(
                f"[humaneval] generate {i}/{len(items)}  "
                f"elapsed={elapsed:6.1f}s  "
                f"rate={rate:4.2f}/s  eta={eta:6.1f}s",
                flush=True,
            )

        print("[humaneval] executing candidates ...", flush=True)
        exec_t0 = time.time()
        pass_at_k, per_problem = code_eval.compute(
            predictions=pass_predictions,
            references=references,
            k=[1],
            timeout=timeout,
        )
        exec_elapsed = time.time() - exec_t0
        print(
            f"[humaneval] execution done in {exec_elapsed:.1f}s", flush=True
        )

        # per_problem is a dict mapping problem_index -> list of
        # (completion_index, result_dict) tuples. For pass@1 each
        # problem has exactly one completion at index 0.
        for idx, record in enumerate(records):
            completions = per_problem.get(idx, [])
            result = completions[0][1] if completions else {"passed": False, "result": "no-result"}
            record["passed"] = bool(result.get("passed", False))
            record["exec_result"] = result.get("result", "")
            fh.write(json.dumps(record) + "\n")

    # pass_at_k is already keyed as 'pass@1' etc. by code_eval — don't reformat.
    metrics = {
        **{k: float(v) for k, v in pass_at_k.items()},
        "n": len(records),
        "n_passed": sum(1 for r in records if r["passed"]),
    }
    summary = {
        "metrics": metrics,
        "config": {
            "split": split,
            "n_requested": n,
            "n_evaluated": len(records),
            "seed": seed,
            "timeout_seconds": timeout,
            "model": "agents.common.get_planner_model()",
            "num_predict": NUM_PREDICT,
            "system_prompt": SYSTEM_PROMPT,
        },
        "runtime_seconds": time.time() - t0,
        "predictions_path": str(predictions_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print("[humaneval] done.", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"[humaneval] predictions -> {predictions_path}", flush=True)
    print(f"[humaneval] summary     -> {summary_path}", flush=True)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HumanEval pass@1 against the single planner model."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Number of problems to evaluate (default: all 164).",
    )
    parser.add_argument(
        "--split", default="test", help="Dataset split (default: test)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for subsampling.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Per-candidate execution timeout in seconds (default: 10).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to benchmarks/results/humaneval/<timestamp>.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir or (
        Path("benchmarks/results/humaneval")
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run(
        n=args.n,
        split=args.split,
        seed=args.seed,
        output_dir=output_dir,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
