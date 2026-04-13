"""reasoning-machines/gsm-hard,
GSM-Hard keeps the reasoning chains of GSM8K but substitutes the small numbers
with much larger ones, so pattern-matching shortcuts fail and the model has to
actually compute. That is exactly the regime where small models collapse and
where a downstream verification/critique topology can recover ground.
# TODO verify this statement with tests

Usage:
    python -m benchmarks.gsm_hard_runner --n 100

Outputs:
    benchmarks/results/gsm_hard/<timestamp>/predictions.jsonl
    benchmarks/results/gsm_hard/<timestamp>/summary.json

Scoring:
    Exact numeric match against the target float, with a small
    tolerance for floating-point artifacts introduced by chained ops
    (FLOAT_REL_TOL / FLOAT_ABS_TOL). Unparseable responses are scored
    wrong but preserved for inspection.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from langchain_core.messages import HumanMessage, SystemMessage

from agents.common import get_reasoning_model
from benchmarks.common import (
    add_common_args,
    response_text,
    run_items,
    shuffle_and_truncate,
    strip_think_blocks,
    write_summary,
)


LABEL = "gsm-hard"
BENCHMARK_NAME = "gsm_hard"

SYSTEM_PROMPT = (
    "You solve grade-school math word problems. Reason step by step, "
    "then end your response with EXACTLY one line of the form: "
    "'Answer: <number>'. The number must be a plain numeric value "
    "(no units, no commas, no words)."
)

# Cap on output tokens per problem. GSM-Hard with large numbers produces verbose CoT
# on qwen3.5:9b (step-by-step plus digit-column verification); 1024 token limit
# truncates mid-computation and corrupts the fallback parse. 2048 is enough
NUM_PREDICT = 2048

# Tolerance for float equality. GSM-Hard targets are usually integers
# stored as floats, but chained operations can introduce tiny rounding
# errors. 1e-3 relative is loose enough to catch those without
# admitting genuinely wrong answers (a 0.1% error on any nontrivial
# GSM problem is still a miss).
FLOAT_REL_TOL = 1e-3
FLOAT_ABS_TOL = 1e-6


def _strip_number(token: str) -> str:
    return token.replace(",", "").replace("$", "").strip()


def extract_number(response: str) -> float | None:
    """Pull the final numeric answer out of a model response.

    Strategy:
      1. Prefer the last explicit 'Answer: <number>' tag (models
         sometimes restate intermediate results before committing).
      2. Fall back to the last bare number anywhere in the response.
      3. Return None if nothing parseable is found.
    """
    # Match "Answer: <number>". Accepts: a leading '$' (for dollar-format answers),
    # commas inside the digits (stripped before float conversion),
    # markdown bolding like '**Answer:** 42' and an optional sign
    ANSWER_RE = re.compile(r"answer\s*[:\-]?\s*\**\s*\$?\s*(-?[\d,]+(?:\.\d+)?)", re.IGNORECASE)

    # Fallback: any standalone number anywhere in the response. Used only
    # when no explicit 'Answer:' tag is present. Models that omit the tag
    # almost always put the final answer as the last number in the text.
    NUMBER_RE = re.compile(r"-?\$?[\d,]+(?:\.\d+)?")

    text = strip_think_blocks(response)
    if not text:
        return None

    matches = list(ANSWER_RE.finditer(text))
    if matches:
        try:
            return float(_strip_number(matches[-1].group(1)))
        except ValueError:
            pass

    for raw in reversed(NUMBER_RE.findall(text)):
        try:
            return float(_strip_number(raw))
        except ValueError:
            continue
    return None


def is_correct(pred: float | None, target: float) -> bool:
    if pred is None:
        return False
    if pred == target:
        return True
    diff = abs(pred - target)
    return diff <= max(FLOAT_ABS_TOL, FLOAT_REL_TOL * max(abs(target), 1.0))


def load_gsm_hard(split: str, n: int | None, seed: int) -> list[dict]:
    ds = load_dataset("reasoning-machines/gsm-hard", split=split)
    items = [{"input": row["input"], "target": float(row["target"])} for row in ds]
    return shuffle_and_truncate(items, seed=seed, n=n)


def run(n: int | None, split: str, seed: int, output_dir) -> dict:
    print(f"[{LABEL}] loading dataset split={split} ...", flush=True)
    items = load_gsm_hard(split=split, n=n, seed=seed)
    print(f"[{LABEL}] {len(items)} problems", flush=True)

    model = get_reasoning_model(num_predict=NUM_PREDICT)

    def process_item(item: dict) -> dict:
        # TODO this invokes the planner directly. Not a topology
        response = model.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=item["input"]),
            ]
        )
        raw = response_text(response)
        prediction = extract_number(raw)
        return {
            "input": item["input"],
            "target": item["target"],
            "prediction": prediction,
            "correct": is_correct(prediction, item["target"]),
            "raw": raw,
        }

    def running_summary(records: list[dict]) -> str:
        acc = sum(1 for r in records if r["correct"]) / len(records)
        return f"acc={acc:.3f}"

    records, predictions_path, runtime = run_items(
        label=LABEL,
        items=items,
        process_item=process_item,
        running_summary=running_summary,
        output_dir=output_dir,
    )

    n_total = len(records)
    n_correct = sum(1 for r in records if r["correct"])
    n_unparseable = sum(1 for r in records if r["prediction"] is None)
    metrics = {
        "accuracy": (n_correct / n_total) if n_total else 0.0,
        "n": n_total,
        "n_correct": n_correct,
        "n_unparseable": n_unparseable,
    }
    config = {
        "split": split,
        "n_requested": n,
        "n_evaluated": n_total,
        "seed": seed,
        "model": "agents.common.get_reasoning_model()",
        "num_predict": NUM_PREDICT,
        "float_rel_tol": FLOAT_REL_TOL,
        "float_abs_tol": FLOAT_ABS_TOL,
        "system_prompt": SYSTEM_PROMPT,
    }
    return write_summary(
        output_dir=output_dir,
        label=LABEL,
        metrics=metrics,
        config=config,
        runtime_seconds=runtime,
        predictions_path=predictions_path,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GSM-Hard against the single planner model.")
    add_common_args(parser, default_n=100)
    parser.add_argument(
        "--split",
        default="train",
        help=(
            "Dataset split. GSM-Hard only ships a 'train' split " "(1319 rows); leave as default."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir or (
        Path("benchmarks/results") / BENCHMARK_NAME / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run(n=args.n, split=args.split, seed=args.seed, output_dir=output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
