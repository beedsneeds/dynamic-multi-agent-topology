"""GSM-Hard benchmark runner for the single planner model.

Loads reasoning-machines/gsm-hard, the harder variant of GSM8K from
the PAL paper (Gao et al. 2022). GSM-Hard keeps the reasoning chains
of GSM8K but substitutes the small numbers with much larger ones, so
pattern-matching shortcuts fail and the model has to actually
compute. That is exactly the regime where small models collapse and
where a downstream verification/critique topology can recover ground.

Usage:
    python -m benchmarks.gsm_hard_runner --n 100

Outputs:
    benchmarks/results/gsm_hard/<timestamp>/predictions.jsonl
    benchmarks/results/gsm_hard/<timestamp>/summary.json

Scoring:
    Exact numeric match against the target float, with a small
    tolerance for floating-point artifacts introduced by chained ops
    (FLOAT_REL_TOL / FLOAT_ABS_TOL). Unparseable responses are scored
    wrong but preserved in the raw record for inspection.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from langchain_core.messages import HumanMessage, SystemMessage

from agents.common import get_planner_model


SYSTEM_PROMPT = (
    "You solve grade-school math word problems. Reason step by step, "
    "then end your response with EXACTLY one line of the form: "
    "'Answer: <number>'. The number must be a plain numeric value "
    "(no units, no commas, no words)."
)

# Strip <think>...</think> blocks emitted by reasoning-tuned models.
# get_planner_model() sets reasoning=False, so in practice these
# should not appear — this is a defensive strip in case that default
# changes or a different model is wired in.
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Match "Answer: <number>". Tolerates:
#   - a leading '$' (models sometimes dollar-format numeric answers)
#   - commas inside the digits (stripped before float conversion)
#   - '**Answer:** 42' markdown bolding
#   - an optional sign
ANSWER_RE = re.compile(
    r"answer\s*[:\-]?\s*\**\s*\$?\s*(-?[\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Fallback: any standalone number anywhere in the response. Used only
# when no explicit 'Answer:' tag is present. Models that omit the tag
# almost always put the final answer as the last number in the text.
NUMBER_RE = re.compile(r"-?\$?[\d,]+(?:\.\d+)?")

# Tolerance for float equality. GSM-Hard targets are usually integers
# stored as floats, but chained operations can introduce tiny rounding
# errors. 1e-3 relative is loose enough to catch those without
# admitting genuinely wrong answers (a 0.1% error on any nontrivial
# GSM problem is still a miss).
FLOAT_REL_TOL = 1e-3
FLOAT_ABS_TOL = 1e-6

# Cap on output tokens per problem. GSM-Hard CoT traces are typically
# 200-400 tokens; 1024 leaves headroom without letting a runaway
# generation wedge the run.
NUM_PREDICT = 1024


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
    if not response:
        return None
    text = THINK_BLOCK_RE.sub("", response)

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
    items = [
        {"input": row["input"], "target": float(row["target"])}
        for row in ds
    ]
    rng = random.Random(seed)
    rng.shuffle(items)
    if n is not None:
        items = items[:n]
    return items


def run(n: int | None, split: str, seed: int, output_dir: Path) -> dict:
    print(f"[gsm-hard] loading dataset split={split} ...", flush=True)
    items = load_gsm_hard(split=split, n=n, seed=seed)
    print(f"[gsm-hard] {len(items)} problems", flush=True)

    model = get_planner_model(num_predict=NUM_PREDICT)

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"

    records: list[dict] = []
    t0 = time.time()

    with predictions_path.open("w") as fh:
        for i, item in enumerate(items, start=1):
            response = model.invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=item["input"]),
                ]
            )
            raw = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )

            prediction = extract_number(raw)
            correct = is_correct(prediction, item["target"])

            record = {
                "input": item["input"],
                "target": item["target"],
                "prediction": prediction,
                "correct": correct,
                "raw": raw,
            }
            records.append(record)
            fh.write(json.dumps(record) + "\n")
            fh.flush()

            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            eta = (len(items) - i) / rate if rate > 0 else float("inf")
            running_acc = sum(r["correct"] for r in records) / len(records)
            print(
                f"[gsm-hard] {i}/{len(items)}  "
                f"acc={running_acc:.3f}  "
                f"elapsed={elapsed:6.1f}s  "
                f"rate={rate:4.2f}/s  eta={eta:6.1f}s",
                flush=True,
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
    summary = {
        "metrics": metrics,
        "config": {
            "split": split,
            "n_requested": n,
            "n_evaluated": n_total,
            "seed": seed,
            "model": "agents.common.get_planner_model()",
            "num_predict": NUM_PREDICT,
            "float_rel_tol": FLOAT_REL_TOL,
            "float_abs_tol": FLOAT_ABS_TOL,
            "system_prompt": SYSTEM_PROMPT,
        },
        "runtime_seconds": time.time() - t0,
        "predictions_path": str(predictions_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print("[gsm-hard] done.", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"[gsm-hard] predictions -> {predictions_path}", flush=True)
    print(f"[gsm-hard] summary     -> {summary_path}", flush=True)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GSM-Hard against the single planner model."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of problems to evaluate (default: 100, dataset has 1319).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help=(
            "Dataset split. GSM-Hard only ships a 'train' split "
            "(1319 rows); leave as default."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for subsampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to benchmarks/results/gsm_hard/<timestamp>.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir or (
        Path("benchmarks/results/gsm_hard")
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run(n=args.n, split=args.split, seed=args.seed, output_dir=output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
