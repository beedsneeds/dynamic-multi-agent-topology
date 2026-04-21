"""TIGER-Lab/MMLU-Pro
Multiple-choice question with variable option counts (A-J, though some items have fewer)
Re-uses ``agents.single_agent.answer_question``
for prompting and response format so we are actually benchmarking the
same code path the agent layer uses.

Usage:
    python -m benchmarks.mmlu_pro_runner --n 200

Outputs:
    benchmarks/results/mmlu_pro/<timestamp>/predictions.jsonl
    benchmarks/results/mmlu_pro/<timestamp>/summary.json

Metrics:
    - Overall accuracy (micro)
    # TODO does macro matter for us?
    - Macro-averaged accuracy across the 14 categories. Macro is the
      primary number because MMLU-Pro is heavily STEM-weighted
      (math/physics/chemistry are ~1.3k items each, business/psychology
      are a few hundred), and the micro average would otherwise be
      dominated by whichever category the model happens to be best at.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from datasets import load_dataset

from agents.single_agent import answer_question
from benchmarks.common import (
    add_common_args,
    response_text,
    run_items,
    shuffle_and_truncate,
    write_summary,
)


LABEL = "mmlu-pro"
BENCHMARK_NAME = "mmlu_pro"

SYSTEM_PROMPT = (
    "You are answering a multiple-choice question. "
    "Reason step by step, then end your response with exactly: 'Answer: X' "
    "where X is one of the option letters shown below."
)


def extract_letter(response: str, valid_letters: str) -> str | None:
    """Pull the final answer letter out of a model response.

    Strategy:
      1. Look for an explicit 'Answer: X' in the response (last match
         wins — models sometimes restate earlier guesses before their
         final answer).
      2. Fall back to the last occurrence of a bare valid letter.
      3. Return None if nothing parseable is found; the item is scored
         as wrong and the raw response is preserved for inspection.
    """
    # Capture the letter from a match of "Answer: X" anywhere in the response.
    # Case-insensitive, allows optional whitespace / punctuation around the
    # letter, and handles "**Answer:** X" style formatting some models use.
    ANSWER_RE = re.compile(r"answer\s*[:\-]?\s*\**\s*([A-J])\b", re.IGNORECASE)

    if not response:
        return None

    matches = list(ANSWER_RE.finditer(response))
    if matches:
        letter = matches[-1].group(1).upper()
        if letter in valid_letters:
            return letter

    # Fallback: find standalone letters that are in the valid set.
    standalone = re.findall(r"\b([A-J])\b", response)
    for letter in reversed(standalone):
        letter = letter.upper()
        if letter in valid_letters:
            return letter

    return None


def load_mmlu_pro(split: str, n: int | None, seed: int) -> list[dict]:
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
    items = [
        {
            "question_id": row["question_id"],
            "question": row["question"],
            "options": list(row["options"]),
            "answer": row["answer"],
            "category": row["category"],
        }
        for row in ds
    ]
    return shuffle_and_truncate(items, seed=seed, n=n)


def compute_metrics(records: list[dict]) -> dict:
    overall_correct = sum(1 for r in records if r["correct"])
    overall_total = len(records)
    overall_acc = overall_correct / overall_total if overall_total else 0.0

    by_category: dict[str, list[bool]] = defaultdict(list)
    for r in records:
        by_category[r["category"]].append(r["correct"])

    category_accs = {cat: sum(vals) / len(vals) for cat, vals in by_category.items()}
    macro_acc = sum(category_accs.values()) / len(category_accs) if category_accs else 0.0

    return {
        "accuracy_micro": overall_acc,
        "accuracy_macro": macro_acc,
        "n": overall_total,
        "n_correct": overall_correct,
        "n_unparseable": sum(1 for r in records if r["prediction"] is None),
        "per_category": {
            cat: {"accuracy": acc, "n": len(by_category[cat])}
            for cat, acc in sorted(category_accs.items())
        },
    }


def run(n: int, split: str, seed: int, output_dir) -> dict:
    print(f"[{LABEL}] loading dataset split={split} ...", flush=True)
    items = load_mmlu_pro(split=split, n=n, seed=seed)
    print(f"[{LABEL}] {len(items)} questions", flush=True)

    def process_item(item: dict) -> dict:
        valid_letters = "ABCDEFGHIJ"[: len(item["options"])]
        raw = response_text(
            answer_question(
                item["question"],
                item["options"],
                system_prompt=SYSTEM_PROMPT,
            )
        )
        prediction = extract_letter(raw, valid_letters)
        return {
            "question_id": item["question_id"],
            "category": item["category"],
            "num_options": len(item["options"]),
            "ground_truth": item["answer"],
            "prediction": prediction,
            "correct": prediction == item["answer"],
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

    metrics = compute_metrics(records)
    config = {
        "split": split,
        "n_requested": n,
        "n_evaluated": len(items),
        "seed": seed,
        "agent": "agents.single_agent.answer_question",
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
    parser = argparse.ArgumentParser(description="Run MMLU-Pro against the single planner model.")
    add_common_args(parser, default_n=10)
    parser.add_argument("--split", default="test", help="Dataset split (default: test).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir or (
        Path("benchmarks/results") / BENCHMARK_NAME / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run(n=args.n, split=args.split, seed=args.seed, output_dir=output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
