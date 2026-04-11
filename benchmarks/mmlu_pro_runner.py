"""MMLU-Pro benchmark runner for the single planner model.

Loads TIGER-Lab/MMLU-Pro, the test split, and scores each example as
a multiple-choice question with variable option counts (A-J, though
some items have fewer). Re-uses ``agents.single_agent.answer_question``
for prompting and response format so we are actually benchmarking the
same code path the agent layer uses.

Usage:
    python -m benchmarks.mmlu_pro_runner --n 200

Outputs:
    benchmarks/results/mmlu_pro/<timestamp>/predictions.jsonl
    benchmarks/results/mmlu_pro/<timestamp>/summary.json

Metrics:
    - Overall accuracy (micro).
    - Macro-averaged accuracy across the 14 categories. Macro is the
      primary number because MMLU-Pro is heavily STEM-weighted
      (math/physics/chemistry are ~1.3k items each, business/psychology
      are a few hundred), and the micro average would otherwise be
      dominated by whichever category the model happens to be best at.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from datasets import load_dataset

from agents.single_agent import answer_question


# Match "Answer: X" anywhere in the response, capture the letter.
# Case-insensitive, allows optional whitespace / punctuation around the
# letter, and handles "**Answer:** X" style formatting some models use.
ANSWER_RE = re.compile(
    r"answer\s*[:\-]?\s*\**\s*([A-J])\b", re.IGNORECASE
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
    rng = random.Random(seed)
    rng.shuffle(items)
    if n is not None:
        items = items[:n]
    return items


def compute_metrics(records: list[dict]) -> dict:
    overall_correct = sum(1 for r in records if r["correct"])
    overall_total = len(records)
    overall_acc = overall_correct / overall_total if overall_total else 0.0

    by_category: dict[str, list[bool]] = defaultdict(list)
    for r in records:
        by_category[r["category"]].append(r["correct"])

    category_accs = {
        cat: sum(vals) / len(vals) for cat, vals in by_category.items()
    }
    macro_acc = (
        sum(category_accs.values()) / len(category_accs)
        if category_accs
        else 0.0
    )

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


# Cap on output tokens per question. MMLU-Pro with visible CoT
# typically produces 200-500 tokens; 1024 leaves ample headroom for
# the 'Answer: X' tail without letting a runaway response wedge the run.
NUM_PREDICT = 1024


def run(n: int, split: str, seed: int, output_dir: Path) -> dict:
    print(f"[mmlu-pro] loading dataset split={split} ...", flush=True)
    items = load_mmlu_pro(split=split, n=n, seed=seed)
    print(f"[mmlu-pro] {len(items)} questions", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"

    records: list[dict] = []

    t0 = time.time()
    with predictions_path.open("w") as fh:
        for i, item in enumerate(items, start=1):
            valid_letters = "ABCDEFGHIJ"[: len(item["options"])]

            raw = answer_question(
                item["question"], item["options"], num_predict=NUM_PREDICT
            )
            if not isinstance(raw, str):
                raw = str(raw)

            prediction = extract_letter(raw, valid_letters)
            correct = prediction == item["answer"]

            record = {
                "question_id": item["question_id"],
                "category": item["category"],
                "num_options": len(item["options"]),
                "ground_truth": item["answer"],
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
                f"[mmlu-pro] {i}/{len(items)}  "
                f"acc={running_acc:.3f}  "
                f"elapsed={elapsed:6.1f}s  "
                f"rate={rate:4.2f}/s  eta={eta:6.1f}s",
                flush=True,
            )

    metrics = compute_metrics(records)
    summary = {
        "metrics": metrics,
        "config": {
            "split": split,
            "n_requested": n,
            "n_evaluated": len(items),
            "seed": seed,
            "agent": "agents.single_agent.answer_question",
            "num_predict": NUM_PREDICT,
        },
        "runtime_seconds": time.time() - t0,
        "predictions_path": str(predictions_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print("[mmlu-pro] done.", flush=True)
    print(
        json.dumps(
            {
                k: metrics[k]
                for k in (
                    "accuracy_micro",
                    "accuracy_macro",
                    "n",
                    "n_correct",
                    "n_unparseable",
                )
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"[mmlu-pro] predictions -> {predictions_path}", flush=True)
    print(f"[mmlu-pro] summary     -> {summary_path}", flush=True)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MMLU-Pro against the single planner model."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of questions to evaluate (default: 10).",
    )
    parser.add_argument(
        "--split", default="test", help="Dataset split (default: test)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling the subset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to benchmarks/results/mmlu_pro/<timestamp>.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir or (
        Path("benchmarks/results/mmlu_pro")
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run(n=args.n, split=args.split, seed=args.seed, output_dir=output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
