"""CommonGen benchmark runner for the single planner model.

Loads the CommonGen validation split from HuggingFace
(`allenai/common_gen`), groups rows by `concept_set_idx` so each unique
concept set is generated once and scored against ALL its human
references, invokes the planner model from ``agents.common``, and
reports corpus BLEU-4 (sacrebleu) and mean ROUGE-L F1 (rouge_score).

Typical use from the repo root::

    python -m benchmarks.commongen_runner --n 200

Outputs:
    benchmarks/results/commongen/<timestamp>/predictions.jsonl
    benchmarks/results/commongen/<timestamp>/summary.json

Notes on metrics:
    - sacrebleu's BLEU uses its own tokenizer, so absolute numbers are
      not directly comparable to the COCO-style BLEU-4 reported in the
      original CommonGen paper. It is, however, stable and reproducible
      across runs and suitable for comparing this model to itself or to
      other entries in this pipeline.
    - ROUGE-L is computed per example against each reference, taking
      the max, then averaged across examples (rouge_score has no
      native multi-reference aggregation).
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
from langchain_core.messages import HumanMessage, SystemMessage
from rouge_score import rouge_scorer
import sacrebleu

from agents.common import get_planner_model


SYSTEM_PROMPT = (
    "You write ONE natural English sentence that uses ALL of the given "
    "concepts in a plausible everyday scenario. Respond with ONLY the "
    "sentence. No preamble, no explanation, no quotes, no list."
)

# Strip <think>...</think> blocks emitted by reasoning-tuned models
# (e.g. qwen3 family) before any whitespace/line parsing.
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def build_user_prompt(concepts: list[str]) -> str:
    return f"Concepts: {', '.join(concepts)}\nSentence:"


def clean_response(raw: str) -> str:
    """Turn a raw model response into a single sentence suitable for scoring."""
    text = THINK_BLOCK_RE.sub("", raw or "").strip()
    # Take the first non-empty line; models sometimes add trailing remarks.
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Drop common prefixes the model may echo back.
        for prefix in ("Sentence:", "sentence:", "Answer:", "answer:"):
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
        # Strip surrounding quotes.
        if len(line) >= 2 and line[0] in {'"', "'"} and line[-1] == line[0]:
            line = line[1:-1].strip()
        return line
    return ""


def load_commongen_grouped(
    split: str, n: int | None, seed: int
) -> list[dict]:
    """Load CommonGen and group multi-reference rows by concept set.

    Returns a list of dicts ``{concept_set_idx, concepts, references}``,
    shuffled deterministically, truncated to ``n`` if provided.
    """
    ds = load_dataset("allenai/common_gen", split=split)
    grouped: dict[int, dict] = {}
    for row in ds:
        idx = row["concept_set_idx"]
        entry = grouped.get(idx)
        if entry is None:
            grouped[idx] = {
                "concept_set_idx": idx,
                "concepts": list(row["concepts"]),
                "references": [row["target"]] if row["target"] else [],
            }
        else:
            if row["target"]:
                entry["references"].append(row["target"])

    items = list(grouped.values())
    rng = random.Random(seed)
    rng.shuffle(items)
    if n is not None:
        items = items[:n]
    return items


def compute_metrics(
    hypotheses: list[str], references_per_item: list[list[str]]
) -> dict:
    """Corpus BLEU-4 (sacrebleu, multi-ref) + mean max ROUGE-L F1."""
    # sacrebleu expects references in a transposed layout of shape
    # [max_refs][num_examples], padded with empty strings where an
    # example has fewer references than the max.
    max_refs = max(len(r) for r in references_per_item)
    refs_transposed: list[list[str]] = [[] for _ in range(max_refs)]
    for refs in references_per_item:
        for i in range(max_refs):
            refs_transposed[i].append(refs[i] if i < len(refs) else "")
    bleu = sacrebleu.corpus_bleu(hypotheses, refs_transposed)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l_scores: list[float] = []
    for hyp, refs in zip(hypotheses, references_per_item):
        best = 0.0
        for ref in refs:
            score = scorer.score(ref, hyp)["rougeL"].fmeasure
            if score > best:
                best = score
        rouge_l_scores.append(best)
    mean_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0

    return {
        "bleu4": bleu.score,
        "rouge_l_f1_mean": mean_rouge_l,
        "n": len(hypotheses),
    }


def run(
    n: int,
    split: str,
    seed: int,
    output_dir: Path,
) -> dict:
    print(f"[commongen] loading dataset split={split} ...", flush=True)
    items = load_commongen_grouped(split=split, n=n, seed=seed)
    print(
        f"[commongen] {len(items)} concept sets "
        f"(avg {sum(len(x['references']) for x in items) / len(items):.2f} refs/set)",
        flush=True,
    )

    model = get_planner_model()

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"

    hypotheses: list[str] = []
    references_per_item: list[list[str]] = []

    t0 = time.time()
    with predictions_path.open("w") as fh:
        for i, item in enumerate(items, start=1):
            prompt = build_user_prompt(item["concepts"])
            response = model.invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
            )
            raw = response.content if isinstance(response.content, str) else str(response.content)
            prediction = clean_response(raw)

            hypotheses.append(prediction)
            references_per_item.append(item["references"])

            fh.write(
                json.dumps(
                    {
                        "concept_set_idx": item["concept_set_idx"],
                        "concepts": item["concepts"],
                        "references": item["references"],
                        "raw": raw,
                        "prediction": prediction,
                    }
                )
                + "\n"
            )
            fh.flush()

            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            eta = (len(items) - i) / rate if rate > 0 else float("inf")
            print(
                f"[commongen] {i}/{len(items)}  "
                f"elapsed={elapsed:6.1f}s  rate={rate:4.2f}/s  "
                f"eta={eta:6.1f}s",
                flush=True,
            )

    metrics = compute_metrics(hypotheses, references_per_item)
    summary = {
        "metrics": metrics,
        "config": {
            "split": split,
            "n_requested": n,
            "n_evaluated": len(items),
            "seed": seed,
            "system_prompt": SYSTEM_PROMPT,
            "model": "agents.common.get_planner_model()",
        },
        "runtime_seconds": time.time() - t0,
        "predictions_path": str(predictions_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print("[commongen] done.", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"[commongen] predictions -> {predictions_path}", flush=True)
    print(f"[commongen] summary     -> {summary_path}", flush=True)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CommonGen against the single planner model.")
    parser.add_argument("--n", type=int, default=200, help="Number of unique concept sets to evaluate (default: 200).")
    parser.add_argument("--split", default="validation", help="Dataset split (default: validation).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for selecting the subset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to benchmarks/results/commongen/<timestamp>.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir or (
        Path("benchmarks/results/commongen") / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run(n=args.n, split=args.split, seed=args.seed, output_dir=output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
