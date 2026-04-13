"""CommonGen-Hard by AgentVerse
Harder variant of CommonGen. Each instance contains 28-31 everyday concepts
(vs. 3-5 in the original CommonGen), and the task is to produce a single natural-sounding
sentence that uses ALL of them. This is a constraint-satisfaction problem heavy enough that
multi-agent planning/critique loops produce measurable gains

Usage:
    python -m benchmarks.commongen_hard_runner --n 100

Outputs:
    benchmarks/results/commongen_hard/<timestamp>/predictions.jsonl
    benchmarks/results/commongen_hard/<timestamp>/summary.json

Metric — concept coverage:
    The dataset has NO gold references; the AgentVerse paper evaluates
    with an LLM judge. To stay reproducible and cheap we score the
    fraction of requested concepts that appear in the generated
    sentence under a lenient lemma-or-inflection match (see
    ``concept_matched``). We report:
        - coverage_mean:       mean of per-item coverage
        - full_coverage_rate:  fraction of items with ALL concepts hit
    If you later want AgentVerse-style fluency judging, it can be a
    post-hoc pass over the same predictions.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

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


LABEL = "commongen-hard"
BENCHMARK_NAME = "commongen_hard"

# Not on HF
SOURCE_URL = (
    "https://raw.githubusercontent.com/OpenBMB/AgentVerse/main/"
    "data/commongen/commongen_hard.jsonl"
)

DEFAULT_CACHE = Path(os.path.expanduser("~/.cache/dynamic-multi-agent/commongen_hard.jsonl"))

SYSTEM_PROMPT = (
    "You write ONE natural English sentence that uses ALL of the given "
    "concepts in a plausible everyday scenario. Each concept must "
    "appear as a word — a direct inflection (e.g. walks/walking for "
    "walk) is fine, but do not skip concepts. Respond with ONLY the "
    "sentence. No preamble, no explanation, no quotes, no list."
)

# Cap on output tokens. One sentence is well under 200 tokens; 512 is
# a safety ceiling — CommonGen-Hard prompts are long (30 concepts), so
# some models pad the output with incidental clauses.
NUM_PREDICT = 512


def build_user_prompt(concepts: list[str]) -> str:
    return f"Concepts: {', '.join(concepts)}\nSentence:"


def clean_response(raw: str) -> str:
    """Turn a raw model response into a single sentence for scoring."""
    text = strip_think_blocks(raw).strip()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for prefix in ("Sentence:", "sentence:", "Answer:", "answer:"):
            if line.startswith(prefix):
                line = line[len(prefix) :].strip()
        if len(line) >= 2 and line[0] in {'"', "'"} and line[-1] == line[0]:
            line = line[1:-1].strip()
        return line
    return ""


# --- Concept coverage metric -------------------------------------------------


def tokenize(sentence: str) -> list[str]:
    _WORD_RE = re.compile(r"[a-z]+")
    return _WORD_RE.findall(sentence.lower())


def _normalize_concept(concept: str) -> str:
    return re.sub(r"[^a-z]", "", concept.lower())


def concept_matched(concept: str, tokens: list[str]) -> bool:
    """Lenient lemma-or-inflection match.

    Returns True if any token equals the concept, or starts with the
    concept followed by up to 4 trailing characters (covers -s, -es,
    -ed, -ing, -ies, -ied, -er). Some over-match risk is accepted for
    simplicity: 'cat' will not match 'category' (tail length 5), but
    'run' matches 'running' (tail length 4). This is a coverage
    proxy, not a linguistic analyzer; if you need stricter matching
    swap in a real stemmer (nltk Porter) later which can do "shine-shone".
    """
    c = _normalize_concept(concept)
    if not c:
        return False
    for t in tokens:
        if t == c:
            return True
        if t.startswith(c):
            tail = len(t) - len(c)
            if 0 < tail <= 4:
                return True
    return False


def score_coverage(concepts: list[str], sentence: str) -> dict:
    tokens = tokenize(sentence)
    matched = [c for c in concepts if concept_matched(c, tokens)]
    matched_set = set(matched)
    missed = [c for c in concepts if c not in matched_set]
    n = len(concepts)
    return {
        "n_concepts": n,
        "n_matched": len(matched),
        "coverage": (len(matched) / n) if n else 0.0,
        "missed": missed,
    }


# --- Data loading ------------------------------------------------------------


def ensure_dataset(cache_path: Path) -> Path:
    if cache_path.exists():
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{LABEL}] downloading {SOURCE_URL}", flush=True)
    urllib.request.urlretrieve(SOURCE_URL, cache_path)
    return cache_path


def load_commongen_hard(cache_path: Path, n: int | None, seed: int) -> list[dict]:
    items: list[dict] = []
    with cache_path.open() as fh:
        for line_num, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            concepts = row.get("concepts")
            if not isinstance(concepts, list) or not concepts:
                continue
            items.append({"example_id": line_num, "concepts": list(concepts)})
    return shuffle_and_truncate(items, seed=seed, n=n)


def run(n: int | None, seed: int, output_dir: Path, cache_path: Path) -> dict:
    cache_path = ensure_dataset(cache_path)
    items = load_commongen_hard(cache_path, n=n, seed=seed)
    avg_concepts = sum(len(x["concepts"]) for x in items) / len(items) if items else 0.0
    print(
        f"[{LABEL}] {len(items)} items  " f"(avg {avg_concepts:.1f} concepts/item)",
        flush=True,
    )

    model = get_reasoning_model(num_predict=NUM_PREDICT)

    def process_item(item: dict) -> dict:
        response = model.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=build_user_prompt(item["concepts"])),
            ]
        )
        raw = response_text(response)
        prediction = clean_response(raw)
        score = score_coverage(item["concepts"], prediction)
        return {
            "example_id": item["example_id"],
            "concepts": item["concepts"],
            "raw": raw,
            "prediction": prediction,
            "n_concepts": score["n_concepts"],
            "n_matched": score["n_matched"],
            "coverage": score["coverage"],
            "missed": score["missed"],
        }

    def running_summary(records: list[dict]) -> str:
        cov = sum(r["coverage"] for r in records) / len(records)
        return f"coverage={cov:.3f}"

    records, predictions_path, runtime = run_items(
        label=LABEL,
        items=items,
        process_item=process_item,
        running_summary=running_summary,
        output_dir=output_dir,
    )

    n_eval = len(records)
    full_coverage_hits = sum(1 for r in records if r["n_matched"] == r["n_concepts"])
    metrics = {
        "coverage_mean": (sum(r["coverage"] for r in records) / n_eval if n_eval else 0.0),
        "full_coverage_rate": (full_coverage_hits / n_eval if n_eval else 0.0),
        "n": n_eval,
    }
    config = {
        "n_requested": n,
        "n_evaluated": n_eval,
        "seed": seed,
        "source_url": SOURCE_URL,
        "cache_path": str(cache_path),
        "model": "agents.common.get_reasoning_model()",
        "num_predict": NUM_PREDICT,
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
    parser = argparse.ArgumentParser(
        description="Run CommonGen-Hard against the single planner model."
    )
    add_common_args(parser, default_n=100)
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE,
        help=f"Local cache for commongen_hard.jsonl (default: {DEFAULT_CACHE}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir or (
        Path("benchmarks/results") / BENCHMARK_NAME / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run(
        n=args.n,
        seed=args.seed,
        output_dir=output_dir,
        cache_path=args.cache_path,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
