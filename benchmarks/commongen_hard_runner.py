"""CommonGen-Hard benchmark runner for the single planner model.

CommonGen-Hard is the harder variant of CommonGen introduced by the
AgentVerse paper (Chen et al. 2023). Each instance contains 28-31
everyday concepts (vs. 3-5 in the original CommonGen), and the task
is to produce a single natural-sounding sentence that uses ALL of
them. This is a constraint-satisfaction problem heavy enough that
multi-agent planning/critique loops produce measurable gains — the
original CommonGen is saturated on this axis.

Data source:
    The only public release of CommonGen-Hard is the JSONL file in
    OpenBMB/AgentVerse's GitHub repo
    (data/commongen/commongen_hard.jsonl, 339 examples). This runner
    downloads and caches it on first use under
    ~/.cache/dynamic-multi-agent/ .

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
import random
import re
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from agents.common import get_planner_model


SOURCE_URL = (
    "https://raw.githubusercontent.com/OpenBMB/AgentVerse/main/"
    "data/commongen/commongen_hard.jsonl"
)

DEFAULT_CACHE = Path(
    os.path.expanduser("~/.cache/dynamic-multi-agent/commongen_hard.jsonl")
)

SYSTEM_PROMPT = (
    "You write ONE natural English sentence that uses ALL of the given "
    "concepts in a plausible everyday scenario. Each concept must "
    "appear as a word — a direct inflection (e.g. walks/walking for "
    "walk) is fine, but do not skip concepts. Respond with ONLY the "
    "sentence. No preamble, no explanation, no quotes, no list."
)

# Strip <think>...</think> blocks emitted by reasoning-tuned models.
# Defensive; get_planner_model() sets reasoning=False.
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Cap on output tokens. One sentence is well under 200 tokens; 512 is
# a safety ceiling — CommonGen-Hard prompts are long (30 concepts), so
# some models pad the output with incidental clauses.
NUM_PREDICT = 512


def build_user_prompt(concepts: list[str]) -> str:
    return f"Concepts: {', '.join(concepts)}\nSentence:"


def clean_response(raw: str) -> str:
    """Turn a raw model response into a single sentence for scoring."""
    text = THINK_BLOCK_RE.sub("", raw or "").strip()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for prefix in ("Sentence:", "sentence:", "Answer:", "answer:"):
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
        if len(line) >= 2 and line[0] in {'"', "'"} and line[-1] == line[0]:
            line = line[1:-1].strip()
        return line
    return ""


# --- Concept coverage metric -------------------------------------------------

_WORD_RE = re.compile(r"[a-z]+")


def tokenize(sentence: str) -> list[str]:
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
    swap in a real stemmer (nltk Porter) later.
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
    print(f"[commongen-hard] downloading {SOURCE_URL}", flush=True)
    urllib.request.urlretrieve(SOURCE_URL, cache_path)
    return cache_path


def load_commongen_hard(
    cache_path: Path, n: int | None, seed: int
) -> list[dict]:
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
    rng = random.Random(seed)
    rng.shuffle(items)
    if n is not None:
        items = items[:n]
    return items


def run(n: int | None, seed: int, output_dir: Path, cache_path: Path) -> dict:
    cache_path = ensure_dataset(cache_path)
    items = load_commongen_hard(cache_path, n=n, seed=seed)
    avg_concepts = (
        sum(len(x["concepts"]) for x in items) / len(items) if items else 0.0
    )
    print(
        f"[commongen-hard] {len(items)} items  "
        f"(avg {avg_concepts:.1f} concepts/item)",
        flush=True,
    )

    model = get_planner_model(num_predict=NUM_PREDICT)

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"

    per_item_coverage: list[float] = []
    full_coverage_hits = 0
    t0 = time.time()

    with predictions_path.open("w") as fh:
        for i, item in enumerate(items, start=1):
            response = model.invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=build_user_prompt(item["concepts"])),
                ]
            )
            raw = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
            prediction = clean_response(raw)
            score = score_coverage(item["concepts"], prediction)

            per_item_coverage.append(score["coverage"])
            if score["n_matched"] == score["n_concepts"]:
                full_coverage_hits += 1

            fh.write(
                json.dumps(
                    {
                        "example_id": item["example_id"],
                        "concepts": item["concepts"],
                        "raw": raw,
                        "prediction": prediction,
                        "n_concepts": score["n_concepts"],
                        "n_matched": score["n_matched"],
                        "coverage": score["coverage"],
                        "missed": score["missed"],
                    }
                )
                + "\n"
            )
            fh.flush()

            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            eta = (len(items) - i) / rate if rate > 0 else float("inf")
            running_cov = sum(per_item_coverage) / len(per_item_coverage)
            print(
                f"[commongen-hard] {i}/{len(items)}  "
                f"coverage={running_cov:.3f}  "
                f"elapsed={elapsed:6.1f}s  "
                f"rate={rate:4.2f}/s  eta={eta:6.1f}s",
                flush=True,
            )

    n_eval = len(per_item_coverage)
    metrics = {
        "coverage_mean": (
            sum(per_item_coverage) / n_eval if n_eval else 0.0
        ),
        "full_coverage_rate": (
            full_coverage_hits / n_eval if n_eval else 0.0
        ),
        "n": n_eval,
    }
    summary = {
        "metrics": metrics,
        "config": {
            "n_requested": n,
            "n_evaluated": n_eval,
            "seed": seed,
            "source_url": SOURCE_URL,
            "cache_path": str(cache_path),
            "model": "agents.common.get_planner_model()",
            "num_predict": NUM_PREDICT,
            "system_prompt": SYSTEM_PROMPT,
        },
        "runtime_seconds": time.time() - t0,
        "predictions_path": str(predictions_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print("[commongen-hard] done.", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"[commongen-hard] predictions -> {predictions_path}", flush=True)
    print(f"[commongen-hard] summary     -> {summary_path}", flush=True)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CommonGen-Hard against the single planner model."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of items to evaluate (default: 100, dataset has 339).",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for subsampling."
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE,
        help=f"Local cache for commongen_hard.jsonl (default: {DEFAULT_CACHE}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to benchmarks/results/commongen_hard/<timestamp>.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir or (
        Path("benchmarks/results/commongen_hard")
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run(
        n=args.n,
        seed=args.seed,
        output_dir=output_dir,
        cache_path=args.cache_path,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
