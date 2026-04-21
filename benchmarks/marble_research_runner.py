"""MultiAgentBench / MARBLE — Research scenario.

100 tasks in total.
Each item is a multi-paragraph introduction from a real ML/AI paper paired
with the MARBLE instruction to produce a new research idea in the '5q' format
(problem, importance, difficulty, prior-art gap, methodology). The dataset
ships no gold reference; MARBLE's official evaluator is an LLM judge scoring
three axes — innovation, safety, feasibility — each on a 1-5 scale. We
mirror that judge prompt verbatim so results stay comparable with the paper.

Usage:
    python -m benchmarks.marble_research_runner --n 30

Outputs:
    benchmarks/results/marble_research/<timestamp>/predictions.jsonl
    benchmarks/results/marble_research/<timestamp>/summary.json

Metric:
    Per-item: innovation, safety, feasibility (1-5) and their unweighted
    mean ("overall"). Aggregated: mean per axis + overall. Unparseable
    judge outputs are excluded from means and counted in n_unparseable.

Caveat:
    LLM-judge noise is real and when judge == generator, absolute scores
    drift high. Use this to compare topologies on the SAME judge model;
    do not compare absolute numbers across runs that swap the judge.
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

from langchain_core.messages import HumanMessage

from agents.common import get_reasoning_model, get_judge_model
from benchmarks.common import (
    add_common_args,
    response_text,
    run_items,
    shuffle_and_truncate,
    strip_think_blocks,
    write_summary,
)


LABEL = "marble-research"
BENCHMARK_NAME = "marble_research"

# Not on HF — pulled from the official MARBLE repo. 100 tasks.
SOURCE_URL = (
    "https://raw.githubusercontent.com/MultiagentBench/MARBLE/main/"
    "multiagentbench/research/research_main.jsonl"
)
DEFAULT_CACHE = Path(os.path.expanduser("~/.cache/dynamic-multi-agent/marble_research_main.jsonl"))

# Verbatim from MARBLE's marble/evaluator/evaluator_prompts.json, key
# research.task_evaluation. Double braces are the format-string escape
# for the JSON output block; the outer `.format(task=..., result=...)`
# fills the two single-brace placeholders.
JUDGE_PROMPT = """
[Context]
Task:
{task}

Result:
{result}

[System]
Evaluate the research idea on three aspects using a 5-point scale:
- Innovation (1-5): Novelty and advancement
- Safety (1-5): Ethical and legal considerations
- Feasibility (1-5): Practicality and achievability

Scale:
5: Excellent
4: Good
3: Average
2: Below Average
1: Poor

[Question]
Provide your ratings in this exact JSON format:
{{
  "innovation": NUMBER,
  "safety": NUMBER,
  "feasibility": NUMBER
}}

Do not include any other text or explanation.
""".strip()


_AXES = ("innovation", "safety", "feasibility")

# Matches a flat JSON object (no nesting). MARBLE's judge prompt asks for
# a flat {innovation, safety, feasibility} dict, so nested braces from a
# chatty judge should be skipped rather than partially matched.
_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_judge_scores(raw: str) -> dict[str, int] | None:
    """Extract the three 1-5 ratings from a judge response.

    Scan JSON-looking blocks back-to-front so that if the judge restates
    an intermediate object before committing to a final one, the last
    block wins. Reject anything that isn't all three axes in range.
    """
    text = strip_think_blocks(raw)
    for match in reversed(list(_JSON_BLOCK_RE.finditer(text))):
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        out: dict[str, int] = {}
        ok = True
        for axis in _AXES:
            v = obj.get(axis)
            # bool is a subclass of int — reject explicitly.
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                ok = False
                break
            if not (1 <= v <= 5):
                ok = False
                break
            out[axis] = int(v)
        if ok:
            return out
    return None


def ensure_dataset(cache_path: Path) -> Path:
    if cache_path.exists():
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{LABEL}] downloading {SOURCE_URL}", flush=True)
    urllib.request.urlretrieve(SOURCE_URL, cache_path)
    return cache_path


def load_marble_research(cache_path: Path, n: int | None, seed: int) -> list[dict]:
    items: list[dict] = []
    with cache_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            content = (row.get("task") or {}).get("content")
            if not content:
                continue
            items.append(
                {
                    "task_id": row.get("task_id"),
                    "task_content": content,
                }
            )
    return shuffle_and_truncate(items, seed=seed, n=n)


def run(n: int | None, seed: int, output_dir: Path, cache_path: Path) -> dict:
    cache_path = ensure_dataset(cache_path)
    items = load_marble_research(cache_path, n=n, seed=seed)
    print(f"[{LABEL}] {len(items)} tasks", flush=True)

    generator = get_reasoning_model()
    judge = get_judge_model()

    def process_item(item: dict) -> dict:
        gen_response = generator.invoke([HumanMessage(content=item["task_content"])])
        raw = response_text(gen_response)

        judge_msg = JUDGE_PROMPT.format(task=item["task_content"], result=raw)
        judge_response = judge.invoke([HumanMessage(content=judge_msg)])
        judge_raw = response_text(judge_response)
        scores = parse_judge_scores(judge_raw)

        overall = sum(scores[a] for a in _AXES) / len(_AXES) if scores else None
        return {
            "task_id": item["task_id"],
            "raw": raw,
            "judge_raw": judge_raw,
            "innovation": scores["innovation"] if scores else None,
            "safety": scores["safety"] if scores else None,
            "feasibility": scores["feasibility"] if scores else None,
            "overall": overall,
            "unparseable": scores is None,
        }

    def running_summary(records: list[dict]) -> str:
        rated = [r for r in records if r["overall"] is not None]
        if not rated:
            return "overall=N/A"
        return f"overall={sum(r['overall'] for r in rated) / len(rated):.2f}"

    records, predictions_path, runtime = run_items(
        label=LABEL,
        items=items,
        process_item=process_item,
        running_summary=running_summary,
        output_dir=output_dir,
    )

    n_eval = len(records)
    rated = [r for r in records if not r["unparseable"]]

    def _mean(key: str) -> float:
        return (sum(r[key] for r in rated) / len(rated)) if rated else 0.0

    metrics = {
        "innovation_mean": _mean("innovation"),
        "safety_mean": _mean("safety"),
        "feasibility_mean": _mean("feasibility"),
        "overall_mean": _mean("overall"),
        "n": n_eval,
        "n_rated": len(rated),
        "n_unparseable": n_eval - len(rated),
    }
    config = {
        "n_requested": n,
        "n_evaluated": n_eval,
        "seed": seed,
        "source_url": SOURCE_URL,
        "cache_path": str(cache_path),
        "generator_model": "agents.common.get_reasoning_model()",
        "judge_model": "agents.common.get_reasoning_model()",
        "judge_prompt": JUDGE_PROMPT,
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
        description="Run MARBLE-Research against the single planner model, judged by an LLM."
    )
    add_common_args(parser, default_n=30)
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE,
        help=f"Local cache for research_main.jsonl (default: {DEFAULT_CACHE}).",
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
