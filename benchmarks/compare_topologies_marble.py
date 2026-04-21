"""Cross-topology comparison on MARBLE-Research.

Runs four topologies — single_agent, chain, tree, dynamic — against the same
MARBLE-Research items and judges each output with the verbatim MARBLE judge
prompt (innovation / safety / feasibility on 1-5). Everything else mirrors
`benchmarks.marble_research_runner`; the only thing varying across rows is
the topology, so differences in mean scores are attributable to structure.

Rate limiting:
    Gemini free tier for the Gemma model caps us at 15 calls/minute. A single
    run through chain/tree/dynamic uses ~4 generator calls and takes 20-30s,
    plus one judge call. We sleep between each topology-item pair to keep
    the effective rate well below the ceiling. The sleep is configurable via
    --sleep-seconds; default is 20 for headroom.

Usage:
    python -m benchmarks.compare_topologies_marble --n 5

Outputs:
    benchmarks/results/compare_marble/<timestamp>/predictions.jsonl
    benchmarks/results/compare_marble/<timestamp>/summary.json

Each predictions.jsonl row is one (task_id, topology) pair so per-topology
filtering stays trivial; summary.json aggregates per-axis + overall means
per topology alongside per-topology runtimes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage

from agents.common import get_reasoning_model
from agents import single_agent, chain, tree, dynamic
from benchmarks.common import response_text, add_common_args
from benchmarks.marble_research_runner import (
    DEFAULT_CACHE,
    JUDGE_PROMPT,
    _AXES,
    ensure_dataset,
    load_marble_research,
    parse_judge_scores,
)


LABEL = "compare-marble"
BENCHMARK_NAME = "compare_marble"

TOPOLOGIES: dict[str, callable] = {
    "single_agent": single_agent.run,
    "chain": chain.run,
    "tree": tree.run,
    "dynamic": dynamic.run,
}


def _mean(rows: list[dict], key: str) -> float:
    rated = [r for r in rows if r[key] is not None]
    return (sum(r[key] for r in rated) / len(rated)) if rated else 0.0


def run(
    n: int | None,
    seed: int,
    output_dir: Path,
    cache_path: Path,
    sleep_seconds: float,
) -> dict:
    cache_path = ensure_dataset(cache_path)
    items = load_marble_research(cache_path, n=n, seed=seed)
    print(f"[{LABEL}] {len(items)} tasks x {len(TOPOLOGIES)} topologies", flush=True)

    judge = get_reasoning_model()

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"

    records: list[dict] = []
    per_topology_runtime: dict[str, float] = {name: 0.0 for name in TOPOLOGIES}

    t0 = time.time()
    with predictions_path.open("w") as fh:
        for i, item in enumerate(items, start=1):
            for topology_name, topology_run in TOPOLOGIES.items():
                print(
                    f"\n===== item {i}/{len(items)} (task_id={item['task_id']}) "
                    f"topology={topology_name} =====",
                    flush=True,
                )
                t_topology = time.time()
                raw = topology_run(item["task_content"])
                # chain/tree/dynamic return str; single_agent returns str too, but
                # guard in case any returns a message-like object in the future.
                raw = response_text(raw)

                judge_msg = JUDGE_PROMPT.format(task=item["task_content"], result=raw)
                judge_raw = response_text(
                    judge.invoke([HumanMessage(content=judge_msg)])
                )
                scores = parse_judge_scores(judge_raw)

                topology_elapsed = time.time() - t_topology
                per_topology_runtime[topology_name] += topology_elapsed

                overall = (
                    sum(scores[a] for a in _AXES) / len(_AXES) if scores else None
                )
                record = {
                    "task_id": item["task_id"],
                    "topology": topology_name,
                    "raw": raw,
                    "judge_raw": judge_raw,
                    "innovation": scores["innovation"] if scores else None,
                    "safety": scores["safety"] if scores else None,
                    "feasibility": scores["feasibility"] if scores else None,
                    "overall": overall,
                    "unparseable": scores is None,
                    "topology_runtime_seconds": topology_elapsed,
                }
                records.append(record)
                fh.write(json.dumps(record) + "\n")
                fh.flush()

                print(
                    f"[{LABEL}] {topology_name}: overall={overall} "
                    f"({topology_elapsed:.1f}s)",
                    flush=True,
                )

                # Gemma free tier: 15 calls/min. A full topology pass (generator
                # + judge) bursts several calls in quick succession; this sleep
                # amortizes the burst so we stay under the ceiling on average.
                # Skip it after the last topology of the last item so we don't
                # sit idle at the very end.
                is_last = (
                    i == len(items)
                    and topology_name == list(TOPOLOGIES.keys())[-1]
                )
                if not is_last and sleep_seconds > 0:
                    print(f"[{LABEL}] sleeping {sleep_seconds}s for rate limit", flush=True)
                    time.sleep(sleep_seconds)

    runtime = time.time() - t0

    per_topology_metrics: dict[str, dict] = {}
    for name in TOPOLOGIES:
        rows = [r for r in records if r["topology"] == name]
        rated = [r for r in rows if not r["unparseable"]]
        per_topology_metrics[name] = {
            "innovation_mean": _mean(rated, "innovation"),
            "safety_mean": _mean(rated, "safety"),
            "feasibility_mean": _mean(rated, "feasibility"),
            "overall_mean": _mean(rated, "overall"),
            "n": len(rows),
            "n_rated": len(rated),
            "n_unparseable": len(rows) - len(rated),
            "runtime_seconds": per_topology_runtime[name],
        }

    summary = {
        "per_topology": per_topology_metrics,
        "config": {
            "n_requested": n,
            "n_evaluated": len(items),
            "seed": seed,
            "topologies": list(TOPOLOGIES.keys()),
            "source_url": "benchmarks.marble_research_runner.SOURCE_URL",
            "cache_path": str(cache_path),
            "generator_model": "agents.common.get_reasoning_model()",
            "judge_model": "agents.common.get_reasoning_model()",
            "judge_prompt": JUDGE_PROMPT,
            "sleep_seconds": sleep_seconds,
        },
        "runtime_seconds": runtime,
        "predictions_path": str(predictions_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n[{LABEL}] done.", flush=True)
    print(json.dumps(per_topology_metrics, indent=2), flush=True)
    print(f"[{LABEL}] predictions -> {predictions_path}", flush=True)
    print(f"[{LABEL}] summary     -> {summary_path}", flush=True)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare single_agent / chain / tree / dynamic on MARBLE-Research."
    )
    add_common_args(parser, default_n=5)
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE,
        help=f"Local cache for research_main.jsonl (default: {DEFAULT_CACHE}).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=20.0,
        help=(
            "Seconds to sleep between each topology-item pair to stay under "
            "the 15 calls/min Gemma free-tier ceiling (default: 20)."
        ),
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
        sleep_seconds=args.sleep_seconds,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
