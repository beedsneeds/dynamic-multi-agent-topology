"""Cross-topology comparison on GSM-Hard.

Runs four topologies — single_agent, chain, tree, dynamic — against the same
GSM-Hard items. Accuracy is measured by numeric match (with tolerance) against
the target.

Usage:
    python -m benchmarks.compare_topologies_gsm_hard --n 10

Outputs:
    benchmarks/results/compare_gsm_hard/<timestamp>/predictions.jsonl
    benchmarks/results/compare_gsm_hard/<timestamp>/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from agents.common import usage_tracker

from agents import single_agent, chain, tree, dynamic
from benchmarks.common import response_text, add_common_args
from benchmarks.gsm_hard_runner import (
    SYNTH_SUFFIX,
    load_gsm_hard,
    extract_number,
    is_correct,
)

LABEL = "compare-gsm-hard"
BENCHMARK_NAME = "compare_gsm_hard"

TOPOLOGIES: dict[str, callable] = {
    "single_agent": single_agent.run,
    "chain": chain.run,
    "tree": tree.run,
    "dynamic": dynamic.run,
}


def run(
    n: int | None,
    split: str,
    seed: int,
    output_dir: Path,
    sleep_seconds: float,
) -> dict:
    print(f"[{LABEL}] loading dataset split={split} ...", flush=True)
    items = load_gsm_hard(split=split, n=n, seed=seed)
    print(f"[{LABEL}] {len(items)} tasks x {len(TOPOLOGIES)} topologies", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"

    records: list[dict] = []
    per_topology_runtime: dict[str, float] = {name: 0.0 for name in TOPOLOGIES}

    t0 = time.time()
    with predictions_path.open("w") as fh:
        for i, item in enumerate(items, start=1):
            for topology_name, topology_run in TOPOLOGIES.items():
                print(
                    f"\n===== item {i}/{len(items)} topology={topology_name} =====",
                    flush=True,
                )
                usage_tracker.reset()
                t_topology = time.time()

                # All topologies share the same run(query, synth_suffix) signature
                raw = topology_run(item["input"], synth_suffix=SYNTH_SUFFIX)
                raw = response_text(raw)

                prediction = extract_number(raw)
                correct = is_correct(prediction, item["target"])

                topology_elapsed = time.time() - t_topology
                per_topology_runtime[topology_name] += topology_elapsed
                usage = usage_tracker.get_usage()

                record = {
                    "input": item["input"],
                    "target": item["target"],
                    "topology": topology_name,
                    "raw": raw,
                    "prediction": prediction,
                    "correct": correct,
                    "topology_runtime_seconds": topology_elapsed,
                    "usage": usage,
                }
                records.append(record)
                fh.write(json.dumps(record) + "\n")
                fh.flush()

                print(
                    f"[{LABEL}] {topology_name}: correct={correct} " f"({topology_elapsed:.1f}s)",
                    f"tokens: in={usage['input_tokens']} out={usage['output_tokens']}",
                    flush=True,
                )

                # Rate limiting for Gemini/Gemma free tier
                is_last = i == len(items) and topology_name == list(TOPOLOGIES.keys())[-1]
                if not is_last and sleep_seconds > 0:
                    print(f"[{LABEL}] sleeping {sleep_seconds}s for rate limit", flush=True)
                    time.sleep(sleep_seconds)

    runtime = time.time() - t0

    per_topology_metrics: dict[str, dict] = {}
    for name in TOPOLOGIES:
        rows = [r for r in records if r["topology"] == name]
        n_total = len(rows)
        n_correct = sum(1 for r in rows if r["correct"])

        total_input = sum(r["usage"]["input_tokens"] for r in rows)
        total_output = sum(r["usage"]["output_tokens"] for r in rows)

        per_topology_metrics[name] = {
            "accuracy": (n_correct / n_total) if n_total else 0.0,
            "n": n_total,
            "n_correct": n_correct,
            "n_unparseable": sum(1 for r in rows if r["prediction"] is None),
            "runtime_seconds": per_topology_runtime[name],
            "avg_runtime": per_topology_runtime[name] / n_total if n_total else 0.0,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "avg_input_tokens": total_input / n_total if n_total else 0.0,
            "avg_output_tokens": total_output / n_total if n_total else 0.0,
        }

    summary = {
        "per_topology": per_topology_metrics,
        "config": {
            "split": split,
            "n_requested": n,
            "n_evaluated": len(items),
            "seed": seed,
            "topologies": list(TOPOLOGIES.keys()),
            "generator_model": "agents.common.get_reasoning_model()",
            "sleep_seconds": sleep_seconds,
        },
        "runtime_seconds": runtime,
        "predictions_path": str(predictions_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n[{LABEL}] done.", flush=True)
    print(json.dumps(per_topology_metrics, indent=2), flush=True)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare single_agent / chain / tree / dynamic on GSM-Hard."
    )
    add_common_args(parser, default_n=5)
    parser.add_argument("--split", default="train", help="Dataset split.")
    parser.add_argument("--sleep-seconds", type=float, default=20.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir or (
        Path("benchmarks/results") / BENCHMARK_NAME / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run(
        n=50,
        split=args.split,
        seed=args.seed,
        output_dir=output_dir,
        sleep_seconds=5,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
