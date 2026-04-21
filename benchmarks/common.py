"""Shared plumbing for benchmark runners under ``benchmarks/``.

Every runner in this directory shares the same skeleton:

    1. Parse --n / --seed / --output-dir.
    2. Load and deterministically subsample a dataset.
    3. For each item: call a model, parse the response, score it.
    4. Stream each record to predictions.jsonl while printing a
       ``[label] i/N metric= elapsed= rate= eta=`` progress line.
    5. Write a summary.json with metrics + config + timing + the
       predictions path, and print a canonical done block.

This module factors out steps 1, 3's progress loop, and 5 so each
runner only writes the benchmark-specific parts: dataset loading,
prompting, parsing, and scoring.

The loop is exposed as :func:`run_items`, which takes two callbacks:

    process_item(item)  -> record dict   # runs the model on one item
    running_summary(rs) -> str           # e.g. "acc=0.783", for logs

The caller builds final metrics from the returned ``records`` list
and passes them to :func:`write_summary`.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Callable


# Strip <think>...</think> blocks emitted by reasoning-tuned models
# agents.common.get_reasoning_model sets reasoning=False so these should not appear
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think_blocks(text: str) -> str:
    return THINK_BLOCK_RE.sub("", text or "")


def response_text(value: Any) -> str:
    """Return plain text from a LangChain message, raw string, or content-block list.

    For block lists, concatenate the ``text`` fields of text-type blocks and
    drop thinking/reasoning blocks. ``str(list)`` would embed Python repr
    (single-quoted dicts, ``\\n`` escaped as two chars) and corrupt any JSON
    the caller tries to parse out of the result.
    """
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def shuffle_and_truncate(items: list[dict], seed: int, n: int | None) -> list[dict]:
    """Deterministic shuffle (with seed) + head (first n) for subsampling benchmark datasets"""
    rng = random.Random(seed)
    rng.shuffle(items)
    if n is not None:
        items = items[:n]
    return items


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    default_n: int | 30,
) -> None:
    """Attach --n, --seed, --output-dir to a benchmark runner's parser.

    The --output-dir default is None; each runner constructs its own
    timestamped default in main() so the timestamp reflects run time,
    not parse time.
    """
    parser.add_argument(
        "--n",
        type=int,
        default=default_n,
        help=f"Number of items to evaluate (default: {default_n}).",
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
        help="Output directory. Defaults to benchmarks/results/<benchmark>/<timestamp>.",
    )


ItemProcessor = Callable[[dict], dict]
RunningSummary = Callable[[list[dict]], str]


# TODO proof read this and next
def run_items(
    *,
    label: str,
    items: list[dict],
    process_item: ItemProcessor,
    running_summary: RunningSummary,
    output_dir: Path,
) -> tuple[list[dict], Path, float]:
    """Stream items through a model callback, logging and recording.

    Parameters
    ----------
    label:
        Short tag for progress lines (e.g. ``"gsm-hard"``). Used in
        every printed line so interleaved runs stay distinguishable.
    items:
        Pre-loaded, already-subsampled dataset items.
    process_item:
        Callable invoked once per item. Must return a JSON-serialisable
        dict — that dict is written verbatim to ``predictions.jsonl``
        and appended to the in-memory records list.
    running_summary:
        Callable receiving all records accumulated so far, returning a
        short string (e.g. ``"acc=0.783"``) that gets appended to each
        progress line. Called after every item.
    output_dir:
        Destination directory. Created if missing. ``predictions.jsonl``
        is written here; ``summary.json`` is left to the caller via
        :func:`write_summary`.

    Returns
    -------
    (records, predictions_path, runtime_seconds)
        - ``records`` is the full list of dicts returned by
          ``process_item``, in evaluation order.
        - ``predictions_path`` is the path of the JSONL that was just
          written.
        - ``runtime_seconds`` is wall-clock time for the loop only,
          excluding dataset loading and final metric computation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"

    records: list[dict] = []
    total = len(items)
    t0 = time.time()

    with predictions_path.open("w") as fh:
        for i, item in enumerate(items, start=1):
            record = process_item(item)
            records.append(record)
            fh.write(json.dumps(record) + "\n")
            fh.flush()

            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            eta = (total - i) / rate if rate > 0 else float("inf")
            summary_str = running_summary(records)
            print(
                f"[{label}] {i}/{total}  "
                f"{summary_str}  "
                f"elapsed={elapsed:6.1f}s  "
                f"rate={rate:4.2f}/s  eta={eta:6.1f}s",
                flush=True,
            )

    return records, predictions_path, time.time() - t0


def write_summary(
    *,
    output_dir: Path,
    label: str,
    metrics: dict,
    config: dict,
    runtime_seconds: float,
    predictions_path: Path,
) -> dict:
    """Write ``summary.json`` and print the canonical done block.

    Returns the summary dict for callers that want to keep working
    with it in-memory (e.g. orchestrators that aggregate across
    multiple benchmark runs without re-reading disk).
    """
    summary_path = output_dir / "summary.json"
    summary = {
        "metrics": metrics,
        "config": config,
        "runtime_seconds": runtime_seconds,
        "predictions_path": str(predictions_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"[{label}] done.", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"[{label}] predictions -> {predictions_path}", flush=True)
    print(f"[{label}] summary     -> {summary_path}", flush=True)
    return summary
