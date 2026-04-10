"""
evaluation/compare_topologies.py
---------------------------------
Runs ALL topologies against the same question(s) and prints a comparison table.
"""

import argparse
from config.observability import init_langsmith, Projects
from pipeline.stub_agent import run_pipeline, AVAILABLE_TOPOLOGIES


def compare_all(question: str, topologies: list[str] = None, enable_tracing: bool = True,
):
    """
    Runs every topology against the same question and prints a comparison table.

    Args:
        question:        The task to send to every topology
        topologies:      Which topologies to run (default: all 6)
        enable_tracing:  If True, initializes LangSmith so every run is traced
    """
    if topologies is None:
        topologies = AVAILABLE_TOPOLOGIES

    if enable_tracing:
        init_langsmith(Projects.BASELINE)

    results = []
    for topo in topologies:
        print(f"\nRunning: {topo}...")
        try:
            result = run_pipeline(question, topology=topo)
            result["topology"] = topo
            results.append(result)
            print(f"  Answer: {result['answer'][:60]}")
            print(f"  Cost: ${result['cost_usd']:.6f} | Latency: {result['latency_ms']}ms | Turns: {result['agent_turns']}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "topology":    topo,
                "answer":      f"ERROR: {e}",
                "cost_usd":    0.0,
                "latency_ms":  0,
                "agent_turns": 0,
            })

    # Print comparison table
    print(f"\n{'='*90}")
    print(f"  TOPOLOGY COMPARISON")
    print(f"  Question: {question[:70]}")
    print(f"{'='*90}")
    print(f"  {'Topology':<15} {'Answer':<30} {'Cost':>10} {'Latency':>10} {'Turns':>6}")
    print(f"  {'-'*15} {'-'*30} {'-'*10} {'-'*10} {'-'*6}")

    for r in results:
        answer_display = r['answer'][:28].replace('\n', ' ')
        print(
            f"  {r['topology']:<15} "
            f"{answer_display:<30} "
            f"${r['cost_usd']:>9.6f} "
            f"{r['latency_ms']:>8}ms "
            f"{r['agent_turns']:>5}"
        )

    # Summary stats
    costs = [r['cost_usd'] for r in results if r['cost_usd'] > 0]
    if costs:
        cheapest = min(results, key=lambda r: r['cost_usd'] if r['cost_usd'] > 0 else float('inf'))
        most_expensive = max(results, key=lambda r: r['cost_usd'])
        fastest = min(results, key=lambda r: r['latency_ms'] if r['latency_ms'] > 0 else float('inf'))

        print(f"\n  Cheapest:        {cheapest['topology']} (${cheapest['cost_usd']:.6f})")
        print(f"  Most expensive:  {most_expensive['topology']} (${most_expensive['cost_usd']:.6f})")
        print(f"  Fastest:         {fastest['topology']} ({fastest['latency_ms']}ms)")
        print(f"  Cost ratio:      {most_expensive['cost_usd'] / cheapest['cost_usd']:.1f}x")

    print(f"{'='*90}")

    return results


def compare_multiple_questions(
    questions: list[str],
    topologies: list[str] = None,
):
    """
    Runs the comparison across multiple questions.
    Useful for showing that the optimal topology varies by task complexity.
    """
    all_results = {}
    for q in questions:
        print(f"\n\n{'#'*90}")
        print(f"  QUESTION: {q[:80]}")
        print(f"{'#'*90}")
        all_results[q] = compare_all(q, topologies)

    return all_results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare all topologies against the same question")
    parser.add_argument(
        "--question", "-q",
        type=str,
        default="How many studio albums did The Beatles release?",
        help="Question to test all topologies against"
    )
    parser.add_argument(
        "--topologies", "-t",
        nargs="+",
        default=None,
        help="Which topologies to run (default: all)"
    )
    parser.add_argument(
        "--multi", "-m",
        action="store_true",
        help="Run a preset multi-question comparison (overrides --question)"
    )
    parser.add_argument(
        "--no-tracing",
        action="store_true",
        help="Disable LangSmith tracing"
    )
    args = parser.parse_args()

    if args.multi:
        # Three questions at different complexity levels to show
        # that optimal topology changes with task difficulty
        questions = [
            # Simple factual — single should win
            "What is the capital of France?",
            # Medium reasoning — chain or critic_loop should do well
            "If a train travels at 60 mph for 2.5 hours, then 80 mph for 1.5 hours, what is the total distance?",
            # Complex multi-step — hub_spoke or hierarchical might shine
            "Compare the GDP, population, and area of the three largest countries by land mass and determine which has the highest GDP per capita.",
        ]
        compare_multiple_questions(questions, args.topologies)
    else:
        compare_all(
            question=args.question,
            topologies=args.topologies,
            enable_tracing=not args.no_tracing,
        )