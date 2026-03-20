import argparse
from config.observability import init_langsmith, Projects
from evaluation.gaia_loader import GAIALoader, score_answer
from pipeline.stub_agent import run_pipeline


def main(n_tasks: int = 3, level: int = 1, topology: str = "single"):
    print("=" * 60)
    print("  Multi-Agent Orchestration — NLP Project")
    print(f"  Topology: {topology} | Tasks: {n_tasks} | GAIA Level: {level}")
    print("=" * 60)


    # Step 1: Initialize LangSmith
    print("\nInitializing LangSmith tracing")
    client = init_langsmith(Projects.BASELINE)


    # Step 2: Load GAIA
    print(f"\nLoading {n_tasks} GAIA Level {level} task(s)")
    loader = GAIALoader(level=level)
    tasks = loader.get_tasks(n=n_tasks)
    print(f"Loaded {len(tasks)} tasks.")


    # Step 3: Run each task through the pipeline
    print(f"\nRunning tasks through pipeline (topology='{topology}')\n")

    results = []
    for i, task in enumerate(tasks, 1):
        print(f"  Task {i}/{len(tasks)}: {task.question[:80]}")

        # run_pipeline is the stable interface, omly topology changes between Stage 1 (single), Stage 2 (chain), and Stage 3 (dynamic)
        pipeline_output = run_pipeline(task.question, topology=topology)

        # Score the agent's answer against GAIA ground truth
        score = score_answer(pipeline_output["answer"], task)

        result = {
            **score,                                  # correct, agent_answer, ground_truth
            "cost_usd":    pipeline_output["cost_usd"],
            "latency_ms":  pipeline_output["latency_ms"],
            "agent_turns": pipeline_output["agent_turns"],
        }
        results.append(result)

        print(f"Agent: '{result['agent_answer'][:50]}' | Truth: '{result['ground_truth']}'")
        print(f"Cost: ${result['cost_usd']:.4f} | Latency: {result['latency_ms']}ms\n")

    # Summary
    n_correct = sum(r["correct"] for r in results)
    accuracy  = n_correct / len(results) if results else 0
    avg_cost  = sum(r["cost_usd"] for r in results) / len(results) if results else 0

    print("=" * 60)
    print(f"  RESULTS - topology: '{topology}'")
    print(f"  Accuracy : {accuracy * 100:.1f}%  ({n_correct}/{len(results)} correct)")
    print(f"  Avg Cost : ${avg_cost:.4f} per task")
    print(f"\n  Open LangSmith to see full traces:")
    print(f"  https://smith.langchain.com - project: {Projects.BASELINE}")
    print("=" * 60)

    # This dict is what experiment_runner.py will eventually collect
    # across all topologies for the ablation comparison table.
    return {
        "topology":  topology,
        "accuracy":  round(accuracy, 3),
        "n_tasks":   len(results),
        "correct":   n_correct,
        "avg_cost":  round(avg_cost, 4),
    }



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks",    type=int, default=3,        help="Number of GAIA tasks to run")
    parser.add_argument("--level",    type=int, default=1,        help="GAIA difficulty level (1/2/3)")
    parser.add_argument("--topology", type=str, default="single", help="Agent topology to use")
    args = parser.parse_args()

    main(n_tasks=args.tasks, level=args.level, topology=args.topology)