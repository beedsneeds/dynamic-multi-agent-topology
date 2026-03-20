import time
from langsmith import Client
from langsmith.evaluation import evaluate
from evaluation.gaia_loader import GAIALoader, normalize_answer
from pipeline.stub_agent import run_pipeline
from config.observability import init_langsmith, Projects


# Step 1: Upload GAIA tasks to LangSmith as a reusable Dataset

def create_gaia_dataset(client: Client, n_tasks: int = 20, level: int = 1, dataset_name: str = None) -> str:
    if dataset_name is None:
        dataset_name = f"GAIA-Level{level}-Validation-{n_tasks}tasks"

    existing = [d.name for d in client.list_datasets()]
    if dataset_name in existing:
        print(f"[LangSmith] Dataset '{dataset_name}' already exists. Skipping upload.")
        return dataset_name

    loader = GAIALoader(level=level)
    tasks = loader.get_tasks(n=n_tasks)

    print(f"[LangSmith] Creating dataset '{dataset_name}' with {len(tasks)} tasks")

    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=f"GAIA Level {level} benchmark - {n_tasks} validation tasks"
    )

    client.create_examples(
        inputs=[{"question": task.question} for task in tasks],
        outputs=[{"answer": task.ground_truth} for task in tasks],
        dataset_id=dataset.id,
    )

    print(f"[LangSmith] Dataset '{dataset_name}' created.")
    return dataset_name



# Step 2: Custom evaluator - what "correct" means for GAIA

def gaia_exact_match_evaluator(run, example) -> dict:
    """
    LangSmith calls this function for every (agent_run, ground_truth) pair.

    run.outputs     = what the agent returned
    example.outputs = the ground truth

    TODO Add more evaluators later:
      - Did the cross-checking agent catch a hallucination?
      - Did the pipeline finish under cost budget?
      - Did the pipeline finish without looping?
      etc.
    """
    agent_answer = run.outputs.get("answer", "")
    ground_truth = example.outputs.get("answer", "")

    is_correct = normalize_answer(agent_answer) == normalize_answer(ground_truth)

    return {
        "key":     "exact_match",
        "score":   int(is_correct),
        "comment": f"Agent: '{agent_answer[:60]}' | Truth: '{ground_truth[:60]}'"
    }



# Step 3: Run a single experiment

def run_experiment(topology: str, dataset_name: str, experiment_name: str = None) -> dict:
    """
    Call this function for every experiment.
    run_experiment("single", dataset_name, "baseline")
    run_experiment("chain",  dataset_name, "chain")
    run_experiment("chain+rag", dataset_name, "rag")
    run_experiment("dynamic", dataset_name, "dynamic")
    """
    if experiment_name is None:
        experiment_name = f"{topology}-{int(time.time())}"

    print(f"\n[Experiment] '{experiment_name}' | topology='{topology}'")

    def agent_fn(inputs: dict) -> dict:
        """LangSmith calls this once per dataset example."""
        result = run_pipeline(inputs["question"], topology=topology)

        # TODO log cost and latency to wandb here
        # wandb.log({"cost_usd": result["cost_usd"], "latency_ms": result["latency_ms"]})

        return {"answer": result["answer"]}

    results = evaluate(
        agent_fn,
        data=dataset_name,
        evaluators=[gaia_exact_match_evaluator],
        experiment_prefix=experiment_name,
        metadata={"topology": topology},
    )

    scores = [r["evaluation_results"]["results"][0].score for r in results._results]
    accuracy = sum(scores) / len(scores) if scores else 0.0

    summary = {
        "experiment": experiment_name,
        "topology":   topology,
        "accuracy":   round(accuracy, 3),
        "correct":    sum(scores),
        "n_tasks":    len(scores),
    }

    print(f"[Experiment] Accuracy: {accuracy * 100:.1f}% ({sum(scores)}/{len(scores)})")
    return summary



# Step 4: Compare multiple topologies

def compare_topologies(topologies: list[str], dataset_name: str) -> list[dict]:
    """
    This runs every topology against the same dataset and prints a comparison table.

    TODO: Call this once all topologies are implemented:
      compare_topologies(["single", "chain", "chain+rag", "dynamic"], dataset_name)
    """
    all_results = []
    for topology in topologies:
        summary = run_experiment(topology, dataset_name, f"ablation-{topology}")
        all_results.append(summary)

    # Print comparison table
    print("\nTopology Comparison")
    print(f"  {'Topology':<20} {'Accuracy':>10} {'Correct':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10}")
    for r in all_results:
        print(f"  {r['topology']:<20} {r['accuracy']*100:>9.1f}% {r['correct']:>4}/{r['n_tasks']:<6}")

    return all_results



if __name__ == "__main__":    
    client = init_langsmith(Projects.BASELINE)
    dataset_name = create_gaia_dataset(client, n_tasks=5, level=1)
    summary = run_experiment(
        topology="single",
        dataset_name=dataset_name,
        experiment_name="single-baseline",
    )
    print("\nFinal summary:", summary)