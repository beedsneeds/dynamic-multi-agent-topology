"""
pipeline/stub_agent.py
----------------------
The stable interface between the evaluation layer and the agent layer.

Evaluation code (main.py, experiment_runner.py) ALWAYS calls run_pipeline().
The topology argument determines which agents/ module is loaded.

Adding a new topology = adding one elif here + a new file in agents/.

Topologies available:
    "single"       — one LLM call, no graph
    "chain"        — Researcher -> Reasoner -> Formatter
    "hub_spoke"    — Supervisor + workers + Synthesizer
    "hierarchical" — Executive -> Managers -> Workers -> Synthesizer
    "critic_loop"  — Planner -> Solver <-> Critic -> Formatter
    "mesh"         — All agents <-> All agents -> Arbitrator
"""

from dotenv import load_dotenv
from agents.base import tokens_to_cost

load_dotenv()


AVAILABLE_TOPOLOGIES = ["single", "chain", "hub_spoke", "hierarchical", "critic_loop", "mesh"]


def run_pipeline(question: str, topology: str = "single") -> dict:
    """
    Stable interface — evaluation code never changes, only this router grows.

    Returns the dict shape that main.py and experiment_runner.py expect:
        {
            "answer":      str,
            "cost_usd":    float,
            "latency_ms":  int,
            "agent_turns": int,
        }
    """
    if topology == "single":
        from agents.single import run
    elif topology == "chain":
        from agents.chain import run
    elif topology == "hub_spoke":
        from agents.hub_spoke import run
    elif topology == "hierarchical":
        from agents.hierarchical import run
    elif topology == "critic_loop":
        from agents.critic_loop import run
    elif topology == "mesh":
        from agents.mesh import run
    else:
        raise ValueError(
            f"Unknown topology: '{topology}'. Available: {AVAILABLE_TOPOLOGIES}"
        )

    # Every agents/ module returns the same AgentResult shape
    result = run(question)

    # Translate AgentResult → evaluation layer format
    return {
        "answer":      result["answer"],
        "cost_usd":    tokens_to_cost(result["tokens_used"]),
        "latency_ms":  result["time_ms"],
        "agent_turns": result["agent_turns"],
    }


if __name__ == "__main__":
    topologies = AVAILABLE_TOPOLOGIES
    q = "What is 2 + 2?"

    for t in topologies:
        print(f"\n--- {t} ---")
        print(run_pipeline(q, topology=t))