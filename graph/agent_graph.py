# Graph Architecture - explicit representation of agent communication

# Why this file needed?
# - In future while working with dynamic topology, we need to have a data structure that represents a topology which we will work with that generate based on the prompt.
# Till now topology is just a string like topology = "single" or "chain".
# For Dynamic, it will be like topology = AgentGraph(nodes=[...], edges=[...])


# How it connects to other files (rest of the code)?
#   topology_selector.py (Stage 3) will call build_topology_for_task() and get back an AgentGraph.
#   run_pipeline() will execute that AgentGraph.
#   LangSmith and wandb will log its structure as metadata.

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# Edge Types: how agents communicate

class EdgeType(Enum):
    """
    The type of communication between two agents.
    This is the key to making topology modular.

    SEQUENTIAL:  A must finish before B starts. Classic pipeline.
                 Example: Planner -> Coder -> Evaluator

    PARALLEL:    A sends to B and C simultaneously. B and C run at the same time.
                 Example: Planner -> [WebSearch, CodeExecutor] (concurrent)
                 LangGraph supports this natively.

    CONDITIONAL: A routes to B or C depending on a condition.
                 Example: Router -> SimpleAgent (if Level 1) | ComplexAgent (if Level 3)
                 This is what your dynamic topology selection becomes.

    FEEDBACK:    B sends back to A (revision loop).
                 Example: Evaluator -> Coder (if code fails QA)
                 CRITICAL: always has max_traversals to prevent infinite loops.
    """
    SEQUENTIAL  = "sequential"
    PARALLEL    = "parallel"
    CONDITIONAL = "conditional"
    FEEDBACK    = "feedback"



# Core data structures

@dataclass
class AgentNode:
    """
    A single agent in the topology.

    name:        Unique identifier. Used as the LangGraph node name.
    role:        What this agent does. Used by the NL descriptor router.
    model:       Which LLM to use. Different agents can use different models.
                 Example: use gpt-4o for the planner, gpt-4o-mini for the coder.
    system_prompt: The agent's persona and instructions.
    """
    name:          str
    role:          str
    model:         str = "gpt-4o-mini"
    system_prompt: str = ""


@dataclass
class AgentEdge:
    """
    A communication channel between two agents.

    source:          Agent that sends
    target:          Agent that receives
    edge_type:       How the message is sent (see EdgeType above)
    condition:       For CONDITIONAL edges: natural language description of
                     when to take this edge. The router uses this for NL matching.
    max_traversals:  For FEEDBACK edges: how many times this edge can be
                     traversed before we force-exit. THIS IS YOUR LOOP GUARD.
    """
    source:          str
    target:          str
    edge_type:       EdgeType
    condition:       Optional[str] = None
    max_traversals:  int = 3         # default: 3 revision cycles max


@dataclass
class AgentGraph:
    """
    A complete topology, the full set of agents and their communication edges.

    name:     Human-readable label. Shows up in LangSmith and wandb.
    nodes:    All agents in this topology.
    edges:    All communication edges between agents.
    entry:    Which agent receives the user's question first.
    """
    name:   str
    nodes:  list[AgentNode] = field(default_factory=list)
    edges:  list[AgentEdge] = field(default_factory=list)
    entry:  str = ""

    def to_metadata(self) -> dict:
        """
        Serializes this topology to a dict for LangSmith/wandb metadata.
        Call this when logging an experiment so you can see which topology
        ran for each task.
        """
        return {
            "topology_name":  self.name,
            "n_agents":       len(self.nodes),
            "n_edges":        len(self.edges),
            "agent_roles":    [n.role for n in self.nodes],
            "edge_types":     [e.edge_type.value for e in self.edges],
        }

    def has_feedback_loops(self) -> bool:
        """Returns True if this topology has any revision loops."""
        return any(e.edge_type == EdgeType.FEEDBACK for e in self.edges)

    def max_possible_turns(self) -> int:
        """
        Upper bound on how many agent calls this topology can make.
        Important for cost estimation before running.
        """
        feedback_edges = [e for e in self.edges if e.edge_type == EdgeType.FEEDBACK]
        base_turns = len(self.nodes)
        loop_turns = sum(e.max_traversals for e in feedback_edges)
        return base_turns + loop_turns



# Pre-built topologies against which we want to benchmark our dynamic graph.

def build_single_agent_topology() -> AgentGraph:
    """
    Stage 1 baseline: one agent, no collaboration.
    Used for simple factual questions.
    """
    return AgentGraph(
        name="single",
        nodes=[
            AgentNode(
                name="Solver",
                role="single general-purpose question answering agent",
                system_prompt=(
                    "You are a precise question-answering assistant. "
                    "Respond with ONLY the final answer. No explanation."
                ),
            )
        ],
        edges=[],
        entry="Solver",
    )


def build_chain_topology() -> AgentGraph:
    """
    Stage 1 target: Coder-Evaluator loop.
    Planner decomposes the task, Coder solves it, Evaluator checks the result.

    This maps directly to app.py:
      Coder node = coder_node()
      Evaluator node = evaluator_node()
      Feedback edge = the conditional loop back to Coder
    """
    return AgentGraph(
        name="chain",
        nodes=[
            AgentNode(
                name="Coder",
                role="Python code writer and problem solver",
                system_prompt="You are an expert Python developer. Write clean code.",
            ),
            AgentNode(
                name="Evaluator",
                role="QA engineer who checks if the code solves the stated task",
                system_prompt="You are a strict QA engineer. Check only what was asked.",
            ),
        ],
        edges=[
            AgentEdge(
                source="Coder",
                target="Evaluator",
                edge_type=EdgeType.SEQUENTIAL,
            ),
            AgentEdge(
                source="Evaluator",
                target="Coder",
                edge_type=EdgeType.FEEDBACK,
                condition="code fails QA check",
                max_traversals=3,   # matches MAX_CYCLES in app.py
            ),
        ],
        entry="Coder",
    )


def build_hub_spoke_topology() -> AgentGraph:
    """
    Stage 2 stretch: a Planner agent that decomposes tasks and delegates
    to specialist agents (WebSearch, Calculator, CodeExecutor).

    TODO Viraj: Implement the actual agents in Stage 3.
    For now this defines the structure so the router can reason about it.
    """
    return AgentGraph(
        name="hub_spoke",
        nodes=[
            AgentNode(name="Planner",       role="task decomposition and delegation planner"),
            AgentNode(name="WebSearch",     role="web search and information retrieval agent"),
            AgentNode(name="Calculator",    role="mathematical computation and reasoning agent"),
            AgentNode(name="CodeExecutor",  role="Python code writing and execution agent"),
            AgentNode(name="Synthesizer",   role="answer synthesis and final response agent"),
        ],
        edges=[
            AgentEdge("Planner",      "WebSearch",    EdgeType.CONDITIONAL, "task requires current information"),
            AgentEdge("Planner",      "Calculator",   EdgeType.CONDITIONAL, "task requires numerical computation"),
            AgentEdge("Planner",      "CodeExecutor", EdgeType.CONDITIONAL, "task requires code execution"),
            AgentEdge("WebSearch",    "Synthesizer",  EdgeType.SEQUENTIAL),
            AgentEdge("Calculator",   "Synthesizer",  EdgeType.SEQUENTIAL),
            AgentEdge("CodeExecutor", "Synthesizer",  EdgeType.SEQUENTIAL),
        ],
        entry="Planner",
    )



# Registry - maps topology name strings to AgentGraph objects
# This is how run_pipeline() will eventually look up topologies

TOPOLOGY_REGISTRY: dict[str, AgentGraph] = {
    "single":    build_single_agent_topology(),
    "chain":     build_chain_topology(),
    "hub_spoke": build_hub_spoke_topology(),
}

def get_topology(name: str) -> AgentGraph:
    if name not in TOPOLOGY_REGISTRY:
        raise ValueError(
            f"Unknown topology '{name}'. "
            f"Available: {list(TOPOLOGY_REGISTRY.keys())}"
        )
    return TOPOLOGY_REGISTRY[name]



if __name__ == "__main__":
    for name, topology in TOPOLOGY_REGISTRY.items():
        print(f"\n{topology.name}")
        print(f"  Agents:       {[n.name for n in topology.nodes]}")
        print(f"  Edges:        {[(e.source, '->', e.target, e.edge_type.value) for e in topology.edges]}")
        print(f"  Has loops:    {topology.has_feedback_loops()}")
        print(f"  Max turns:    {topology.max_possible_turns()}")
        print(f"  Metadata:     {topology.to_metadata()}")