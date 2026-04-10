# Dynamic Topology Builder
# Instead of matching against static descriptors (topology_selector.py),
# this uses an LLM to design a brand-new topology at runtime for each task.
# The LLM acts as a "topology architect" — it decides how many agents,
# what roles they have, and how they communicate.

import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from graph.agent_graph import AgentGraph, AgentNode, AgentEdge, EdgeType

load_dotenv()


TOPOLOGY_ARCHITECT_PROMPT = """
You are a multi-agent topology architect. Given a user's task, you design the optimal agent collaboration graph to solve it.

You must output a single JSON object (no markdown fences, no extra text) with this schema:

{
  "name": "<short descriptive name for this topology>",
  "reasoning": "<1-2 sentences explaining why this structure fits the task>",
  "nodes": [
    {
      "name": "<unique agent name, PascalCase, e.g. Planner>",
      "role": "<what this agent does>",
      "model": "<LLM model id, default gpt-4o-mini>",
      "system_prompt": "<the agent's instructions>"
    }
  ],
  "edges": [
    {
      "source": "<source agent name>",
      "target": "<target agent name>",
      "edge_type": "<sequential | parallel | conditional | feedback>",
      "condition": "<required for conditional/feedback edges, null otherwise>",
      "max_traversals": <int, only for feedback edges, default 3>
    }
  ],
  "entry": "<name of the agent that receives the user's question first>"
}

Design rules:
1. Use the MINIMUM number of agents needed. A simple factual question needs just 1 agent.
2. Every agent name in edges must exist in nodes.
3. The entry agent must exist in nodes.
4. For feedback (revision) edges, always set max_traversals (2-4) to prevent infinite loops.
5. Prefer sequential edges for linear pipelines, parallel for independent subtasks, conditional for routing decisions, and feedback for iterative refinement.
6. Agent names must be unique.
7. Do NOT over-engineer: if a task can be solved with 1-2 agents, do not create 5.
8. Use gpt-4o-mini as the default model. Only use gpt-4o for agents that need stronger reasoning (e.g. a planner for very complex tasks).

Examples of good designs:
- "What is 2+2?" -> 1 node (Solver), 0 edges
- "Write and test a Python sort function" -> 2 nodes (Coder, Evaluator), 1 sequential + 1 feedback edge
- "Research 3 ML frameworks, compare them, and write a recommendation" -> 3-4 nodes (Planner, Researcher, Analyst, Writer) with appropriate edges
"""


def build_dynamic_topology(question: str, verbose: bool = False) -> AgentGraph:
    """
    Uses an LLM to design a topology for the given task at runtime.

    Instead of picking from a fixed set of topologies (like topology_selector.py),
    this generates a brand-new topology every time based on the task.

    Args:
        question: The user's task description
        verbose:  If True, print the raw LLM output

    Returns:
        An AgentGraph object ready to be executed by run_pipeline()
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    response = llm.invoke([
        SystemMessage(content=TOPOLOGY_ARCHITECT_PROMPT),
        HumanMessage(content=f"Design the optimal agent topology for this task:\n\n{question}"),
    ])

    raw_json = response.content.strip()

    if verbose:
        print(f"\n[DynamicBuilder] Raw LLM output:\n{raw_json}")

    # Parse the JSON response into an AgentGraph
    topology_dict = json.loads(raw_json)

    return _dict_to_agent_graph(topology_dict)


def _dict_to_agent_graph(d: dict) -> AgentGraph:
    """
    Converts the LLM's JSON output into an AgentGraph object.
    This bridges the dynamic builder with the rest of the codebase
    (agent_graph.py, run_pipeline, LangSmith metadata logging).
    """
    nodes = []
    for node_dict in d["nodes"]:
        nodes.append(AgentNode(
            name=node_dict["name"],
            role=node_dict["role"],
            model=node_dict.get("model", "gpt-4o-mini"),
            system_prompt=node_dict.get("system_prompt", ""),
        ))

    edges = []
    for edge_dict in d["edges"]:
        edge_type_str = edge_dict["edge_type"]
        edges.append(AgentEdge(
            source=edge_dict["source"],
            target=edge_dict["target"],
            edge_type=EdgeType(edge_type_str),
            condition=edge_dict.get("condition"),
            max_traversals=edge_dict.get("max_traversals") or 3,
        ))

    return AgentGraph(
        name=d.get("name", "dynamic"),
        nodes=nodes,
        edges=edges,
        entry=d["entry"],
    )


def validate_topology(graph: AgentGraph) -> list[str]:
    """
    Validates that the LLM-generated topology is structurally sound.
    Returns a list of error strings. Empty list = valid.
    """
    errors = []
    node_names = {n.name for n in graph.nodes}

    # Entry must exist
    if graph.entry not in node_names:
        errors.append(f"Entry agent '{graph.entry}' not found in nodes: {node_names}")

    # All edge references must exist
    for edge in graph.edges:
        if edge.source not in node_names:
            errors.append(f"Edge source '{edge.source}' not found in nodes")
        if edge.target not in node_names:
            errors.append(f"Edge target '{edge.target}' not found in nodes")

    # Feedback edges must have max_traversals
    for edge in graph.edges:
        if edge.edge_type == EdgeType.FEEDBACK and edge.max_traversals < 1:
            errors.append(f"Feedback edge {edge.source}->{edge.target} has no max_traversals")

    # No duplicate node names
    if len(node_names) != len(graph.nodes):
        errors.append("Duplicate node names detected")

    return errors



if __name__ == "__main__":
    test_questions = [
        "What is the capital of France?",
        "Write a Python function to compute factorial and test it.",
        "Scrape this webpage, extract all tables, clean the data, and store it in a SQLite database.",
        "Research the top 5 electric car manufacturers, compare their market share, battery range, and pricing, then write a structured report with recommendations.",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Task: {q}")
        print(f"{'='*60}")

        graph = build_dynamic_topology(q, verbose=True)

        # Validate
        errors = validate_topology(graph)
        if errors:
            print(f"[VALIDATION ERRORS] {errors}")
        else:
            print(f"[VALID]")

        # Print structure
        print(f"  Name:      {graph.name}")
        print(f"  Agents:    {[n.name for n in graph.nodes]}")
        print(f"  Edges:     {[(e.source, '->', e.target, e.edge_type.value) for e in graph.edges]}")
        print(f"  Entry:     {graph.entry}")
        print(f"  Has loops: {graph.has_feedback_loops()}")
        print(f"  Max turns: {graph.max_possible_turns()}")
        print(f"  Metadata:  {graph.to_metadata()}")