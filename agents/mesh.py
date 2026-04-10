"""
agents/mesh.py — Fully Connected / Mesh Topology
-------------------------------------------------
Every agent can communicate directly with every other agent.
Each agent sees all other agents' outputs before producing its own.

    ┌─────────────────────────────┐
    │  Agent A ↔ Agent B          │
    │     ↕   ×   ↕               │
    │  Agent C ↔ Agent D          │
    └─────────────────────────────┘
              ↓
          [Arbitrator] → Answer

Round 1 — each agent answers independently.
Round 2 — each agent sees all other agents' round 1 answers and can revise.
Arbitrator — reads all final answers and picks the most consistent one.

Agents:
    Factual    : Answers from established knowledge.
    Analytical : Answers through step-by-step analysis.
    Critical   : Actively looks for what could be wrong.
    Creative   : Considers unconventional angles or interpretations.

Known tradeoffs:
    - Maximum information sharing between agents.
    - Token costs scale with N² (every agent reads every other agent's output).
    - Risk of context pollution — agents may converge on a confident wrong answer.
    - Communication rounds are capped at MAX_ROUNDS to control cost.

Best for  : Tasks that benefit from diverse perspectives and peer review.
Tradeoff  : Most expensive topology. Highest risk of groupthink on hard tasks.
"""

from typing import Annotated, List
import operator
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from agents.base import get_llm, count_tokens, Timer, make_result, BaseState

load_dotenv()

MAX_ROUNDS = 2   # Number of full mesh communication rounds.
                 # Each round = all agents read all other agents' outputs.
                 # Keep low — cost grows fast (N agents × N outputs × rounds).

AGENTS = ["Factual", "Analytical", "Critical", "Creative"]

PERSONAS = {
    "Factual": (
        "You are a factual knowledge agent. Answer the question using only "
        "established, verifiable facts. Be precise and concise."
    ),
    "Analytical": (
        "You are an analytical reasoning agent. Break the question into parts "
        "and derive the answer step by step. Show your reasoning clearly."
    ),
    "Critical": (
        "You are a critical thinking agent. Your job is to identify flaws, "
        "edge cases, and assumptions. Challenge the question and give the most "
        "defensible answer."
    ),
    "Creative": (
        "You are a creative reasoning agent. Consider unconventional angles, "
        "alternative interpretations, and lateral thinking approaches to answer "
        "the question."
    ),
}


# ── State ──────────────────────────────────────────────────────────────────────
class MeshState(BaseState):
    # Each agent's latest answer — keyed by agent name
    agent_answers:  dict          # {"Factual": "...", "Analytical": "...", ...}
    current_round:  int           # which communication round we're on
    total_tokens:   int


# ── Helper: format all other agents' answers for one agent to read ─────────────
def _peer_context(agent_name: str, agent_answers: dict) -> str:
    """Builds a summary of all OTHER agents' current answers for one agent to read."""
    peers = {k: v for k, v in agent_answers.items() if k != agent_name and v}
    if not peers:
        return ""
    lines = [f"[{name}]: {answer}" for name, answer in peers.items()]
    return "Other agents' current answers:\n\n" + "\n\n".join(lines)


# ── Agent node factory ─────────────────────────────────────────────────────────
def _make_agent_node(agent_name: str):
    """
    Creates a node for one mesh agent.
    Round 1: answers independently.
    Round 2+: reads all peer answers, then revises or confirms its own.
    """
    def agent_node(state: MeshState) -> dict:
        round_num = state["current_round"]
        print(f"  [Mesh] {agent_name} (round {round_num}): thinking...")
        llm = get_llm()

        peer_ctx = _peer_context(agent_name, state["agent_answers"])

        if peer_ctx:
            # Mesh round: read peers, revise if needed
            messages = [
                SystemMessage(content=PERSONAS[agent_name]),
                *state["messages"],
                HumanMessage(content=(
                    f"{peer_ctx}\n\n"
                    f"Given your own reasoning and your peers' answers above, "
                    f"what is your final answer? You may revise or confirm your previous position."
                )),
            ]
        else:
            # First round: answer independently
            messages = [
                SystemMessage(content=PERSONAS[agent_name]),
                *state["messages"],
            ]

        response = llm.invoke(messages)
        tokens = count_tokens(response)

        # Update this agent's answer in the shared dict
        updated_answers = {**state["agent_answers"], agent_name: response.content}

        return {
            "messages":     [AIMessage(content=response.content, name=agent_name)],
            "agent_answers": updated_answers,
            "total_tokens": state["total_tokens"] + tokens,
        }

    agent_node.__name__ = f"{agent_name.lower()}_node"
    return agent_node


# ── Round counter node ─────────────────────────────────────────────────────────
def increment_round(state: MeshState) -> dict:
    """Increments the round counter after all agents have spoken."""
    return {"current_round": state["current_round"] + 1}


# ── Round router ───────────────────────────────────────────────────────────────
def should_continue(state: MeshState) -> str:
    """After each round, decide whether to do another mesh round or exit."""
    if state["current_round"] >= MAX_ROUNDS:
        return "Arbitrator"
    return "next_round"


# ── Arbitrator ─────────────────────────────────────────────────────────────────
def arbitrator_node(state: MeshState) -> dict:
    """
    Reads all agents' final answers and produces the single correct answer.
    Looks for consensus — if agents agree, that answer wins.
    If they disagree, arbitrates based on which reasoning is most sound.
    """
    print("  [Mesh] Arbitrator: reaching consensus...")
    llm = get_llm()

    all_answers = "\n\n".join(
        f"[{name}]: {answer}"
        for name, answer in state["agent_answers"].items()
        if answer
    )

    response = llm.invoke([
        SystemMessage(content=(
            "You are an arbitrator. You will receive answers from multiple agents "
            "who have debated the question across several rounds. "
            "Identify the most accurate answer based on the quality of reasoning "
            "and degree of consensus. "
            "Output ONLY the bare final answer — no explanation, no preamble."
        )),
        *state["messages"],
        HumanMessage(content=f"Agent answers after {state['current_round']} round(s):\n\n{all_answers}"),
    ])

    tokens = count_tokens(response)
    return {
        "messages":     [AIMessage(content=response.content, name="Arbitrator")],
        "total_tokens": state["total_tokens"] + tokens,
    }


# ── Graph ──────────────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    workflow = StateGraph(MeshState)

    # Add all agent nodes
    for agent_name in AGENTS:
        workflow.add_node(agent_name, _make_agent_node(agent_name))

    # Add round management nodes
    workflow.add_node("increment_round", increment_round)
    workflow.add_node("Arbitrator",      arbitrator_node)

    # Entry: all agents run in round 1
    workflow.set_entry_point(AGENTS[0])
    for i in range(len(AGENTS) - 1):
        workflow.add_edge(AGENTS[i], AGENTS[i + 1])

    # After last agent, increment the round counter
    workflow.add_edge(AGENTS[-1], "increment_round")

    # After incrementing, decide: another round or go to Arbitrator
    workflow.add_conditional_edges(
        "increment_round",
        should_continue,
        {
            "next_round": AGENTS[0],   # loop back to start of agent sequence
            "Arbitrator": "Arbitrator",
        }
    )

    workflow.add_edge("Arbitrator", END)

    return workflow.compile()


# ── Public interface ───────────────────────────────────────────────────────────

def run(question: str) -> dict:
    graph = _build_graph()

    initial_state: MeshState = {
        "messages":      [HumanMessage(content=question)],
        "agent_answers": {name: "" for name in AGENTS},
        "current_round": 0,
        "total_tokens":  0,
    }

    with Timer() as t:
        final_state = graph.invoke(initial_state)

    answer = final_state["messages"][-1].content
    turns  = len(AGENTS) * MAX_ROUNDS + 1   # all agents × rounds + Arbitrator

    return make_result(
        answer=answer,
        elapsed_ms=t.elapsed_ms,
        tokens=final_state["total_tokens"],
        turns=turns,
    )


if __name__ == "__main__":
    import sys
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the capital of Australia?"
    result = run(question)
    print(result)
