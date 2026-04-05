"""
agents/hub_spoke.py — Star / Hub-Spoke Topology
-------------------------------------------------
A central Supervisor routes the question to specialist workers,
then synthesizes their outputs into a final answer.

              [Web Worker]
                   ↑
    Question → [Supervisor] → [Math Worker]   → [Synthesizer] → Answer
                   ↓
              [Logic Worker]

Supervisor  : Reads the question and decides which workers to consult.
              Fans out to ALL workers in this implementation (simpler and
              more robust for GAIA tasks than selective routing).
Workers     : Each has a specialist persona — factual, mathematical, logical.
Synthesizer : Collects all worker outputs and produces the single final answer.

Best for  : Tasks that benefit from multiple expert perspectives.
Tradeoff  : Supervisor is a single point of failure. More tokens than chain.

LangSmith shows you each worker's individual reasoning trace.
"""

from typing import Annotated, List
import operator
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from agents.base import get_llm, count_tokens, Timer, make_result, BaseState

load_dotenv()


# ── State ──────────────────────────────────────────────────────────────────────
class HubSpokeState(BaseState):
    worker_outputs: Annotated[List[str], operator.add]  # collects each worker's answer
    total_tokens:   Annotated[int, operator.add]


# ── Supervisor ─────────────────────────────────────────────────────────────────

def supervisor_node(state: HubSpokeState) -> dict:
    """
    Reads the question and produces a task decomposition.
    This message is passed to all workers so they know what angle to take.
    """
    print("  [Hub-Spoke] Supervisor: decomposing task...")
    llm = get_llm()

    response = llm.invoke([
        SystemMessage(content=(
            "You are a task supervisor. Read the question and briefly describe "
            "what information is needed to answer it. Be concise — this is a briefing "
            "for specialist agents, not the final answer."
        )),
        *state["messages"],
    ])

    tokens = count_tokens(response)
    return {
        "messages":      [AIMessage(content=response.content, name="Supervisor")],
        "total_tokens":  state["total_tokens"] + tokens,
        "worker_outputs": [],
    }


# ── Workers ────────────────────────────────────────────────────────────────────

def _make_worker(name: str, persona: str):
    """Factory that creates a worker node with a given persona."""
    def worker_node(state: HubSpokeState) -> dict:
        print(f"  [Hub-Spoke] {name}: working...")
        llm = get_llm()

        response = llm.invoke([
            SystemMessage(content=persona),
            *state["messages"],
        ])

        tokens = count_tokens(response)
        return {
            "messages":      [AIMessage(content=response.content, name=name)],
            "worker_outputs": [f"[{name}]: {response.content}"],
            "total_tokens":  state["total_tokens"] + tokens,
        }
    worker_node.__name__ = f"{name.lower()}_node"
    return worker_node


factual_worker  = _make_worker("FactualWorker",  "You are a factual knowledge expert. Answer the question using only established facts. Be precise and concise.")
math_worker     = _make_worker("MathWorker",     "You are a mathematical reasoning expert. If the question involves numbers or calculations, solve it step by step. If not, state that math is not the primary approach.")
logical_worker  = _make_worker("LogicalWorker",  "You are a logical reasoning expert. Analyse the question's structure and constraints. Derive the answer through pure logic.")


# ── Synthesizer ────────────────────────────────────────────────────────────────

def synthesizer_node(state: HubSpokeState) -> dict:
    """
    Reads all worker outputs and produces the single final answer.
    This is what the evaluator scores.
    """
    print("  [Hub-Spoke] Synthesizer: producing final answer...")
    llm = get_llm()

    worker_summary = "\n\n".join(state["worker_outputs"])

    response = llm.invoke([
        SystemMessage(content=(
            "You are a synthesis agent. You will receive answers from multiple specialist agents. "
            "Determine the most accurate final answer by weighing their reasoning. "
            "Output ONLY the bare final answer — no explanation, no preamble."
        )),
        *state["messages"],
        HumanMessage(content=f"Specialist outputs:\n\n{worker_summary}"),
    ])

    tokens = count_tokens(response)
    return {
        "messages":     [AIMessage(content=response.content, name="Synthesizer")],
        "total_tokens": state["total_tokens"] + tokens,
    }


# ── Graph ──────────────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    workflow = StateGraph(HubSpokeState)

    workflow.add_node("Supervisor",    supervisor_node)
    workflow.add_node("FactualWorker", factual_worker)
    workflow.add_node("MathWorker",    math_worker)
    workflow.add_node("LogicalWorker", logical_worker)
    workflow.add_node("Synthesizer",   synthesizer_node)

    workflow.set_entry_point("Supervisor")

    # Fan out: Supervisor → all workers (run sequentially; LangGraph parallel fan-out needs Send API)
    workflow.add_edge("Supervisor",    "FactualWorker")
    workflow.add_edge("Supervisor",    "MathWorker")
    workflow.add_edge("Supervisor",    "LogicalWorker")

    # Fan in: all workers → Synthesizer
    workflow.add_edge("FactualWorker", "Synthesizer")
    workflow.add_edge("MathWorker",    "Synthesizer")
    workflow.add_edge("LogicalWorker", "Synthesizer")

    workflow.add_edge("Synthesizer", END)

    return workflow.compile()


# ── Public interface ───────────────────────────────────────────────────────────

def run(question: str) -> dict:
    graph = _build_graph()

    initial_state: HubSpokeState = {
        "messages":      [HumanMessage(content=question)],
        "worker_outputs": [],
        "total_tokens":  0,
    }

    with Timer() as t:
        final_state = graph.invoke(initial_state)

    answer = final_state["messages"][-1].content

    return make_result(
        answer=answer,
        elapsed_ms=t.elapsed_ms,
        tokens=final_state["total_tokens"],
        turns=5,   # Supervisor + 3 workers + Synthesizer
    )


if __name__ == "__main__":
    result = run("What is the capital of Australia?")
    print(result)
