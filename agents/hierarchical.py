"""
agents/hierarchical.py — Hierarchical Topology
-----------------------------------------------
Two layers of coordination: an Executive delegates to Managers,
each of whom coordinates their own Worker.

              Executive
             /         \\
       Mgr-Research   Mgr-Reasoning
           |               |
        Worker-A        Worker-B
             \\         /
            [Synthesizer] → Answer

Executive      : Reads the question, splits it into two parallel sub-tasks.
Mgr-Research   : Directs Worker-A to gather facts.
Mgr-Reasoning  : Directs Worker-B to derive the logical answer.
Workers        : Execute their manager's specific instruction.
Synthesizer    : Combines both branches into the final answer.

Best for  : Large tasks with natural decomposition into independent sub-problems.
Tradeoff  : More agent turns = higher cost and latency vs chain/hub-spoke.
            Communication overhead increases with depth.

LangSmith will show the full two-level delegation trace.
"""

from typing import Annotated, List
import operator
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from agents.base import get_llm, count_tokens, Timer, make_result, BaseState

load_dotenv()


# ── State ──────────────────────────────────────────────────────────────────────
class HierarchicalState(BaseState):
    research_output:  str   # Worker-A's result
    reasoning_output: str   # Worker-B's result
    total_tokens: Annotated[int, operator.add]


# ── Executive ──────────────────────────────────────────────────────────────────

def executive_node(state: HierarchicalState) -> dict:
    """Top-level coordinator. Frames the task for both managers."""
    print("  [Hierarchical] Executive: delegating...")
    llm = get_llm()

    response = llm.invoke([
        SystemMessage(content=(
            "You are an executive coordinator. Read the question and produce a brief "
            "task brief that will be given to two specialist teams: "
            "(1) a research team that gathers facts, and "
            "(2) a reasoning team that derives the answer logically. "
            "Keep the brief concise and actionable."
        )),
        *state["messages"],
    ])

    tokens = count_tokens(response)
    return {
        "messages":     [AIMessage(content=response.content, name="Executive")],
        "total_tokens": state["total_tokens"] + tokens,
    }


# ── Research branch ────────────────────────────────────────────────────────────

def mgr_research_node(state: HierarchicalState) -> dict:
    """Manager: instructs the research worker on what facts to find."""
    print("  [Hierarchical] ResearchManager: briefing worker...")
    llm = get_llm()

    response = llm.invoke([
        SystemMessage(content=(
            "You are a research manager. Based on the executive brief, "
            "instruct your research worker exactly what factual information "
            "to find. Be specific."
        )),
        *state["messages"],
    ])

    tokens = count_tokens(response)
    return {
        "messages":     [AIMessage(content=response.content, name="ResearchManager")],
        "total_tokens": state["total_tokens"] + tokens,
    }


def worker_research_node(state: HierarchicalState) -> dict:
    """Worker: executes the research manager's instruction."""
    print("  [Hierarchical] ResearchWorker: gathering facts...")
    llm = get_llm()

    response = llm.invoke([
        SystemMessage(content=(
            "You are a research worker. Follow your manager's instructions precisely. "
            "Gather and state all relevant facts. Be thorough."
        )),
        *state["messages"],
    ])

    tokens = count_tokens(response)
    return {
        "messages":        [AIMessage(content=response.content, name="ResearchWorker")],
        "research_output": response.content,
        "total_tokens":    state["total_tokens"] + tokens,
    }


# ── Reasoning branch ───────────────────────────────────────────────────────────

def mgr_reasoning_node(state: HierarchicalState) -> dict:
    """Manager: instructs the reasoning worker on how to approach the problem."""
    print("  [Hierarchical] ReasoningManager: briefing worker...")
    llm = get_llm()

    response = llm.invoke([
        SystemMessage(content=(
            "You are a reasoning manager. Based on the executive brief, "
            "instruct your reasoning worker exactly how to derive the answer logically. "
            "Specify the reasoning approach to use."
        )),
        *state["messages"],
    ])

    tokens = count_tokens(response)
    return {
        "messages":     [AIMessage(content=response.content, name="ReasoningManager")],
        "total_tokens": state["total_tokens"] + tokens,
    }


def worker_reasoning_node(state: HierarchicalState) -> dict:
    """Worker: executes the reasoning manager's instruction."""
    print("  [Hierarchical] ReasoningWorker: reasoning...")
    llm = get_llm()

    response = llm.invoke([
        SystemMessage(content=(
            "You are a reasoning worker. Follow your manager's instructions precisely. "
            "Reason step-by-step to the answer."
        )),
        *state["messages"],
    ])

    tokens = count_tokens(response)
    return {
        "messages":        [AIMessage(content=response.content, name="ReasoningWorker")],
        "reasoning_output": response.content,
        "total_tokens":    state["total_tokens"] + tokens,
    }


# ── Synthesizer ────────────────────────────────────────────────────────────────

def synthesizer_node(state: HierarchicalState) -> dict:
    """Combines both branch outputs into the final bare answer."""
    print("  [Hierarchical] Synthesizer: final answer...")
    llm = get_llm()

    combined = (
        f"[Research findings]:\n{state['research_output']}\n\n"
        f"[Logical reasoning]:\n{state['reasoning_output']}"
    )

    response = llm.invoke([
        SystemMessage(content=(
            "You are a synthesis agent. Using the research findings and logical reasoning "
            "provided, determine the single correct answer. "
            "Output ONLY the bare final answer — no explanation, no preamble."
        )),
        *state["messages"],
        HumanMessage(content=combined),
    ])

    tokens = count_tokens(response)
    return {
        "messages":     [AIMessage(content=response.content, name="Synthesizer")],
        "total_tokens": state["total_tokens"] + tokens,
    }


# ── Graph ──────────────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    workflow = StateGraph(HierarchicalState)

    workflow.add_node("Executive",        executive_node)
    workflow.add_node("ResearchManager",  mgr_research_node)
    workflow.add_node("ResearchWorker",   worker_research_node)
    workflow.add_node("ReasoningManager", mgr_reasoning_node)
    workflow.add_node("ReasoningWorker",  worker_reasoning_node)
    workflow.add_node("Synthesizer",      synthesizer_node)

    workflow.set_entry_point("Executive")

    # Research branch
    workflow.add_edge("Executive",       "ResearchManager")
    workflow.add_edge("ResearchManager", "ResearchWorker")

    # Reasoning branch
    workflow.add_edge("Executive",        "ReasoningManager")
    workflow.add_edge("ReasoningManager", "ReasoningWorker")

    # Both branches converge
    workflow.add_edge("ResearchWorker",  "Synthesizer")
    workflow.add_edge("ReasoningWorker", "Synthesizer")
    workflow.add_edge("Synthesizer",     END)

    return workflow.compile()


# ── Public interface ───────────────────────────────────────────────────────────

def run(question: str) -> dict:
    graph = _build_graph()

    initial_state: HierarchicalState = {
        "messages":         [HumanMessage(content=question)],
        "research_output":  "",
        "reasoning_output": "",
        "total_tokens":     0,
    }

    with Timer() as t:
        final_state = graph.invoke(initial_state)

    answer = final_state["messages"][-1].content

    return make_result(
        answer=answer,
        elapsed_ms=t.elapsed_ms,
        tokens=final_state["total_tokens"],
        turns=6,   # Executive + 2 Managers + 2 Workers + Synthesizer
    )


if __name__ == "__main__":
    result = run("What is the capital of Australia?")
    print(result)
