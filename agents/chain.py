"""
agents/chain.py — Chain / Pipeline Topology
--------------------------------------------
Agents execute in a fixed sequence. Each agent processes the output
of the previous one — like an assembly line.

    Question → [Researcher] → [Reasoner] → [Formatter] → Answer

Researcher : Gathers and states all relevant facts about the question.
Reasoner   : Takes those facts and derives the answer step-by-step.
Formatter  : Strips all explanation and returns the bare final answer.

Best for  : Multi-stage reasoning where each step has a clear role.
Tradeoff  : Simple to debug. No parallelism. One weak agent hurts the whole chain.

LangSmith will show you the full message trace across all three agents.
"""

from typing import Annotated, List
import operator
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from agents.base import get_llm, count_tokens, Timer, make_result, BaseState

load_dotenv()

# ── State ──────────────────────────────────────────────────────────────────────
# Inherits messages list from BaseState. We track token usage across all nodes.
class ChainState(BaseState):
    total_tokens: int


# ── Nodes ──────────────────────────────────────────────────────────────────────

def researcher_node(state: ChainState) -> dict:
    """
    Stage 1 of 3.
    Reads the original question and surfaces all relevant facts.
    Does NOT attempt to answer — only gathers.
    """
    print("  [Chain] Researcher: gathering facts...")
    llm = get_llm()

    response = llm.invoke([
        SystemMessage(content=(
            "You are a research assistant. Your ONLY job is to identify and state "
            "all facts relevant to answering the user's question. "
            "Do not give a final answer. Just list the facts clearly."
        )),
        *state["messages"],
    ])

    tokens = count_tokens(response)
    return {
        "messages": [AIMessage(content=response.content, name="Researcher")],
        "total_tokens": state["total_tokens"] + tokens,
    }


def reasoner_node(state: ChainState) -> dict:
    """
    Stage 2 of 3.
    Takes the Researcher's facts and reasons to a conclusion.
    Shows its working — the Formatter will clean it up.
    """
    print("  [Chain] Reasoner: deriving answer...")
    llm = get_llm()

    response = llm.invoke([
        SystemMessage(content=(
            "You are a logical reasoning agent. You will be given a question and "
            "a set of relevant facts. Reason step-by-step to arrive at the correct answer. "
            "Show your reasoning clearly."
        )),
        *state["messages"],
    ])

    tokens = count_tokens(response)
    return {
        "messages": [AIMessage(content=response.content, name="Reasoner")],
        "total_tokens": state["total_tokens"] + tokens,
    }


def formatter_node(state: ChainState) -> dict:
    """
    Stage 3 of 3.
    Reads the Reasoner's conclusion and strips everything except the bare answer.
    This is what the evaluator scores against GAIA ground truth.
    """
    print("  [Chain] Formatter: extracting final answer...")
    llm = get_llm()

    response = llm.invoke([
        SystemMessage(content=(
            "You are a precise answer extractor. You will be given a reasoning chain. "
            "Extract ONLY the final answer — no explanation, no units unless asked for, "
            "no punctuation beyond what's in the answer itself. Just the bare answer."
        )),
        *state["messages"],
    ])

    tokens = count_tokens(response)
    return {
        "messages": [AIMessage(content=response.content, name="Formatter")],
        "total_tokens": state["total_tokens"] + tokens,
    }


# ── Graph ──────────────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    workflow = StateGraph(ChainState)

    workflow.add_node("Researcher", researcher_node)
    workflow.add_node("Reasoner",   reasoner_node)
    workflow.add_node("Formatter",  formatter_node)

    workflow.set_entry_point("Researcher")
    workflow.add_edge("Researcher", "Reasoner")
    workflow.add_edge("Reasoner",   "Formatter")
    workflow.add_edge("Formatter",  END)

    return workflow.compile()


# ── Public interface ───────────────────────────────────────────────────────────

def run(question: str) -> dict:
    graph = _build_graph()

    initial_state: ChainState = {
        "messages":     [HumanMessage(content=question)],
        "total_tokens": 0,
    }

    with Timer() as t:
        final_state = graph.invoke(initial_state)

    # The last message is always the Formatter's bare answer
    answer = final_state["messages"][-1].content

    return make_result(
        answer=answer,
        elapsed_ms=t.elapsed_ms,
        tokens=final_state["total_tokens"],
        turns=3,   # Researcher → Reasoner → Formatter
    )


if __name__ == "__main__":
    result = run("What is the capital of Australia?")
    print(result)
