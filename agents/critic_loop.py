"""
agents/critic_loop.py — Critic Loop Topology
---------------------------------------------
A self-correcting loop where a Critic evaluates the Solver's answer
and sends feedback back until the answer passes or MAX_ROUNDS is reached.

    Question → [Planner] → [Solver] ←→ [Critic] → [Formatter] → Answer
                                  ↑___________|
                                  (loop until passed or max_rounds)

Planner   : Breaks the question into a clear solution strategy.
Solver    : Attempts to answer following the plan.
            On retry rounds, reads the Critic's feedback and corrects itself.
Critic    : Evaluates the Solver's answer using structured output.
            If passed → exit loop to Formatter.
            If failed → inject feedback into message history and loop back.
Formatter : Strips the final accepted answer to bare output for evaluation.

MAX_ROUNDS guards against infinite loops (important for cost control on GAIA).

Best for  : Tasks where the first attempt is likely wrong and iteration helps.
Tradeoff  : Best accuracy potential. Unpredictable cost (depends on loop count).
            Risk of looping on questions the model fundamentally cannot answer.
"""

from typing import Annotated, List
import operator
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from agents.base import get_llm, count_tokens, Timer, make_result, BaseState

load_dotenv()

MAX_ROUNDS = 3   # Max Critic→Solver loops before we accept best answer


# ── State ──────────────────────────────────────────────────────────────────────
class CriticLoopState(BaseState):
    answer_passed: bool   # Critic's verdict — controls the loop
    rounds:        int    # How many Solver→Critic cycles have run
    total_tokens:  int


# ── Structured Critic output ───────────────────────────────────────────────────
class CriticVerdict(BaseModel):
    passed:   bool = Field(description="True if the answer is correct and complete.")
    feedback: str  = Field(description="If passed is False, explain what is wrong and how to fix it.")


# ── Planner ────────────────────────────────────────────────────────────────────

def planner_node(state: CriticLoopState) -> dict:
    """
    Runs once at the start. Produces a solution strategy for the Solver.
    This reduces the chance of the Solver going in the wrong direction on loop 1.
    """
    print("  [CriticLoop] Planner: building strategy...")
    llm = get_llm()

    response = llm.invoke([
        SystemMessage(content=(
            "You are a strategic planner. Read the question and produce a clear, "
            "step-by-step plan for how to arrive at the correct answer. "
            "Do not solve it — just plan."
        )),
        *state["messages"],
    ])

    tokens = count_tokens(response)
    return {
        "messages":     [AIMessage(content=response.content, name="Planner")],
        "total_tokens": state["total_tokens"] + tokens,
    }


# ── Solver ─────────────────────────────────────────────────────────────────────

def solver_node(state: CriticLoopState) -> dict:
    """
    Attempts to answer the question.
    On later rounds, the Critic's feedback is in the message history — the Solver
    reads it and tries to correct its previous answer.
    """
    round_num = state["rounds"] + 1
    print(f"  [CriticLoop] Solver: attempt {round_num}...")
    llm = get_llm()

    response = llm.invoke([
        SystemMessage(content=(
            "You are a precise solver. Follow the planner's strategy to answer the question. "
            "If you received feedback from the critic, address every point raised. "
            "State your answer clearly."
        )),
        *state["messages"],
    ])

    tokens = count_tokens(response)
    return {
        "messages":     [AIMessage(content=response.content, name="Solver")],
        "rounds":       round_num,
        "total_tokens": state["total_tokens"] + tokens,
    }


# ── Critic ─────────────────────────────────────────────────────────────────────

def critic_node(state: CriticLoopState) -> dict:
    """
    Evaluates the Solver's latest answer using structured output so the
    router gets a clean bool. If MAX_ROUNDS reached, forces passed=True
    to break the loop and accept the best answer we have.
    """
    print(f"  [CriticLoop] Critic: evaluating (round {state['rounds']})...")
    llm = get_llm()

    # Hard stop: accept the best answer we have
    if state["rounds"] >= MAX_ROUNDS:
        print(f"  [CriticLoop] MAX_ROUNDS ({MAX_ROUNDS}) reached — accepting current answer.")
        return {"answer_passed": True, "total_tokens": state["total_tokens"]}

    structured_critic = llm.with_structured_output(CriticVerdict)
    verdict = structured_critic.invoke([
        SystemMessage(content=(
            "You are a strict QA critic. Review the solver's answer to the original question. "
            "Check for factual errors, logical mistakes, and completeness. "
            "Be rigorous — only pass if you are confident the answer is correct."
        )),
        *state["messages"],
    ])

    if verdict.passed:
        print("  [CriticLoop] Critic: PASS ✓")
        return {"answer_passed": True, "total_tokens": state["total_tokens"]}
    else:
        print(f"  [CriticLoop] Critic: FAIL — {verdict.feedback[:80]}")
        feedback_msg = HumanMessage(
            content=f"Your answer failed review. Fix these issues:\n{verdict.feedback}"
        )
        return {
            "messages":      [feedback_msg],
            "answer_passed": False,
            "total_tokens":  state["total_tokens"],
        }


# ── Formatter ──────────────────────────────────────────────────────────────────

def formatter_node(state: CriticLoopState) -> dict:
    """
    Runs once after the Critic accepts the answer.
    Strips everything down to the bare answer string for the evaluator.
    """
    print("  [CriticLoop] Formatter: extracting final answer...")
    llm = get_llm()

    response = llm.invoke([
        SystemMessage(content=(
            "You are a precise answer extractor. Read the conversation and extract "
            "ONLY the final answer — no explanation, no units unless asked, "
            "no punctuation beyond what the answer itself requires."
        )),
        *state["messages"],
    ])

    tokens = count_tokens(response)
    return {
        "messages":     [AIMessage(content=response.content, name="Formatter")],
        "total_tokens": state["total_tokens"] + tokens,
    }


# ── Router ─────────────────────────────────────────────────────────────────────

def should_continue(state: CriticLoopState) -> str:
    if state["answer_passed"]:
        return "Formatter"
    return "Solver"   # loop back


# ── Graph ──────────────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    workflow = StateGraph(CriticLoopState)

    workflow.add_node("Planner",   planner_node)
    workflow.add_node("Solver",    solver_node)
    workflow.add_node("Critic",    critic_node)
    workflow.add_node("Formatter", formatter_node)

    workflow.set_entry_point("Planner")
    workflow.add_edge("Planner", "Solver")
    workflow.add_edge("Solver",  "Critic")

    # The loop edge: Critic either sends back to Solver or exits to Formatter
    workflow.add_conditional_edges(
        "Critic",
        should_continue,
        {
            "Solver":    "Solver",
            "Formatter": "Formatter",
        }
    )

    workflow.add_edge("Formatter", END)

    return workflow.compile()


# ── Public interface ───────────────────────────────────────────────────────────

def run(question: str) -> dict:
    graph = _build_graph()

    initial_state: CriticLoopState = {
        "messages":      [HumanMessage(content=question)],
        "answer_passed": False,
        "rounds":        0,
        "total_tokens":  0,
    }

    with Timer() as t:
        final_state = graph.invoke(initial_state)

    answer = final_state["messages"][-1].content
    turns  = final_state["rounds"] + 2   # Planner + N Solver/Critic cycles + Formatter

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
