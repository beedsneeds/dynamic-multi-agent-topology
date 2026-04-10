"""
agents/single.py — Single Agent Topology
-----------------------------------------
The simplest possible topology: one LLM call, no agent loop.

    Question → [LLM] → Answer

This is the Stage 1 baseline. Every other topology is compared against this.
"""

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from agents.base import get_llm, count_tokens, Timer, make_result

load_dotenv()


def run(question: str) -> dict:
    """
    Single LLM call. No graph, no loop, no agents talking to each other.
    Used as the performance/cost baseline for all topology comparisons.
    """
    llm = get_llm()
    tokens = 0

    with Timer() as t:
        response = llm.invoke([
            SystemMessage(content=(
                "You are a precise question-answering assistant. "
                "Read the question carefully, reason through it, then respond with "
                "ONLY the final answer. No explanation, no working, no units unless "
                "the question explicitly asks for them. Just the bare answer."
            )),
            HumanMessage(content=question),
        ])
        tokens = count_tokens(response)

    return make_result(
        answer=response.content,
        elapsed_ms=t.elapsed_ms,
        tokens=tokens,
        turns=1,
    )


if __name__ == "__main__":
    result = run("What is the capital of Australia?")
    print(result)
