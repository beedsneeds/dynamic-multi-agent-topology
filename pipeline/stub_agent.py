# This is a dummy agent that does nothing. It is used for testing purposes. Agents will be developed by Viraj and replaced with this stub agent when they are ready.

import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

def run_single_agent(question: str) -> dict:
    """
    The simplest possible "agent" - just a single LLM call.

    Viraj will replace this with a proper LangGraph graph.
    Evaluation code calls run_pipeline() which calls this.
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",   # trial model just for testing
        temperature=0,          # deterministic answers for reproducibility
    )

    start_ms = time.time() * 1000

    # TODO (Viraj): Replace this single LLM call with a LangGraph graph.
    # The graph will have multiple agents (coder, evaluator, critic).
    response = llm.invoke([
    SystemMessage(content=(
        "You are a precise question-answering assistant. "
        "Read the question carefully, reason through it, then respond with "
        "ONLY the final answer. No explanation, no working, no units unless "
        "the question asks for them. Just the bare answer."
    )),
    HumanMessage(content=question)
])

    end_ms = time.time() * 1000

    # TODO: Get real token costs from the response metadata.
    # llm.invoke() returns a message with usage_metadata if we configure it.
    # For now, this is a rough estimate based on gpt-4o-mini pricing.
    estimated_cost = 0.0001  # placeholder - ~$0.0001 per call at gpt-4o-mini rates

    return {
        "answer":      response.content,
        "cost_usd":    estimated_cost,
        "latency_ms":  int(end_ms - start_ms),
        "agent_turns": 1,   # single agent = 1 turn. Multi-agent will be higher.
    }


# This is the function evaluators always call.
# TODO: Update run_pipeline() once Viraj builds the real graph.
# and all the evaluation code stays untouched.
def run_pipeline(question: str, topology: str = "single") -> dict:
    if topology == "single":
        return run_single_agent(question)

    # TODO: add cases as Viraj builds them:
    # elif topology == "chain":
    #     from agents.chain import run_chain
    #     return run_chain(question)
    else:
        raise ValueError(f"Unknown topology: '{topology}'. "
                         f"Available: 'single'. More coming in Stage 2.")


if __name__ == "__main__":
    result = run_pipeline("What is 2 + 2?", topology="single")
    print(result)