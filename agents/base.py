"""
agents/base.py
--------------
Shared utilities for all topology agents.

Every topology's run() function must return an AgentResult dict:
    {
        "answer":      str,   # final answer string
        "time_ms":     int,   # wall-clock latency
        "tokens_used": int,   # total tokens (prompt + completion)
        "agent_turns": int,   # number of agent node executions
    }

run_pipeline() in stub_agent.py translates this into what the
evaluation layer expects (cost_usd, latency_ms, etc).
"""

import time
from typing import TypedDict, Annotated, List
import operator
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI


# ── Shared LLM ────────────────────────────────────────────────────────────────
# All topologies use the same base model. Swap here to change globally.
def get_llm(temperature: int = 0) -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)


# ── Shared State schema ────────────────────────────────────────────────────────
# Every topology's LangGraph state must include at minimum these two fields.
# Topologies can extend this with their own fields (e.g. code_passed, routing_decision).
class BaseState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]


# ── Token counter ──────────────────────────────────────────────────────────────
def count_tokens(response) -> int:
    """
    Extract total token usage from a LangChain LLM response.
    Returns 0 if metadata is unavailable (e.g. streaming or structured output).
    """
    try:
        usage = response.response_metadata.get("token_usage", {})
        return usage.get("total_tokens", 0)
    except Exception:
        return 0


# ── Cost helper ────────────────────────────────────────────────────────────────
# GPT-4o-mini pricing as of 2025. Update if model changes.
_COST_PER_TOKEN = {
    "gpt-4o-mini": 0.00000015,   # $0.15 per 1M tokens (blended prompt+completion estimate)
    "gpt-4o":      0.000005,
}

def tokens_to_cost(tokens: int, model: str = "gpt-4o-mini") -> float:
    rate = _COST_PER_TOKEN.get(model, 0.00000015)
    return round(tokens * rate, 6)


# ── Timer context manager ──────────────────────────────────────────────────────
class Timer:
    """Usage: with Timer() as t: ...  then t.elapsed_ms"""
    def __enter__(self):
        self._start = time.time()
        return self
    def __exit__(self, *_):
        self.elapsed_ms = int((time.time() - self._start) * 1000)


# ── Result builder ─────────────────────────────────────────────────────────────
def make_result(answer: str, elapsed_ms: int, tokens: int, turns: int) -> dict:
    """Constructs the standard AgentResult dict every topology must return."""
    return {
        "answer":      answer,
        "time_ms":     elapsed_ms,
        "tokens_used": tokens,
        "agent_turns": turns,
    }
