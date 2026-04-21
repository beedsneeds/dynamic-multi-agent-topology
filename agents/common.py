from typing import TypedDict, Annotated, List, Any
import operator
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler

load_dotenv()


REASONING_ACTION_LIST = [
    "reasoning",
    "critique",
    "question",
    "reflect",
    "conclude",
    "summarize",
    "planning",
    "modify",
]
TOOL_ACTION_LIST = ["search_arxiv", "search_bing", "access_website", "run_python", "read_file"]
TERMINATION_ACTION_LIST = ["terminate"]

# https://docs.langchain.com/oss/python/langgraph/graph-api#accessing-and-handling-the-recursion-counter
# TODO implement this for loops

# https://docs.langchain.com/oss/python/langgraph/graph-api#accessing-and-handling-the-recursion-counter
# TODO implement this for loops


class TokenUsageTracker:
    """Thread-safe-ish global tracker for benchmark token utilization."""

    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0

    def reset(self):
        self.input_tokens = 0
        self.output_tokens = 0

    def add(self, input_tokens: int, output_tokens: int):
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def get_usage(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
        }


usage_tracker = TokenUsageTracker()


class TokenTrackingHandler(BaseCallbackHandler):
    """Intercepts LLM results to update the global usage tracker."""

    def on_llm_end(self, response, **kwargs: Any) -> Any:
        for generation in response.generations:
            for g in generation:
                if hasattr(g.message, "usage_metadata") and g.message.usage_metadata:
                    usage = g.message.usage_metadata
                    usage_tracker.add(usage.get("input_tokens", 0), usage.get("output_tokens", 0))


tracking_callback = TokenTrackingHandler()


def get_judge_model(temperature: int = 0) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=temperature,
        callbacks=[tracking_callback],
    )


def get_reasoning_model(temperature: int = 0) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemma-4-31b-it",
        temperature=temperature,
        callbacks=[tracking_callback],
    )


# its not really that much smaller
def get_small_model(temperature: int = 0) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemma-4-26b-a4b-it",
        temperature=temperature,
        callbacks=[tracking_callback],
    )


# The planner model (qwen3.5:9b) is a reasoning-capable model and,
# left on defaults, produces long <think> blocks before answering
# — a single call takes 10+ minutes on CPU.
# num_predict caps output tokens: even with reasoning=False, a
# CoT prompt can push the model into a 2000+ token explanation
# that wedges a benchmark run on CPU. Cap from the call site.
# TODO try llama.cpp
def get_local_reasoning_model(temperature: int = 0, num_predict: int | None = None) -> ChatOllama:
    return ChatOllama(
        model="qwen3.5:9b",
        temperature=temperature,
        reasoning=False,
        num_predict=num_predict,
        callbacks=[tracking_callback],
    )


def get_local_small_model(temperature: int = 0, num_predict: int | None = None) -> ChatOllama:
    return ChatOllama(
        model="qwen3.5:4b",
        temperature=temperature,
        reasoning=False,
        num_predict=num_predict,
        callbacks=[tracking_callback],
    )


# Maybe a smaller model like ministral-3:3b for tool use. Its good for edge use cases
