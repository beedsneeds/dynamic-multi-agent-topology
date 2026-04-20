from typing import TypedDict, Annotated, List
import operator
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

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


def get_reasoning_model(temperature: int = 0) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemma-4-31b-it",
        temperature=temperature,
    )


# its not really that much smaller
def get_small_model(temperature: int = 0) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemma-4-26b-a4b-it",
        temperature=temperature,
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
    )


def get_local_small_model(temperature: int = 0, num_predict: int | None = None) -> ChatOllama:
    return ChatOllama(
        model="qwen3.5:4b",
        temperature=temperature,
        reasoning=False,
        num_predict=num_predict,
    )


# Maybe a smaller model like ministral-3:3b for tool use. Its good for edge use cases
