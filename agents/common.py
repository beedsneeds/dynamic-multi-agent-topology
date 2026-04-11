from typing import TypedDict, Annotated, List
import operator
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama

    # The planner model (qwen3.5:9b) is a reasoning-capable model and,
    # left on defaults, produces long <think> blocks before answering
    # — a single call takes 10+ minutes on CPU. CommonGen doesn't need
    # chain-of-thought, so we disable reasoning for this benchmark.
    # Uses model_copy so we honour whatever base model/temperature
    # agents.common currently configures.
    # num_predict caps output tokens: even with reasoning=False, a
    # CoT prompt can push the model into a 2000+ token explanation
    # that wedges a benchmark run on CPU. Cap from the call site.
    # TODO try llama.cpp
def get_planner_model(temperature: int = 0, num_predict: int | None = None) -> ChatOllama:
    return ChatOllama(
        model="qwen3.5:9b",
        temperature=temperature,
        reasoning=False,
        num_predict=num_predict,
    )

def get_worker_model(temperature: int = 0, num_predict: int | None = None) -> ChatOllama:
    return ChatOllama(
        model="qwen3.5:4b",
        temperature=temperature,
        reasoning=False,
        num_predict=num_predict,
    )

# Maybe a smaller model like ministral-3:3b for tool use. Its good for edge use cases