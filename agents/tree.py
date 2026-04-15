from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from typing_extensions import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END

from dataclasses import dataclass

from agents.common import get_reasoning_model, get_small_model


# ensures that the provided input matches the expected structure
class InputState(TypedDict):
    user_input: str
    require_separate_context: bool


# ensures that the generated output matches the expected structure
# filters all other data
class OutputState(TypedDict):
    graph_output: str


class IsolatedState(TypedDict):
    i_state: str


class OverallState(InputState, OutputState):
    additional_state: str


"""
We're limiting depth here
"""


def hierarchical_root(state: InputState) -> OverallState:
    # if the task is complex AND there's a fairly isolated but well-defined task that might exceed context window,
    # spawn another child
    model = get_reasoning_model()
    response = model.invoke(state["user_input"])
    return {"state": response.content}


def hierarchical_child_1(state: OverallState) -> OverallState:
    model = get_small_model()
    response = model.invoke(state["state"])
    return {"graph_output": response.content}


def hierarchical_child_2(state: OverallState) -> OverallState:
    model = get_small_model()
    response = model.invoke(state["state"])
    return {"graph_output": response.content}


def hierarchical_synthesizer(state: OverallState) -> OutputState:
    model = get_reasoning_model()
    response = model.invoke(state["state"])
    return {"graph_output": response.content}


graph_builder = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
graph_builder.add_node("hierarchical_root", hierarchical_root)
graph_builder.add_node("hierarchical_child_1", hierarchical_child_1)
graph_builder.add_node("hierarchical_child_2", hierarchical_child_2)
graph_builder.add_node("hierarchical_synthesizer", hierarchical_synthesizer)
# Edges
graph_builder.add_edge(START, "hierarchical_root")
graph_builder.add_edge("hierarchical_root", "hierarchical_child_1")
graph_builder.add_edge("hierarchical_root", "hierarchical_child_2")
graph_builder.add_edge("hierarchical_child_1", "hierarchical_synthesizer")
graph_builder.add_edge("hierarchical_child_2", "hierarchical_synthesizer")
graph_builder.add_edge("hierarchical_synthesizer", END)


graph = graph_builder.compile()


# for children name them subagents


if __name__ == "__main__":
    with open("agents/tree.png", "wb") as f:
        f.write(graph.get_graph(xray=True).draw_mermaid_png())
