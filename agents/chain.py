from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from typing_extensions import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END

from dataclasses import dataclass

from agents.common import get_reasoning_model, get_small_model

# Schema, using dataclass over TypedDict so we can assign default values
# @dataclass

"""
Splitting state to prevent the exhausting the final LLM's context window
"""


# ensures that the provided input matches the expected structure
class InputState(TypedDict):
    user_input: str
    require_worker: bool


# ensures that the generated output matches the expected structure
# filters all other data
class OutputState(TypedDict):
    graph_output: str


class OverallState(InputState, OutputState):
    state: str


def planner(state: InputState) -> OverallState:
    # if the task is complex AND there's well-defined phases,
    # send to intermediate agent, else pass to output agent
    # Is this possible: list tasks and tools to equip the intermediate agent with

    model = get_reasoning_model()
    response = model.invoke(state["user_input"])
    return {"state": response.content}


# TODO what's a better name for this?
def executor(state: OverallState) -> OverallState:
    model = get_small_model()
    response = model.invoke(state["state"])
    return {"graph_output": response.content}


def synthesizer(state: OverallState) -> OutputState:
    model = get_reasoning_model()
    response = model.invoke(state["state"])
    return {"graph_output": response.content}


graph_builder = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
graph_builder.add_node("planner", planner)
graph_builder.add_node("executor", executor)
graph_builder.add_node("synthesizer", synthesizer)
# Edges
graph_builder.add_edge(START, "planner")
graph_builder.add_edge("planner", "executor")
graph_builder.add_edge("executor", "synthesizer")
graph_builder.add_edge("synthesizer", END)


graph = graph_builder.compile()


# for children name them subagents


if __name__ == "__main__":
    with open("agents/chain.png", "wb") as f:
        f.write(graph.get_graph(xray=True).draw_mermaid_png())


# def require_worker(state: OverallState):
#     if state["require_worker"] is True:
#         return True
#     else:
#         return False


# graph_builder = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
# graph_builder.add_node("planner_node", planner_node)
# graph_builder.add_node("synthesizer_node", synthesizer_node)
# graph_builder.add_edge(START, "planner_node")
# graph_builder.add_edge("synthesizer_node", END)
# # Add conditional node and edges
# graph_builder.add_node("sequential_worker_node", sequential_worker_node)
# graph_builder.add_edge("sequential_worker_node", "synthesizer_node")
# graph_builder.add_conditional_edges(
#     "planner_node",
#     require_worker,
#     {True: "sequential_worker_node", False: "synthesizer_node"},
# )
