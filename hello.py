from langchain.tools import tool
from langchain_ollama import ChatOllama

from langchain.messages import HumanMessage, SystemMessage, AnyMessage, ToolMessage
from typing_extensions import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END

import operator

from dotenv import load_dotenv
load_dotenv()


# gpt4o_chat = ChatOpenAI(model="gpt-4o", temperature=0)
# gpt35_chat = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
model = ChatOllama(model="qwen3.5:4b", temperature=0)

# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b


# Augment the LLM with tools
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

def llm_call(state: dict):

    return {
        "messages": [
            model_with_tools.invoke(
                [SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                )]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

def tool_node(state: dict):
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


from typing import Literal
from langgraph.graph import StateGraph, START, END


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END

agent_builder = StateGraph(MessagesState)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

agent = agent_builder.compile()

if __name__ == "__main__":
    with open("hello.png", "wb") as f:
        f.write(agent.get_graph(xray=True).draw_mermaid_png())

    messages = [HumanMessage(content="Add 4 and 5")]
    messages = agent.invoke({"messages": messages})
    for m in messages["messages"]:
        m.pretty_print()