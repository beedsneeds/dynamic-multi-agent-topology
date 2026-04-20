from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain.agents import create_agent

import ast
import operator

from graphlib import TopologicalSorter, CycleError
from typing import TYPE_CHECKING

# Avoid a circular import error since we need the PlannerOutput shape here
if TYPE_CHECKING:
    from agents.dynamic import PlannerOutput


_CALC_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_CALC_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _calc_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _CALC_UNARY_OPS:
        return _CALC_UNARY_OPS[type(node.op)](_calc_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _CALC_BIN_OPS:
        # Guard pow exponent: huge integer exponents can lock the CPU for
        # minutes on a runaway model. 100 is well above anything GSM-Hard needs.
        if isinstance(node.op, ast.Pow):
            right = _calc_eval(node.right)
            if isinstance(right, (int, float)) and abs(right) > 100:
                raise ValueError("exponent magnitude > 100 rejected")
            return _calc_eval(node.left) ** right
        return _CALC_BIN_OPS[type(node.op)](_calc_eval(node.left), _calc_eval(node.right))
    raise ValueError(f"unsupported expression element: {ast.dump(node)}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a numeric arithmetic expression and return the result.

    Use this for ANY arithmetic over multi-digit numbers — mental math by the
    model is unreliable. Supports +, -, *, /, //, %, **, unary -, and
    parentheses over int/float literals. No variables, no function calls.

    Args:
        expression: e.g. "2+2", "(3.5 * 17) - 4**2", "1000000 / 7".

    Returns:
        The numeric result as a string, or "ERROR: <reason>" if the
        expression is malformed or uses unsupported syntax.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_calc_eval(tree.body))
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def _worker_tools(role: str) -> list:
    # Keep tool construction lazy so missing API keys surface only when the
    # role that needs them is actually invoked, not at import time.
    if role == "researcher":
        return [TavilySearch(max_results=5)]
    if role == "executor":
        return [calculator]
    return []


_RESEARCHER_NO_TOOL_NUDGE = (
    "You produced HITS without calling tavily_search. Any hit not taken "
    "verbatim from a tavily_search result you made this turn is a "
    "fabrication. Call tavily_search now, then rewrite HITS from the "
    "actual results."
)

_EXECUTOR_NO_TOOL_NUDGE = (
    "You wrote an ACTIONS log without actually invoking calculator. Text "
    "like [calculator(...)] in your reply does not execute — any numeric "
    "result you reported is a fabrication. Call calculator now via the "
    "tool-calling interface, then rewrite ACTIONS and SUMMARY from the "
    "real return value."
)


def run_worker_once(role: str, task: str, user_input: str) -> str:
    """Execute one worker step and return its text output.

    Shared between chain and tree topologies; `agents.dynamic` inlines an
    equivalent path (with additional reviewer/revision handling). Keeping
    worker semantics identical across topologies — including the
    tool-call-or-retry guard for researcher/executor — means the topology
    itself is the only variable when comparing runs.
    """
    # Deferred imports: these modules are heavy and most dynamic_utils
    # callers (calculator, _worker_tools, validate_plan_deps) don't need
    # them. Also avoids a circular edge: agents.agent_role_prompts and
    # agents.common both live under `agents/` alongside this file.
    from agents.common import get_reasoning_model
    from agents.agent_role_prompts import role_prompt

    model = get_reasoning_model()
    system_prompt = role_prompt(role)

    user_messages = [
        HumanMessage(
            content=(
                f"Original user request (for context — do not answer it directly):\n"
                f"{user_input}\n\n"
                f"Your specific task:\n{task}"
            )
        )
    ]

    tools = _worker_tools(role)
    if not tools:
        response = model.invoke([SystemMessage(content=system_prompt), *user_messages])
        return response.content

    agent = create_agent(model=model, tools=tools, system_prompt=system_prompt)
    agent_result = agent.invoke({"messages": user_messages})

    # Tool-bound roles that emit output without any tool call are almost
    # always fabricating from parametric memory. One forced retry with a
    # pointed correction usually recovers; a second pass that still skips
    # the tool gets returned as-is (reviewer/synth can flag it).
    def _called_a_tool() -> bool:
        return any(bool(getattr(m, "tool_calls", None)) for m in agent_result["messages"])

    if role == "researcher" and not _called_a_tool():
        print("researcher emitted no tool calls — forcing retry")
        retry_messages = list(agent_result["messages"]) + [
            HumanMessage(content=_RESEARCHER_NO_TOOL_NUDGE)
        ]
        agent_result = agent.invoke({"messages": retry_messages})
    elif role == "executor" and not _called_a_tool():
        print("executor emitted no tool calls — forcing retry")
        retry_messages = list(agent_result["messages"]) + [
            HumanMessage(content=_EXECUTOR_NO_TOOL_NUDGE)
        ]
        agent_result = agent.invoke({"messages": retry_messages})

    return agent_result["messages"][-1].content


# Finds only structural errors in the plan
# TODO put this in a helper file
def validate_plan_deps(plan: "PlannerOutput") -> tuple[list[str], dict[str, set[str]]]:
    errors = []
    ids = [s["id"] for s in plan["steps"]]

    if len(ids) != len(set(ids)):
        errors.append("duplicate step ids")

    id_set = set(ids)
    graph: dict[str, set[str]] = {}
    for s in plan["steps"]:
        for dep in s["depends_on"]:
            if dep not in id_set:
                errors.append(f"step {s['id']} depends on unknown id {dep}")
        graph[s["id"]] = set(s["depends_on"])

    try:
        TopologicalSorter(graph).prepare()  # throws if cyclic
    except CycleError as e:
        errors.append(f"cycle: {e.args[1]}")

    return errors, graph
