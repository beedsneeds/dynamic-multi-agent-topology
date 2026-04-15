from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from typing_extensions import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from langgraph.types import Command, Send
from dataclasses import dataclass
import logging
import operator
from graphlib import TopologicalSorter, CycleError

from agents.common import get_reasoning_model, get_small_model

import agents.agent_role_prompts as prompts


class InputState(TypedDict):
    user_input: str


class OutputState(TypedDict):
    graph_output: str


class TriageDecision(TypedDict):
    route: Literal["responder", "planner"]
    reason: str


class Step(TypedDict):
    id: str
    task: str
    agent: Literal[
        "researcher", "coder", "analyst", "executor", "planner"
    ]  # TODO planner too if a step is complex enough to need breaking down
    tools: list[str]
    depends_on: list[str]
    require_reviewer: bool


class StepResult(TypedDict):
    id: str
    output: str


class PlannerOutput(TypedDict):
    objective: str
    steps: list[Step]
    # whether a step is parallelizable or serial is determined by depends_on
    # If its an empty list or contains only completed items: step can execute


class OverallState(InputState, OutputState):
    # Should I manually compress this context?
    plan: PlannerOutput
    completed_steps: Annotated[list[StepResult], operator.add]


# Private channel between orchestrator and planner when needs reviewing
class PlannerReview(OverallState):
    plan_errors: list[str]


# Routing with commmand rather than # def route_from_triage(state: TriageDecision) -> str: return state["route"]
def triage(state: InputState) -> Command[Literal["responder", "planner"]]:
    model = get_small_model().with_structured_output(TriageDecision)

    response = model.invoke(
        [
            SystemMessage(content=prompts.role_prompt("triage")),
            HumanMessage(content=state["user_input"]),
        ]
    )
    logging.debug(f'triage route={response["route"]} reason={response["reason"]}')
    print(f'triage route={response["route"]} reason={response["reason"]}')
    return Command(goto=response["route"])


# TODO escalate to planner with reason
def responder(state: InputState) -> OutputState:
    # DON'T USE STRUCTURED OUTPUT FOR OUTPUT, IT FAILS
    # You'll probably need it eventually for escalating
    model = get_small_model()

    response = model.invoke(
        [
            SystemMessage(content=prompts.role_prompt("responder")),
            HumanMessage(content=state["user_input"]),
        ]
    )
    return {"graph_output": response.content}


def planner(state: PlannerReview) -> PlannerOutput:
    # InputState is a sub-subclass
    model = get_reasoning_model().with_structured_output(PlannerOutput)

    messages = [
        SystemMessage(content=prompts.role_prompt("planner")),
        HumanMessage(content=state["user_input"]),
    ]

    # On retry, get structural errors so the planner can correct them.
    prior_errors = state.get("plan_errors") or []
    if prior_errors:
        messages.append(
            HumanMessage(
                content="Prior plan had errors — fix them:\n- " + "\n- ".join(prior_errors)
            )
        )

    response = model.invoke(messages)
    return {"plan": response}


# Finds only structural errors in the plan
# TODO put this in a helper
def validate_plan_deps(plan: PlannerOutput) -> tuple[list[str], dict[str, set[str]]]:
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
        TopologicalSorter(graph).prepare()  # raises if cyclic
    except CycleError as e:
        errors.append(f"cycle: {e.args[1]}")

    return errors, graph


# TODO replace with sythesizer
def assemble_output(state: OverallState) -> str:
    # Placeholder — replace with an LLM synthesis pass when you need one.
    return "\n\n".join(f"[{item['id']}]\n{item['output']}" for item in state["completed_steps"])


def orchestrator(state: PlannerReview) -> Command:
    plan = state["plan"]

    # Flag planner if there's structural inconsistencies - cycles, unknown ids, duplicates
    # Rebuilding graph at every invocation because the plan might have updated
    errors, graph = validate_plan_deps(plan)
    if errors:
        return Command(goto="planner", update={"plan_errors": errors})

    completed_ids = {item["id"] for item in state["completed_steps"]}
    by_id = {s["id"]: s for s in plan["steps"]}
    all_ids = set(by_id)

    # Done. TODO call synthesizer
    if completed_ids >= all_ids:
        return Command(
            goto=END,
            update={"graph_output": assemble_output(state)},
        )

    # Rebuild the topo sorter and determine what's ready to run next
    ts = TopologicalSorter(graph)
    ts.prepare()
    for sid in completed_ids:
        ts.done(sid)
    ready = [by_id[sid] for sid in ts.get_ready()]
    if ready:
        max_parallel = prompts.DEFAULTS["MAX_PARALLEL"]
        return Command(goto=[Send(s["agent"], {"step": s}) for s in ready[:max_parallel]])

    # Not done, nothing ready. Validator passed, so no cycle — but a step's deps
    # never resolved (worker failed silently, or plan expects a dep that no step
    # produces). Escalate to planner with a stateable reason. Replace this branch
    # with an LLM judgment call (retry / revise / wind down) when you wire it in.
    stuck = [sid for sid in all_ids - completed_ids]
    return Command(
        goto="planner",
        update={"plan_errors": [f"stuck: unresolved steps {stuck}"]},
    )


# def arbiter(state: InputState) -> OverallState:
#     # The arbiter must compare against hard-acceptance criteria and nudge movement towards that criteria
#     # constantly checks recursion limit proactively https://docs.langchain.com/oss/python/langgraph/graph-api#recursion-limit
#     # one extra super-step or two are fine but no redudant work should be continued
#     # While a critic evaluates the output and has a role in refinement, the arbiter holds the macro-view:
#     # "is this the right way to achieve our ends? convince me"

#     pass


# def sequential_output_node(state: OverallState) -> OutputState:
#     model = get_reasoning_model()
#     response = model.invoke(state["state"])
#     return {"graph_output": response.content}


# def sequential_worker_node(state: OverallState) -> OverallState:
#     model = get_reasoning_model()
#     response = model.invoke(state["state"])
#     return {"graph_output": response.content}


def build_graph() -> StateGraph:

    graph_builder = StateGraph(OverallState, input_schema=InputState)

    # Nodes
    graph_builder.add_node("triage", triage)
    graph_builder.add_node("responder", responder)
    graph_builder.add_node("planner", planner)

    # Initialization edges
    graph_builder.add_edge(START, "triage")
    # triage to responder / planner is a Command
    graph_builder.add_edge("responder", END)

    # THIS IS ONLY A PLACEHOLDER
    graph_builder.add_edge("planner", END)

    return graph_builder.compile()


# cant use context schema: context_schema
# The schema class that defines the runtime context.
# Use this to expose immutable context data to your nodes, like user_id, db_conn, etc


def run(query: str) -> str:
    graph = build_graph()
    result = graph.invoke({"user_input": query})

    with open("agents/dynamic.png", "wb") as f:
        f.write(graph.get_graph(xray=True).draw_mermaid_png())

    return result["graph_output"]


if __name__ == "__main__":
    print(run("what's a cat"))
    # with open("agents/dynamic.png", "wb") as f:
    #     f.write(graph.get_graph(xray=True).draw_mermaid_png())
