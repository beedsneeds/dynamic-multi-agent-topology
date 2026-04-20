"""Tree topology: fan-out-fan-in baseline for topology comparisons.

The Planner (root) decomposes the objective into ≤5 steps (same prompt
as `agents.dynamic`); every step is dispatched in parallel in a single
superstep via `Send`; the Synthesizer integrates the merged outputs at
the fan-in. `depends_on` from the Planner's output is intentionally
ignored — the defining feature of this topology is that all branches
run at once — as is `require_reviewer`, since no Reviewer node exists.

Relative to `agents.dynamic`, this topology removes:
  - DAG-aware wave dispatch (single wave, all branches in parallel)
  - Per-step Reviewer loop
  - Rolling-horizon replans and Steward check-ins

Relative to `agents.chain`, it trades sequential execution for full
parallelism, isolating the effect of wave-dispatch and DAG awareness
in cross-topology comparisons.

Fan-in works via the `completed_steps` reducer: each worker returns
`Command(goto="synthesizer", update={"completed_steps": [result]})`,
LangGraph merges the parallel writes across the superstep, then fires
the Synthesizer once with the full list.

Run shape:
    START → planner → [worker_1, ..., worker_k in parallel] → synthesizer → END
"""

import operator

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, Send
from typing_extensions import TypedDict, Annotated, Literal, NotRequired

from agents.common import get_reasoning_model
from agents.utils import run_worker_once
import agents.agent_role_prompts as prompts

load_dotenv()


class InputState(TypedDict):
    user_input: str
    # Optional directive appended to the Synthesizer's user message. Matches
    # the contract in agents.dynamic so benchmark callers can swap topologies
    # without changing how they shape the final output.
    synth_suffix: NotRequired[str]


class OutputState(TypedDict):
    graph_output: str


class Step(TypedDict):
    id: str
    task: str
    agent: Literal["researcher", "coder", "analyst", "executor", "planner"]
    tools: list[Literal["tavily_search", "calculator"]]
    # Retained to accept the shared Planner's output verbatim. Unused at
    # execution time — every step is dispatched in one parallel wave.
    depends_on: list[str]
    require_reviewer: bool


class PlannerOutput(TypedDict):
    objective: str
    steps: list[Step]
    # Retained for schema compatibility with the shared Planner; this
    # topology does not re-plan, so the flag is read but ignored.
    more_planning_needed: bool


class StepResult(TypedDict):
    id: str
    output: str


class OverallState(InputState, OutputState):
    plan: PlannerOutput
    completed_steps: Annotated[list[StepResult], operator.add]


# Send payload: 'planner to worker'
class WorkerInput(TypedDict):
    step: Step
    user_input: str


def planner(
    state: InputState,
) -> Command[Literal["researcher", "coder", "analyst", "executor", "synthesizer"]]:
    print("planner invoked")
    model = get_reasoning_model().with_structured_output(PlannerOutput, method="json_mode")
    response = model.invoke(
        [
            SystemMessage(content=prompts.role_prompt("planner")),
            HumanMessage(content=state["user_input"]),
        ]
    )
    print(response)

    steps = response.get("steps") or []
    if not steps:
        # Degenerate case: Planner returned no steps. Hand straight to
        # Synthesizer so the graph doesn't dangle; it'll emit the
        # "no work completed" response.
        return Command(goto="synthesizer", update={"plan": response})

    return Command(
        goto=[Send(s["agent"], {"step": s, "user_input": state["user_input"]}) for s in steps],
        update={"plan": response},
    )


def _worker_step(role: str, state: WorkerInput) -> Command[Literal["synthesizer"]]:
    step = state["step"]
    content = run_worker_once(role, step["task"], state["user_input"])
    result: StepResult = {"id": step["id"], "output": content}
    return Command(goto="synthesizer", update={"completed_steps": [result]})


def researcher(state: WorkerInput) -> Command[Literal["synthesizer"]]:
    print("researcher invoked")
    return _worker_step("researcher", state)


def coder(state: WorkerInput) -> Command[Literal["synthesizer"]]:
    print("coder invoked")
    return _worker_step("coder", state)


def analyst(state: WorkerInput) -> Command[Literal["synthesizer"]]:
    print("analyst invoked")
    return _worker_step("analyst", state)


def executor(state: WorkerInput) -> Command[Literal["synthesizer"]]:
    print("executor invoked")
    return _worker_step("executor", state)


def synthesizer(state: OverallState) -> Command[Literal["__end__"]]:
    print("synthesizer invoked")
    completed = state.get("completed_steps") or []
    if not completed:
        return Command(goto=END, update={"graph_output": "No work completed."})

    prior_plan = state.get("plan")
    objective_line = (
        f"Planner's restated objective: {prior_plan['objective']}\n\n" if prior_plan else ""
    )
    steps_block = "\n\n".join(f"[{item['id']}]\n{item['output']}" for item in completed)
    user_message = (
        f"User's original objective:\n{state['user_input']}\n\n"
        f"{objective_line}"
        f"Completed step outputs:\n{steps_block}"
    )
    suffix = state.get("synth_suffix") or ""
    if suffix:
        user_message = f"{user_message}\n\n{suffix}"

    model = get_reasoning_model()
    response = model.invoke(
        [
            SystemMessage(content=prompts.role_prompt("synthesizer")),
            HumanMessage(content=user_message),
        ]
    )
    return Command(goto=END, update={"graph_output": response.content})


def build_graph() -> StateGraph:
    graph_builder = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)

    # Nodes
    graph_builder.add_node("planner", planner)
    graph_builder.add_node("researcher", researcher)
    graph_builder.add_node("coder", coder)
    graph_builder.add_node("analyst", analyst)
    graph_builder.add_node("executor", executor)
    graph_builder.add_node("synthesizer", synthesizer)

    # Edges — every other transition is driven by Command(goto=...):
    graph_builder.add_edge(START, "planner")
    #   planner      to  [worker_1..worker_k (Send) | synthesizer]
    #   worker       to  synthesizer       (fan-in via completed_steps reducer)
    #   synthesizer  to  END

    return graph_builder.compile()


def run(query: str, synth_suffix: str = "") -> str:
    graph = build_graph()
    initial: InputState = {"user_input": query}
    if synth_suffix:
        initial["synth_suffix"] = synth_suffix
    result = graph.invoke(initial)

    with open("agents/tree.png", "wb") as f:
        f.write(graph.get_graph(xray=True).draw_mermaid_png())

    return result["graph_output"]


if __name__ == "__main__":
    # prompt = (
    #     "Compare how octopuses and honeybees each solve navigation, and "
    #     "highlight one mechanism each uses that the other does not."
    # )
    prompt = "How many attempts should you make to cannulate a patient before passing the job on to a senior colleague, according to the medical knowledge of 2020?"

    print(run(prompt))
