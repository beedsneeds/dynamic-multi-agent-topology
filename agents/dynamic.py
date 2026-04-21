from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
from dotenv import load_dotenv

from typing_extensions import TypedDict, Annotated, Literal, NotRequired
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from langgraph.types import Command, Send
from dataclasses import dataclass
import logging
import operator
from graphlib import TopologicalSorter

from agents.common import get_reasoning_model, get_small_model
from agents.utils import validate_plan_deps, _worker_tools

import agents.agent_role_prompts as prompts

load_dotenv()


class InputState(TypedDict):
    user_input: str
    # Optional directive appended to the synthesizer's user message. Lets callers
    # (e.g. benchmarks) shape the final output format without modifying the
    # general-purpose synthesizer prompt or leaking format rules into user_input,
    # which the planner also sees.
    synth_suffix: NotRequired[str]


class OutputState(TypedDict):
    graph_output: str


class Step(TypedDict):
    id: str
    task: str
    agent: Literal[
        "researcher", "coder", "analyst", "executor", "planner"
    ]  # TODO planner too if a step is complex enough to need breaking down
    # Keep this enum in sync with _worker_tools(). Add a value only when a
    # corresponding tool is bound for some role; the PLANNER prompt also
    # restates the enum so the model biases toward valid names.
    tools: list[Literal["tavily_search", "calculator"]]
    # whether a step is parallelizable or serial is determined by depends_on
    # If its an empty list or contains only completed items: step can execute
    depends_on: list[str]
    require_reviewer: bool


class PlannerOutput(TypedDict):
    objective: str
    steps: list[Step]
    # more_planning_needed: planner sets to true when it truncates the plan and wants
    # to be re-invoked with completed_steps[] once this chunk finishes (rolling horizon)
    more_planning_needed: bool


# TODO rename to step_id and subclass RejectedOutput
class StepResult(TypedDict):
    id: str
    output: str


class RejectedOutput(TypedDict):
    step_id: str
    output: str
    task: str
    feedback: str


class OverallState(InputState, OutputState):
    # Should I manually compress this context?
    plan: PlannerOutput
    completed_steps: Annotated[list[StepResult], operator.add]

    # Reviewer-escalation channel. Written by the reviewer after 2 failed
    # review rounds on a step; read by the planner to trigger review-driven replanning
    reviewer_escalations: NotRequired[list[str]]
    # Counts planner invocations triggered by reviewer escalation. Distinct from
    # rolling-horizon replan_count — this one bounds the planner ↔ reviewer
    # runoff that occurs when a worker can't satisfy the reviewer.
    reviewer_replan_count: NotRequired[int]
    # Outputs that the reviewer blocked. Carried so the planner can either
    # re-strategize around them or, at MAX_REVIEWER_REPLANS, package them into
    # a best-effort wind-down answer instead of re-issuing similar steps.
    reviewer_rejected_outputs: Annotated[list[RejectedOutput], operator.add]

    # Orchestrator-level escalations (structural or stuck)
    plan_errors: NotRequired[list[str]]
    # Rolling-horizon bookkeeping
    replan_count: NotRequired[int]  # mechanical replan cap for rolling horizon
    steward_attached: NotRequired[bool]
    steward_verdict: NotRequired[str]  # forwarded to planner on replan


# Send payload: 'orchestrator to planner' when plan needs reviewing
class PlannerReview(OverallState):
    plan_errors: list[str]


class ReviewState(InputState):
    step: Step
    revision: NotRequired[int]
    # Carried through worker to reviewer so the reviewer can evaluate whether
    # the new attempt addressed its prior feedback. Present only on revision > 0.
    prior_output: NotRequired[str]
    prior_feedback: NotRequired[str]
    # Outputs of steps listed in step["depends_on"]. The orchestrator filters
    # completed_steps down to direct dependencies and attaches them here so the
    # worker can see what prior steps produced — depends_on otherwise only
    # affects scheduling, not context.
    prior_outputs: NotRequired[list[StepResult]]


# Send payload: 'orchestrator to worker' during hand-off and 'reviewer to worker' on revision
class WorkerInput(ReviewState):
    feedback: NotRequired[str]


# Send payload: 'worker to reviewer'
class ReviewerInput(ReviewState):
    output: str


class ReviewerDecision(TypedDict):
    verdict: Literal["APPROVE", "REVISE"]
    feedback: str  # empty on APPROVE


class StewardVerdict(TypedDict):
    on_track: Literal["yes", "drifting", "stalled"]
    verdict: Literal["CONTINUE", "NUDGE", "REDIRECT", "WIND_DOWN"]
    feedback: str  # free-text guidance for the planner


def planner(state: PlannerReview) -> Command[Literal["orchestrator", "synthesizer"]]:
    print("planner invoked")

    prior_errors = state.get("plan_errors") or []
    reviewer_escalations = state.get("reviewer_escalations") or []
    reviewer_rejected = state.get("reviewer_rejected_outputs") or []

    reviewer_replans = state.get("reviewer_replan_count", 0) + (1 if reviewer_escalations else 0)
    max_reviewer_replans = prompts.DEFAULTS["MAX_REVIEWER_REPLANS"]

    # If hard cap is reached, hand the reviewer_rejected_outputs to the synthesizer
    if reviewer_replans > max_reviewer_replans and reviewer_rejected:
        return Command(
            goto="synthesizer",
            update={
                "plan_errors": [],
                "reviewer_escalations": [],
                "reviewer_replan_count": reviewer_replans,
                "steward_verdict": "",
            },
        )

    messages = [
        SystemMessage(content=prompts.role_prompt("planner")),
        HumanMessage(content=state["user_input"]),
    ]

    # Get orchestrator-driven errors so the planner can correct them
    if prior_errors:
        messages.append(
            HumanMessage(
                content="Prior plan had errors — fix them:\n- " + "\n- ".join(prior_errors)
            )
        )

    # Get reviewer-driven escalations so the planner can encode the failure context
    # into new step tasks (workers don't see prior plan history otherwise)
    if reviewer_rejected:
        # TODO should we remove the 800 haracter limit since we set num_predict to 1500?
        rejected_summary = "\n---\n".join(
            f"[{r['step_id']}] task: {r['task']}\n"
            f"reviewer feedback: {r['feedback']}\n"
            f"draft excerpt: {r['output'][:800]}"
            for r in reviewer_rejected
        )
        messages.append(
            HumanMessage(
                content=(
                    f"Prior steps were rejected by the Reviewer ({reviewer_replans} of "
                    f"{max_reviewer_replans} review-driven replans used). Materially "
                    f"change strategy — do not re-issue the same task. Encode the prior "
                    f"failure into the new step's `task` text so the worker knows what "
                    f"was tried and what to avoid.\n\nRejected drafts:\n{rejected_summary}"
                )
            )
        )

    # TODO truncation for now, replace with a summary field on StepResult.
    completed = state.get("completed_steps") or []
    prior_plan = state.get("plan")
    if completed and prior_plan:
        completed_summary = "\n".join(
            f"- {item['id']}: {item['output'][:1500]}" for item in completed
        )
        messages.append(
            HumanMessage(
                content=(
                    f"Prior plan objective: {prior_plan['objective']}\n"
                    f"Completed steps (id: output):\n{completed_summary}\n\n"
                    "Plan the next chunk to advance the objective. Do not re-do "
                    "completed work. Reuse completed step ids in `depends_on` only "
                    "if new steps genuinely depend on them."
                )
            )
        )

    steward_verdict = state.get("steward_verdict")
    if steward_verdict:
        messages.append(HumanMessage(content=f"Steward verdict on progress:\n{steward_verdict}"))

    # InputState is a sub-subclass
    model = get_reasoning_model().with_structured_output(PlannerOutput, method="json_mode")

    response = model.invoke(messages)
    return Command(
        goto="orchestrator",
        update={
            "plan": response,
            "plan_errors": [],
            "reviewer_escalations": [],
            "reviewer_replan_count": reviewer_replans,
            "steward_verdict": "",
        },
    )


# Terminal node
def synthesizer(state: OverallState) -> Command[Literal["__end__"]]:
    print("synthesizer invoked")
    completed = state.get("completed_steps") or []
    rejected = state.get("reviewer_rejected_outputs") or []
    if not completed and not rejected:
        return Command(goto=END, update={"graph_output": "No work completed."})

    prior_plan = state.get("plan")
    objective_line = (
        f"Planner's restated objective: {prior_plan['objective']}\n\n" if prior_plan else ""
    )

    parts = [f"User's original objective:\n{state['user_input']}\n", objective_line]

    if completed:
        steps_block = "\n\n".join(f"[{item['id']}]\n{item['output']}" for item in completed)
        parts.append(f"Completed step outputs:\n{steps_block}")

    # Graceful degradation path (winding down): occurs when max_reviewer_replans is reached
    # The drafts below were blocked by the reviewer. TODO treat their claims skeptically
    if rejected:
        rejected_block = "\n---\n".join(
            f"[{r['step_id']}] task: {r['task']}\n"
            f"reviewer feedback: {r['feedback']}\n"
            f"draft output:\n{r['output']}"
            for r in rejected
        )
        parts.append(
            "Reviewer-rejected drafts (wind-down): build a best-effort answer "
            "from these, but state explicitly which claims could not be verified "
            "and why. Do not fabricate sources to fill gaps. If the underlying "
            "finding is that the requested artifact does not exist in the form "
            "the question presumes, say so plainly.\n\n" + rejected_block
        )

    user_message = "\n\n".join(parts)

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


def orchestrator(
    state: PlannerReview,
) -> Command[
    Literal["planner", "steward", "researcher", "coder", "analyst", "executor", "synthesizer"]
]:
    plan = state["plan"]
    print("orchestrator invoked")
    print(plan)

    # Flag planner if there's structural inconsistencies - cycles, unknown ids, duplicates
    # Rebuilding graph at every invocation because the plan might have updated
    errors, graph = validate_plan_deps(plan)
    if errors:
        return Command(goto="planner", update={"plan_errors": errors})

    completed_ids = {item["id"] for item in state["completed_steps"]}
    by_id = {s["id"]: s for s in plan["steps"]}
    all_ids = set(by_id)

    # All steps that were planned are done. If planner flagged for more planning,
    # consult the Steward before re-planning (steward_attached is set here)
    if completed_ids >= all_ids:
        if plan.get("more_planning_needed"):
            # Mechanical hard cap on replans independent of the Steward's judgment
            # TODO more graceful degradation?
            if state.get("replan_count", 0) >= prompts.DEFAULTS["MAX_REPLANS"]:
                return Command(goto="synthesizer")
            return Command(
                goto="steward",
                update={
                    "steward_attached": True,
                    "replan_count": state.get("replan_count", 0) + 1,
                },
            )
        return Command(goto="synthesizer")

    # Rebuild the sorted graph and determine what's ready to run next.
    # graphlib only accepts done() on ids it has issued via get_ready(), so
    # we walk the waves: pull ready, mark completed ones done, and stop at
    # the first wave that surfaces pending (not-yet-completed) work.
    ts = TopologicalSorter(graph)
    ts.prepare()
    ready_ids: list[str] = []
    while True:
        batch = ts.get_ready()
        if not batch:
            break
        pending = [sid for sid in batch if sid not in completed_ids]
        for sid in batch:
            if sid in completed_ids:
                ts.done(sid)
        if pending:
            ready_ids = pending
            break
    ready = [by_id[sid] for sid in ready_ids]
    if ready:
        max_parallel = prompts.DEFAULTS["MAX_PARALLEL"]
        completed_by_id = {item["id"]: item for item in state["completed_steps"]}
        sends = []
        for s in ready[:max_parallel]:
            payload: WorkerInput = {"step": s, "user_input": state["user_input"]}
            prior_outputs = [completed_by_id[d] for d in s["depends_on"] if d in completed_by_id]
            if prior_outputs:
                payload["prior_outputs"] = prior_outputs
            sends.append(Send(s["agent"], payload))
        return Command(goto=sends)

    # Error State: Not completed yet nothing marked ready. But no cycle (validated), but possibly
    # a step's deps never resolved (worker failed silently or plan expects a dep that no step produces)
    # Escalate to planner with context
    # TODO Replace this branch with an LLM judgment call (retry / revise / wind down) when you wire it in.
    # TODO should this simply be list(all_ids - completed_ids)
    stuck = list(all_ids - completed_ids)
    return Command(
        goto="planner",
        update={
            "plan_errors": [
                f"Execution stalled. The following steps are planned but their dependencies never satisfied: {stuck}. Ensure depends_on lists valid step IDs from the current or prior chunks."
            ]
        },
    )


def _run_worker(role: str, state: WorkerInput) -> Command:
    step = state["step"]
    model = get_reasoning_model()
    system_prompt = prompts.role_prompt(role)

    prior_outputs = state.get("prior_outputs") or []
    prior_block = ""
    if prior_outputs:
        rendered = "\n\n".join(f"[{p['id']}]\n{p['output']}" for p in prior_outputs)
        prior_block = f"\n\nOutputs from prior steps your task depends on:\n{rendered}"

    user_messages = [
        HumanMessage(
            content=(
                f"Original user request (for context — do not answer it directly):\n"
                f"{state['user_input']}\n\n"
                f"Your specific task:\n{step['task']}"
                f"{prior_block}"
            )
        )
    ]
    # On revision rounds the worker needs the rejected draft AND the reviewer's
    # feedback — feedback alone tells it what's wrong but not what to keep.
    # Without this the worker often re-generates from scratch and re-introduces
    # the same fabrications the reviewer just flagged.
    prior_output = state.get("prior_output")
    feedback = state.get("feedback")
    if prior_output or feedback:
        revision_block = "(Revision round — your prior attempt was rejected.)"
        if prior_output:
            revision_block += f"\n\nYour prior output:\n{prior_output}"
        if feedback:
            revision_block += f"\n\nReviewer feedback to address:\n{feedback}"
        revision_block += (
            "\n\nProduce a new attempt that fixes the flagged issues. Keep what was "
            "correct; do not re-introduce what the reviewer blocked."
        )
        user_messages.append(HumanMessage(content=revision_block))

    tools = _worker_tools(role)
    if tools:
        agent = create_agent(model=model, tools=tools, system_prompt=system_prompt)
        agent_result = agent.invoke({"messages": user_messages})

        # Researchers that produce HITS without calling tavily_search are
        # fabricating sources from parametric memory. Retry once with an
        # explicit correction; if the second pass still doesn't search, let
        # the reviewer reject it.
        if role == "researcher" and not any(
            bool(getattr(m, "tool_calls", None)) for m in agent_result["messages"]
        ):
            print("researcher emitted no tool calls — forcing retry")
            retry_messages = list(agent_result["messages"]) + [
                HumanMessage(
                    content=(
                        "You produced HITS without calling tavily_search. Any "
                        "hit not taken verbatim from a tavily_search result you "
                        "made this turn is a fabrication. Call tavily_search "
                        "now, then rewrite HITS from the actual results."
                    )
                )
            ]
            agent_result = agent.invoke({"messages": retry_messages})

        # Executors that emit an ACTIONS log without calling the calculator
        # are hallucinating the tool's return value. Same retry pattern.
        if role == "executor" and not any(
            bool(getattr(m, "tool_calls", None)) for m in agent_result["messages"]
        ):
            print("executor emitted no tool calls — forcing retry")
            retry_messages = list(agent_result["messages"]) + [
                HumanMessage(
                    content=(
                        "You wrote an ACTIONS log without actually invoking "
                        "calculator. Text like [calculator(...)] in your reply "
                        "does not execute — any numeric result you reported is "
                        "a fabrication. Call calculator now via the tool-calling "
                        "interface, then rewrite ACTIONS and SUMMARY from the "
                        "real return value."
                    )
                )
            ]
            agent_result = agent.invoke({"messages": retry_messages})

        content = agent_result["messages"][-1].content
        # print("agent result messages 1")
        # print(agent_result["messages"])
        # print("agent result messages 2")
    else:
        response = model.invoke([SystemMessage(content=system_prompt), *user_messages])
        content = response.content
    print(content)

    if step["require_reviewer"]:
        reviewer_payload: ReviewerInput = {
            "step": step,
            "output": content,
            "revision": state.get("revision", 0),
            "user_input": state["user_input"],
        }
        if state.get("prior_output") is not None:
            reviewer_payload["prior_output"] = state["prior_output"]
        if state.get("prior_feedback") is not None:
            reviewer_payload["prior_feedback"] = state["prior_feedback"]
        if prior_outputs:
            reviewer_payload["prior_outputs"] = prior_outputs
        return Command(goto=Send("reviewer", reviewer_payload))

    result: StepResult = {"id": step["id"], "output": content}
    return Command(goto="orchestrator", update={"completed_steps": [result]})


def researcher(state: WorkerInput) -> Command[Literal["reviewer", "orchestrator"]]:
    print("researcher invoked \n \n")
    return _run_worker("researcher", state)


def coder(state: WorkerInput) -> Command[Literal["reviewer", "orchestrator"]]:
    print("coder invoked \n \n")
    return _run_worker("coder", state)


def analyst(state: WorkerInput) -> Command[Literal["reviewer", "orchestrator"]]:
    print("analyst invoked \n \n")
    return _run_worker("analyst", state)


def executor(state: WorkerInput) -> Command[Literal["reviewer", "orchestrator"]]:
    print("executor invoked \n \n")
    return _run_worker("executor", state)


def reviewer(
    state: ReviewerInput,
) -> Command[Literal["orchestrator", "planner", "researcher", "coder", "analyst", "executor"]]:
    step = state["step"]
    role_prompt_name = f"reviewer_{step['agent']}"

    user_content = f"Task: {step['task']}\n\nWorker output:\n{state['output']}"
    prior_output = state.get("prior_output")
    prior_feedback = state.get("prior_feedback")
    if prior_output is not None and prior_feedback is not None:
        user_content += (
            "\n\n(Revision round — the prior attempt was rejected.)\n"
            f"Prior output:\n{prior_output}\n\n"
            f"Prior feedback you gave:\n{prior_feedback}"
        )

    model = get_small_model().with_structured_output(ReviewerDecision, method="json_mode")
    decision = model.invoke(
        [
            SystemMessage(content=prompts.role_prompt(role_prompt_name)),
            HumanMessage(content=user_content),
        ]
    )
    print(decision)

    if decision["verdict"] == "APPROVE":
        result: StepResult = {"id": step["id"], "output": state["output"]}
        return Command(goto="orchestrator", update={"completed_steps": [result]})

    # REVISE. Cap at 2 failed reviews per step; then escalate to planner.
    revision = state.get("revision", 0) + 1
    if revision >= prompts.DEFAULTS["MAX_STEP_REVISIONS"]:
        rejection: RejectedOutput = {
            "step_id": step["id"],
            "task": step["task"],
            "output": state["output"],
            "feedback": decision["feedback"],
        }
        return Command(
            goto="planner",
            update={
                "reviewer_escalations": [
                    f"step {step['id']} failed review twice: {decision['feedback']}"
                ],
                "reviewer_rejected_outputs": [rejection],
            },
        )

    revise_payload: WorkerInput = {
        "step": step,
        "feedback": decision["feedback"],
        "revision": revision,
        "prior_output": state["output"],
        "prior_feedback": decision["feedback"],
        "user_input": state["user_input"],
    }
    if state.get("prior_outputs"):
        revise_payload["prior_outputs"] = state["prior_outputs"]
    return Command(goto=Send(step["agent"], revise_payload))


def steward(state: OverallState) -> Command[Literal["planner", "synthesizer"]]:
    model = get_reasoning_model().with_structured_output(StewardVerdict, method="json_mode")

    completed_summary = "\n".join(
        f"- {item['id']}: {item['output'][:2000]}" for item in state["completed_steps"]
    )
    state_report = (
        f"User objective: {state['user_input']}\n"
        f"Prior plan objective (restated): {state['plan']['objective']}\n"
        f"Completed steps (id: output):\n{completed_summary}\n"
        f"Replans used: {state.get('replan_count', 0)} of {prompts.DEFAULTS['MAX_REPLANS']}"
    )

    verdict = model.invoke(
        [
            SystemMessage(content=prompts.role_prompt("steward")),
            HumanMessage(content=state_report),
        ]
    )

    if verdict["verdict"] == "WIND_DOWN":
        return Command(goto="synthesizer")

    return Command(
        goto="planner",
        update={"steward_verdict": f"{verdict['verdict']}: {verdict['feedback']}"},
    )


def build_graph() -> StateGraph:

    graph_builder = StateGraph(OverallState, input_schema=InputState)

    # Nodes
    graph_builder.add_node("planner", planner)
    graph_builder.add_node("orchestrator", orchestrator)
    graph_builder.add_node("researcher", researcher)
    graph_builder.add_node("coder", coder)
    graph_builder.add_node("analyst", analyst)
    graph_builder.add_node("executor", executor)
    graph_builder.add_node("reviewer", reviewer)
    graph_builder.add_node("steward", steward)
    graph_builder.add_node("synthesizer", synthesizer)

    # Edges
    graph_builder.add_edge(START, "planner")
    # All other others is Command(goto=...)
    #   planner         to      orchestrator
    #   orchestrator    to      [workers  (Send) | planner | steward | synthesizer]
    #   worker          to      [reviewer (Send) | orchestrator]
    #   reviewer        to      [worker (Send, on REVISE) | orchestrator (on ACCEPT) | planner (after 2 fails)]
    #   steward         to      [planner | synthesizer (on WIND_DOWN)]
    #   synthesizer     to      END

    return graph_builder.compile()


def run(query: str, synth_suffix: str = "") -> str:
    graph = build_graph()
    initial: InputState = {"user_input": query}
    if synth_suffix:
        initial["synth_suffix"] = synth_suffix
    result = graph.invoke(initial)

    with open("agents/dynamic.png", "wb") as f:
        f.write(graph.get_graph(xray=True).draw_mermaid_png())

    return result["graph_output"]


if __name__ == "__main__":
    # prompt = "what's a cat"
    prompt = "How many attempts should you make to cannulate a patient before passing the job on to a senior colleague, according to the medical knowledge of 2020?"

    print(run(prompt))
