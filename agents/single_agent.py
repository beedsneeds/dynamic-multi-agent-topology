from langchain_core.messages import HumanMessage, SystemMessage

from agents.common import get_reasoning_model


def format_question(question: str, options: list[str]) -> str:
    letters = "ABCDEFGHIJ"
    lines = [f"Question: {question}", "", "Options:"]
    for i, opt in enumerate(options):
        lines.append(f"  {letters[i]}. {opt}")
    return "\n".join(lines)


def answer_question(
    question: str,
    options: list[str],
    system_prompt: str,
    num_predict: int | None = None,
) -> str:
    """Invoke the planner model and return the raw response text."""
    model = get_reasoning_model(num_predict=num_predict)
    prompt = format_question(question, options)
    response = model.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ]
    )
    return response.content


def run(query: str, synth_suffix: str = "") -> str:
    """Single-agent baseline: invoke the reasoning model once on the raw
    query and return its response. No Planner, no specialist workers, no
    Synthesizer — the model's answer IS the graph output.

    Purpose: the no-topology rung of the cross-topology ladder
    (single_agent → chain → tree → dynamic). Shares the
    `run(query, synth_suffix="")` signature of `agents.chain`,
    `agents.tree`, and `agents.dynamic` so a benchmark runner can swap it
    in without special-casing the call site.

    `synth_suffix` is appended to the user message — same role it plays
    elsewhere (shaping output format for the benchmark parser without
    polluting the objective the Planner would otherwise see). There's no
    system prompt here on purpose: this baseline is meant to isolate
    "what can the raw reasoning model do on this problem," so any
    formatting constraint should arrive via the caller's `synth_suffix`.
    """
    model = get_reasoning_model()
    user_message = query
    if synth_suffix:
        user_message = f"{user_message}\n\n{synth_suffix}"
    response = model.invoke([HumanMessage(content=user_message)])
    return response.content
