from langchain_core.messages import HumanMessage, SystemMessage

from agents.common import get_reasoning_model

SYSTEM_PROMPT = (
    "You are answering a multiple-choice question. "
    "Reason step by step, then end your response with exactly: 'Answer: X' "
    "where X is one of the option letters shown below."
)
# TODO move this prompt to mmlu


def format_question(question: str, options: list[str]) -> str:
    letters = "ABCDEFGHIJ"
    lines = [f"Question: {question}", "", "Options:"]
    for i, opt in enumerate(options):
        lines.append(f"  {letters[i]}. {opt}")
    return "\n".join(lines)


def answer_question(question: str, options: list[str], num_predict: int | None = None) -> str:
    """Invoke the planner model and return the raw response text."""
    model = get_reasoning_model(num_predict=num_predict)
    prompt = format_question(question, options)
    response = model.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )
    return response.content
