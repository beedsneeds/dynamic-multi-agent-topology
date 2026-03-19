import os
import re
from dataclasses import dataclass
from typing import Optional
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

@dataclass
class GAIATask:
    task_id: str
    question: str
    ground_truth: str       # The exact answer we compare against
    level: int              # 1, 2, or 3
    annotator_steps: str    # How a human would solve it
    file_name: Optional[str] = None


class GAIALoader:
    """Loads and manages GAIA benchmark tasks."""

    def __init__(self, level: int = 1):
        assert level in (1, 2, 3), "GAIA has levels 1, 2, and 3 only"
        self.level = level
        self._dataset = None


    def _load(self):
        if self._dataset is not None:
            return
        
        print(f"[GAIA] Loading Level {self.level} tasks from HuggingFace.")

        self._dataset = load_dataset(
            "gaia-benchmark/GAIA",
            f"2023_level{self.level}",
            token=os.getenv("HF_TOKEN"),
        )

        self._raw = self._dataset["validation"]
        print(f"[GAIA] Loaded {len(self._raw)} Level {self.level} validation tasks.")


    def get_tasks(self, n: int = 10) -> list[GAIATask]:     # n = 10 is just for testing and to keep API costs low. We use n=50+ for actual benchmark runs.
        self._load()

        tasks = []
        for raw in self._raw.select(range(min(n, len(self._raw)))):
            tasks.append(GAIATask(
                task_id       = raw["task_id"],
                question      = raw["Question"],
                ground_truth  = raw["Final answer"],
                level         = raw["Level"],
                annotator_steps = raw.get("Annotator Metadata", {}).get("Steps", ""),
                file_name     = raw.get("file_name") or None,
            ))

        return tasks
    

    def get_task_by_id(self, task_id: str) -> Optional[GAIATask]:
        """Use to re-run a specific task that fail."""
        self._load()
        for raw in self._raw:
            if raw["task_id"] == task_id:
                return GAIATask(
                    task_id       = raw["task_id"],
                    question      = raw["Question"],
                    ground_truth  = raw["Final answer"],
                    level         = raw["Level"],
                    annotator_steps = raw.get("Annotator Metadata", {}).get("Steps", ""),
                    file_name     = raw.get("file_name") or None,
                )
        return None
    


def normalize_answer(raw: str) -> str:
    """
    TODO Add more normalization rules as we encounter edge cases. Common ones:
      - Currency: "$1.5 million" = "1500000"
      - Dates: "January 5th" = "january 5"
      - Units: "5895m" = "5895"
    """
    answer = raw.strip().lower()
    answer = re.sub(r"[,\$]", "", answer)  # remove commas and dollar signs
    answer = re.sub(r"\s+", " ", answer)   # collapse whitespace

    # Word-to-digit mapping for common GAIA answers
    word_to_num = {
        "zero": "0", "one": "1", "two": "2", "three": "3",
        "four": "4", "five": "5", "six": "6", "seven": "7",
        "eight": "8", "nine": "9", "ten": "10"
    }
    if answer in word_to_num:
        answer = word_to_num[answer]
        
    # TODO: add more rules here as we encounter them
    return answer


def score_answer(agent_output: str, task: GAIATask) -> dict:
    """Compares agent output against GAIA ground truth."""
    agent_norm = normalize_answer(agent_output)
    truth_norm = normalize_answer(task.ground_truth)

    # TODO: Right now this is exact match after normalization.
    # Some GAIA tasks have answers where partial credit makes sense.
    # For Stage 1, exact match is fine. Flagging this for Stage 3 analysis.
    is_correct = agent_norm == truth_norm

    return {
        "correct":      is_correct,
        "agent_answer": agent_norm,
        "ground_truth": truth_norm,
        "task_id":      task.task_id,
        "level":        task.level,
    }



if __name__ == "__main__":
    loader = GAIALoader(level=1)
    tasks = loader.get_tasks(n=3)

    print(f"\nSample GAIA Level 1 Tasks:\n")
    for i, task in enumerate(tasks, 1):
        print(f"Task {i} [{task.task_id[:8]}]")
        print(f"  Q: {task.question[:120]}")
        print(f"  A: {task.ground_truth}")
        print()

    # Test the scorer
    fake_output = tasks[0].ground_truth
    result = score_answer(fake_output, tasks[0])
    print(f"Scorer test (should be correct=True): {result}")