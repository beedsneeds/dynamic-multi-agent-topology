# It takes any question and automatically picks the right topology for it.

# How it works?
#  1. Each topology has a plain-English descriptor: "good for X, Y, Z tasks"
#  2. When a question arrives, we embed both the question and all descriptors
#  3. We find which descriptor is semantically closest to the question
#  4. We return that topology


import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()


# Topology Descriptors = the "Keys" in Key-Query matching
TOPOLOGY_DESCRIPTORS: dict[str, str] = {
    "single": (
        "Simple, direct questions with a single clear answer. Factual lookups, basic arithmetic, straightforward definitions. Questions that need one step to answer and do not require verification. Examples: capital cities, simple calculations, vocabulary questions, yes/no questions, single-fact retrieval."
    ),

    "chain": (
        "Tasks that require writing code, debugging, or solving a problem that benefits from review and iteration. Programming tasks, technical problem solving, tasks where the first attempt may have errors that need fixing. Examples: write a Python function, implement an algorithm, create a script, fix a bug, write a solution with edge cases."
    ),

    "hub_spoke": (
        "Complex multi-step tasks that require planning, research, and synthesis. Tasks with multiple sub-questions, tasks needing both web search and computation, tasks requiring coordination of multiple specialized tools. Examples: research a topic and summarize findings, compare multiple options, build an application with testing, analyze data and report results, tasks with more than two distinct subtasks."
    ),
}


class TopologySelector:
    """
    Selects the best topology for a given task using semantic similarity between the task description and topology descriptors.
    """

    def __init__(self, descriptors: dict[str, str] = None):
        """
        Args:
            descriptors: Mapping of topology_name -> descriptor string.
                         Defaults to TOPOLOGY_DESCRIPTORS defined above.
                         Pass a custom dict to override for experiments.
        """
        self.descriptors = descriptors or TOPOLOGY_DESCRIPTORS
        self._embedder = OpenAIEmbeddings(model="text-embedding-3-small")

        # Pre-embed all descriptors once at init time, not on every call.
        print("[TopologySelector] Embedding topology descriptors")
        self._descriptor_embeddings = {
            name: self._embedder.embed_query(desc)
            for name, desc in self.descriptors.items()
        }
        print(f"[TopologySelector] Topologies: {list(self.descriptors.keys())}")

    def select(self, question: str, verbose: bool = False) -> str:
        """
        Selects the best topology for a given question.
        """
        # Embed the incoming question
        question_embedding = self._embedder.embed_query(question)

        # Compute cosine similarity against each topology descriptor
        scores = {}
        for name, desc_embedding in self._descriptor_embeddings.items():
            scores[name] = self._cosine_similarity(question_embedding, desc_embedding)

        # Pick the topology with the highest similarity score
        best_topology = max(scores, key=scores.get)

        if verbose:
            print(f"\n[TopologySelector] Question: '{question[:80]}'")
            for name, score in sorted(scores.items(), key=lambda x: -x[1]):
                marker = " <- selected" if name == best_topology else ""
                print(f"  {name:<15} similarity: {score:.4f}{marker}")

        return best_topology

    def select_with_scores(self, question: str) -> dict:
        """
        Same as select() but returns all scores, not just the winner.
        """
        question_embedding = self._embedder.embed_query(question)

        scores = {
            name: self._cosine_similarity(question_embedding, desc_emb)
            for name, desc_emb in self._descriptor_embeddings.items()
        }

        return {
            "selected": max(scores, key=scores.get),
            "scores":   {k: round(v, 4) for k, v in scores.items()},
            "question": question,
        }

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """
        Computes cosine similarity between two embedding vectors.

        We don't import numpy here to keep dependencies minimal.
        For production, numpy is faster. For our scale, this is fine.
        """
        dot   = sum(a * b for a, b in zip(vec_a, vec_b))
        mag_a = sum(a * a for a in vec_a) ** 0.5
        mag_b = sum(b * b for b in vec_b) ** 0.5

        if mag_a == 0 or mag_b == 0:
            return 0.0

        return dot / (mag_a * mag_b)


# Convenience function, used by run_pipeline() in stub_agent.py

_selector_instance = None   # module-level singleton - init once, reuse across calls

def select_topology(question: str, verbose: bool = False) -> str:
    """
    Module-level function so stub_agent.py can call select_topology(question) without managing the TopologySelector instance itself.
    """
    global _selector_instance
    if _selector_instance is None:
        _selector_instance = TopologySelector()
    return _selector_instance.select(question, verbose=verbose)



if __name__ == "__main__":
    selector = TopologySelector()

    test_questions = [
        # Expect: single
        "What is the capital of France?",
        "How many studio albums did The Beatles release?",
        "Give me a modular python code for a music player app with end to end tested on multiple edge cases.",

        # Expect: chain
        "Write a Python function to compute factorial of n.",
        "Give me Python code for this app along with proper end-to-end testing.",
        "Debug this function — it returns wrong results for negative inputs.",

        # Expect: hub_spoke
        "Research the top 5 electric car manufacturers and compare their market share, battery range, and pricing in a structured report.",
        "Build a web scraper that collects job listings, stores them in a database, and emails a daily digest.",
    ]

    print("\nTopology Routing Test:\n")
    for q in test_questions:
        result = selector.select_with_scores(q)
        print(f"Q: {q[:75]}")
        print(f"   - {result['selected'].upper()} | scores: {result['scores']}\n")