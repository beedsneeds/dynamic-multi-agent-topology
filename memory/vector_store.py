# Why we need this file?
# - Before an agent tries to answer a question, it first looks up similar questions it has answered successfully before and uses those past solutions as context.

import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()


class AgentMemoryStore:
    """
    Stores and retrieves past successful agent solutions using ChromaDB.

    LIFECYCLE OF A SOLUTION:
      1. Agent answers a question and gets scored as correct
      2. store_solution() saves: (question, answer, topology, score)
      3. Next time a similar question comes in:
         retrieve_similar() finds the past solution
      4. Agent receives the past solution as additional context, less likely to hallucinate on similar problems
    """

    COLLECTION_NAME = "agent_solutions"


    def __init__(self, persist_dir: str = "./memory/chroma_db"):
        """
        Args:
            persist_dir: Where ChromaDB stores its data on disk.
        """
        os.makedirs(persist_dir, exist_ok=True)

        # PersistentClient saves to disk, survives between Python runs
        self._client = chromadb.PersistentClient(path=persist_dir)

        # OpenAI embeddings for semantic search
        self._embed_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small",
        )

        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self._embed_fn,
            metadata={"description": "Successful agent solutions for RAG retrieval"},
        )

        count = self._collection.count()
        print(f"[Memory] ChromaDB ready. {count} solutions stored.")


    def store_solution(self, question: str, answer: str, topology: str, score: float, task_id: str = None,) -> None:
        """
        Stores a successful solution in the memory bank.

        Only store solutions with score=1.0 (correct answers).
        """
        if score < 1.0:
            return  # Never store wrong answers as reference material

        # Use task_id as the document ID if available, otherwise hash the question
        doc_id = task_id or str(abs(hash(question)))

        # The document text is what gets embedded and searched
        # We embed the question so future similar questions find this solution
        document_text = question

        try:
            self._collection.add(
                ids=[doc_id],
                documents=[document_text],
                metadatas=[{
                    "question": question[:500],   # truncate for metadata storage
                    "answer":   answer[:500],
                    "topology": topology,
                    "score":    score,
                }],
            )
        except Exception:
            # ChromaDB raises if ID already exists, update instead
            self._collection.update(
                ids=[doc_id],
                documents=[document_text],
                metadatas=[{
                    "question": question[:500],
                    "answer":   answer[:500],
                    "topology": topology,
                    "score":    score,
                }],
            )


    def retrieve_similar(self, question: str, n: int = 3) -> list[dict]:
        """
        Finds the n most semantically similar past solutions.

        Returns:
            List of dicts, each with: question, answer, topology, similarity
            Sorted by similarity.
        """
        count = self._collection.count()
        if count == 0:
            return []  # nothing stored yet

        results = self._collection.query(
            query_texts=[question],
            n_results=min(n, count),   # can't request more than what's stored
            include=["metadatas", "distances"],
        )

        # ChromaDB returns distances (lower = more similar)
        # Convert to similarity scores (higher = more similar)
        similar = []
        for metadata, distance in zip(
            results["metadatas"][0],
            results["distances"][0],
        ):
            similar.append({
                "question":   metadata["question"],
                "answer":     metadata["answer"],
                "topology":   metadata["topology"],
                "similarity": round(1 - distance, 4),  # convert distance to similarity
            })

        return similar

    def build_rag_context(self, question: str, n: int = 2) -> str:
        """
        Builds a context string from past solutions to inject into agent prompts.

        The agent sees this before seeing the current question, so it can reference past solutions when reasoning.

        Args:
            question: The current question
            n: How many past solutions to include in context

        Returns:
            A formatted string ready to prepend to the agent's system prompt.
            Returns empty string if no similar solutions found.
        """
        similar = self.retrieve_similar(question, n=n)

        if not similar:
            return ""  # no context available — agent works from scratch

        lines = ["Here are similar problems that were solved correctly before:"]
        for s in similar:
            if s["similarity"] > 0.7:    # only include high-similarity results
                lines.append("---")
                lines.append(f"Problem: {s['question']}")
                lines.append(f"Solution: {s['answer']}")

        if len(lines) == 1:
            return ""  # nothing cleared the similarity threshold

        lines.append("---")
        lines.append("Use these as reference if helpful, but reason independently.")

        return "\n".join(lines)

    def count(self) -> int:
        """Returns how many solutions are currently stored."""
        return self._collection.count()

    def clear(self) -> None:
        """
        Wipes all stored solutions.
        Use this between experiments to get a clean baseline.
        Don't call this in production.
        """
        self._client.delete_collection(self.COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self._embed_fn,
        )
        print("[Memory] Cleared all stored solutions.")




if __name__ == "__main__":
    print("Testing AgentMemoryStore\n")

    memory = AgentMemoryStore(persist_dir="./memory/chroma_db_test")

    # Store some fake solutions
    memory.store_solution(
        question="What is the capital of France?",
        answer="Paris",
        topology="single",
        score=1.0,
        task_id="test-001",
    )
    memory.store_solution(
        question="Write a Python function to compute the factorial of n.",
        answer="def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
        topology="chain",
        score=1.0,
        task_id="test-002",
    )
    memory.store_solution(
        question="This answer was wrong and should not be stored.",
        answer="wrong answer",
        topology="single",
        score=0.0,
        task_id="test-003",
    )

    print(f"\nStored solutions: {memory.count()}\n")

    # Retrieve similar
    print("Retrieval test")
    results = memory.retrieve_similar("What is France's capital city?", n=2)
    for r in results:
        print(f"  similarity: {r['similarity']} | answer: {r['answer']}")

    # Build RAG context
    print("\nRAG context test")
    context = memory.build_rag_context("Compute n factorial in Python")
    print(context if context else "(no context, similarity threshold not met)")

    # Cleanup test DB
    memory.clear()
    print("\nTest complete.")