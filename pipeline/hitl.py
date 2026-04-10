# Why this file needed?
# - In the Production agent system, we never let agents make high-stakes decisions without a human checkpoint.

# How it works in the pipeline?
# - After an agent produces an answer, the gate checks a confidence signal. If confidence is below a threshold, it pauses and asks a human to review before the pipeline continues.

# Where it fits in the codebase?
# - In run_pipeline() inside stub_agent.py, after getting the agent result:

# TODO
#     result = run_single_agent(question)
#     gate = HITLGate(confidence_threshold=0.6)
#     result = gate.check(question, result)
#     return result


class HITLGate:
    """
    Pauses the pipeline and requests human review when agent confidence is low.

    Args:
        confidence_threshold: If agent confidence < this, trigger human review.
                              0.0 = never trigger, 1.0 = always trigger.
                              0.6 is a reasonable default.
        enabled: Set to False to skip HITL entirely (useful for automated runs).
    """

    def __init__(self, confidence_threshold: float = 0.6, enabled: bool = True):
        self.threshold = confidence_threshold
        self.enabled   = enabled
        self._review_count = 0   # track how many times humans intervened

    def check(self, question: str, pipeline_result: dict) -> dict:
        """
        Main entry point. Call this after every agent run.

        Checks if the result needs human review and handles it.
        Always returns a result dict with the same shape as the input, either the original or a human-corrected version.

        Args:
            question:        The original task question
            pipeline_result: The dict returned by run_pipeline()

        Returns:
            pipeline_result, possibly with "answer" updated by human reviewer
        """
        if not self.enabled:
            return pipeline_result

        confidence = self._estimate_confidence(pipeline_result)

        if confidence < self.threshold:
            print(f"\n[HITL] Low confidence ({confidence:.2f}) -> requesting human review")
            pipeline_result = self._request_review(question, pipeline_result, confidence)
            self._review_count += 1
        else:
            print(f"[HITL] Confidence {confidence:.2f} -> no review needed")

        # Always tag the result with the confidence score
        pipeline_result["confidence"] = confidence
        pipeline_result["hitl_reviewed"] = (confidence < self.threshold)

        return pipeline_result


    def _estimate_confidence(self, result: dict) -> float:
        """
        Estimates how confident the agent is about its answer.

        This is a heuristic, we don't have a real probability score from the LLM unless we use logprobs (advanced, not needed now).

        Current signals used:
          - Answer length: very long answers often mean the agent is rambling and unsure rather than giving a clean response
          - Hedging phrases: "I think", "possibly", "it might be" = low confidence
          - Very short answers: often mean the agent gave up

        TODO: In Stage 3, replace this with actual LLM logprobs or a dedicated confidence-scoring call.
        """
        answer = result.get("answer", "")

        # Signal 1: answer is suspiciously long (agent is explaining, not answering)
        if len(answer) > 300:
            return 0.3

        # Signal 2: answer contains hedging language
        hedging_phrases = [
            "i think", "i believe", "possibly", "it might", "not sure",
            "i'm not certain", "approximately", "i cannot", "i don't know",
            "unable to", "i'm unable"
        ]
        answer_lower = answer.lower()
        if any(phrase in answer_lower for phrase in hedging_phrases):
            return 0.4

        # Signal 3: empty or extremely short answer (gave up)
        if len(answer.strip()) < 2:
            return 0.1

        # Default: assume reasonable confidence
        return 0.8

    def _request_review(
        self, question: str, result: dict, confidence: float
    ) -> dict:
        """
        Handles the actual human review interaction.

        Stage 1 implementation: simple CLI prompt.
        The human can either accept the agent's answer or provide a correction.
        """
        print("HUMAN REVIEW REQUESTED")
        print(f"  Question:       {question[:100]}")
        print(f"  Agent answer:   {result['answer'][:100]}")
        print(f"  Confidence:     {confidence:.2f}")
        print(f"  Agent turns:    {result.get('agent_turns', '?')}")
        print(f"  Cost so far:    ${result.get('cost_usd', 0):.6f}")
        print("\n  Options:")
        print("  [Enter]        Accept the agent's answer")
        print("  [type answer]  Override with your own answer")
        print("  [skip]         Skip this task (mark as unanswered)")

        user_input = input("\n  Your decision: ").strip()

        if user_input == "" :
            print("  -> Accepted agent's answer.")
        elif user_input.lower() == "skip":
            result["answer"] = ""
            result["hitl_skipped"] = True
            print("  -> Task skipped.")
        else:
            original = result["answer"]
            result["answer"] = user_input
            result["hitl_override"] = True
            print(f"  -> Answer overridden: '{original[:50]}' -> '{user_input}'")

        print()
        return result

    @property
    def review_stats(self) -> dict:
        """Call after a batch run to see how often the HITL gate triggered."""
        return {"total_reviews": self._review_count}



if __name__ == "__main__":
    gate = HITLGate(confidence_threshold=0.6, enabled=True)

    # Test 1: high confidence answer should pass through without review
    print("Test 1: Short clean answer (should PASS)")
    result_clean = {"answer": "42", "cost_usd": 0.0001, "latency_ms": 500, "agent_turns": 1}
    result_clean = gate.check("What is 6 times 7?", result_clean)
    print(f"Result: {result_clean}\n")

    # Test 2: hedging answer — should trigger HITL review
    print("Test 2: Hedging answer (should TRIGGER review)")
    result_hedging = {"answer": "I think it might be around 42, but I'm not certain.", "cost_usd": 0.0002, "latency_ms": 800, "agent_turns": 1}
    result_hedging = gate.check("What is 6 times 7?", result_hedging)
    print(f"Result: {result_hedging}\n")

    print(f"Review stats: {gate.review_stats}")