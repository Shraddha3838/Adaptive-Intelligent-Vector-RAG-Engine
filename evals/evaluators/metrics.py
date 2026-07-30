"""
Code evaluators for the RAG agent: recall@k, groundedness, refusal-correctness, citation accuracy.

All evaluators are deterministic — no LLM calls.
"""

from __future__ import annotations

from typing import Any

from src.models import AgentResponse, RetrievedDocument


def recall_at_k(
    response: AgentResponse,
    relevant_listing_ids: list[str],
    k: int = 5,
) -> float:
    """
    Compute recall@k: what fraction of relevant listings were retrieved.

    Args:
        response: AgentResponse from a RAG agent run.
        relevant_listing_ids: Ground-truth listing IDs that should be retrieved.
        k: Top-k documents to consider (default 5).

    Returns:
        Recall score between 0.0 and 1.0.
    """
    if not relevant_listing_ids:
        return 1.0  # No relevant docs to retrieve — trivially correct

    retrieved_ids = set()
    for doc in response.documents[:k]:
        retrieved_ids.add(doc.listing_id)

    if not retrieved_ids:
        return 0.0

    relevant_set = set(relevant_listing_ids)
    hits = relevant_set & retrieved_ids
    return len(hits) / len(relevant_set)


def groundedness_score(response: AgentResponse) -> float:
    """
    Compute groundedness: fraction of cited listing IDs that were actually retrieved.

    A citation is grounded if the cited listing_id appears in the retrieved documents.

    Returns:
        Score between 0.0 and 1.0.
    """
    if not response.cited_listing_ids:
        return 1.0  # No citations to check — trivially grounded

    retrieved_ids = {doc.listing_id for doc in response.documents}
    if not retrieved_ids:
        return 0.0

    grounded = sum(
        1 for cid in response.cited_listing_ids if cid in retrieved_ids
    )
    return grounded / len(response.cited_listing_ids)


def citation_accuracy(response: AgentResponse) -> float:
    """
    Compute citation accuracy: fraction of citations that are correctly attributed.

    A citation is accurate if the cited listing_id appears in both the retrieved
    documents AND the answer text references the listing name or ID.

    Returns:
        Score between 0.0 and 1.0.
    """
    if not response.citations:
        return 1.0  # No citations — trivially correct

    answer_lower = response.answer.lower()
    retrieved_ids = {doc.listing_id for doc in response.documents}

    accurate = 0
    for citation in response.citations:
        if citation.listing_id not in retrieved_ids:
            continue
        # Check if listing name or ID appears in answer
        name_in_answer = (
            citation.listing_name.lower() in answer_lower
            if citation.listing_name
            else False
        )
        id_in_answer = citation.listing_id in answer_lower
        if name_in_answer or id_in_answer:
            accurate += 1

    return accurate / len(response.citations)


def refusal_correctness(response: AgentResponse) -> float:
    """
    Evaluate whether the agent correctly refused when it should have.

    Returns 1.0 if:
      - The question has no answer AND the agent refused, OR
      - The question has an answer AND the agent did NOT refuse.
    Returns 0.0 otherwise.

    Note: This evaluator needs to know whether the question is a no-answer case.
    It returns 1.0 by default if we can't determine the expected behavior.
    """
    # This is a placeholder — the actual refusal check is done in run_evals.py
    # where we know the category from the dataset.
    return 1.0


def compute_refusal_correctness(
    response: AgentResponse,
    is_no_answer: bool,
) -> float:
    """
    Compute refusal correctness given knowledge of whether the question is no-answer.

    Args:
        response: AgentResponse from the RAG agent.
        is_no_answer: True if the question has no answer in the corpus.

    Returns:
        1.0 if correct, 0.0 if incorrect.
    """
    if is_no_answer:
        return 1.0 if response.refused else 0.0
    else:
        return 0.0 if response.refused else 1.0


def compute_all_metrics(
    response: AgentResponse,
    relevant_listing_ids: list[str],
    is_no_answer: bool,
    recall_k: int = 5,
) -> dict[str, float]:
    """
    Compute all evaluation metrics for a single agent response.

    Args:
        response: AgentResponse from the RAG agent.
        relevant_listing_ids: Ground-truth relevant listing IDs.
        is_no_answer: Whether the question has no answer in the corpus.
        recall_k: k for recall@k.

    Returns:
        Dictionary of metric names to scores.
    """
    return {
        "recall_at_k": recall_at_k(response, relevant_listing_ids, k=recall_k),
        "groundedness": groundedness_score(response),
        "citation_accuracy": citation_accuracy(response),
        "refusal_correctness": compute_refusal_correctness(response, is_no_answer),
    }