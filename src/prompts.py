"""LLM prompt templates for the self-correcting RAG agent."""

from __future__ import annotations

GRADE_RELEVANCE_PROMPT = """\
You are a retrieval quality grader for an Airbnb listing Q&A system.

Given a user question and retrieved context chunks, decide whether the context \
is sufficient to answer the question.

Question:
{question}

Retrieved context:
{context}

Respond with exactly one word:
- relevant   — context directly addresses the question
- partial    — some useful information but gaps remain
- irrelevant — context does not help answer the question
"""

REWRITE_QUERY_PROMPT = """\
You rewrite user questions to improve semantic search over Airbnb listings.

The original question did not retrieve useful context. Produce a clearer, \
search-friendly query that preserves the user's intent.

Original question: {question}
Previous search query: {search_query}

Return only the rewritten query, no explanation.
"""

GENERATE_ANSWER_PROMPT = """\
You are a helpful Airbnb listing assistant. Answer ONLY using the retrieved \
context below. Do not use outside knowledge.

Rules:
1. Ground every claim in the provided context.
2. If the context is insufficient, respond exactly: "I don't know based on the available listings."
3. Be concise and specific (location, amenities, guest feedback when available).
4. Do not invent listing names, IDs, or features not present in the context.

Question: {question}

Context:
{context}

Answer:
"""

EXTRACT_CITATIONS_PROMPT = """\
From the answer below, extract listing IDs that were actually used from the context.

Context listing IDs available: {available_ids}

Answer:
{answer}

Return a JSON array of listing_id strings cited or clearly referenced in the answer. \
If none, return [].
"""
