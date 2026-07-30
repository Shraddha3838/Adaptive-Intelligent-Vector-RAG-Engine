"""
LangGraph self-correcting RAG agent.

Nodes: retrieve -> grade_relevance -> (rewrite -> retrieve)* -> generate -> cite

Uses Groq (free) for chat inference and HuggingFace (local) for embeddings.

Usage:
    python -m src.graph "quiet apartment near the beach with wifi"
    python -m src.graph "beach wifi" --stream
"""

from __future__ import annotations

import argparse
import json
from typing import Any, AsyncIterator, Iterator, Literal

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from typing_extensions import TypedDict

from src.config import Settings, get_settings
from src.constants import REFUSAL_PHRASES
from src.models import AgentResponse, Citation, RelevanceGrade, RetrievedDocument
from src.prompts import (
    GENERATE_ANSWER_PROMPT,
    GRADE_RELEVANCE_PROMPT,
    REWRITE_QUERY_PROMPT,
)
from src.retrieval import VectorRetriever
from src.utils import get_logger, log_duration, setup_logging

logger = get_logger("graph")


class AgentState(TypedDict, total=False):
    """Typed state passed between LangGraph nodes."""

    question: str
    search_query: str
    retrieval_attempts: int
    rewritten_queries: list[str]
    documents: list[RetrievedDocument]
    context: str
    citations: list[Citation]
    listing_ids: list[str]
    relevance_grade: str
    answer: str
    cited_listing_ids: list[str]
    refused: bool


def _is_refusal(answer: str) -> bool:
    """Return True when the answer is a grounded refusal."""
    normalized = answer.lower().strip()
    return any(phrase in normalized for phrase in REFUSAL_PHRASES)


def _parse_relevance_grade(text: str) -> RelevanceGrade:
    """Parse LLM grade output into a RelevanceGrade enum value."""
    token = text.strip().lower().split()[0] if text.strip() else ""
    if token.startswith("relevant"):
        return RelevanceGrade.RELEVANT
    if token.startswith("partial"):
        return RelevanceGrade.PARTIAL
    if token.startswith("irrelevant"):
        return RelevanceGrade.IRRELEVANT
    return RelevanceGrade.PARTIAL


def _extract_cited_listing_ids(answer: str, available_ids: list[str]) -> list[str]:
    """Extract listing IDs referenced in the generated answer."""
    cited: list[str] = []
    for listing_id in available_ids:
        if listing_id in answer:
            cited.append(listing_id)
    return cited


class RAGAgent:
    """Self-correcting RAG agent built on LangGraph."""

    def __init__(
        self,
        settings: Settings | None = None,
        retriever: VectorRetriever | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._owns_retriever = retriever is None
        self.retriever = retriever or VectorRetriever(self.settings)
        self._llm = self._create_llm()
        self._graph = self._build_graph()

    def _create_llm(self) -> ChatOpenAI:
        """Create the LLM using Groq (free) or OpenAI fallback."""
        if self.settings.chat_provider == "groq":
            logger.info("Using Groq chat provider with model: %s", self.settings.groq_chat_model)
            return ChatOpenAI(
                model=self.settings.groq_chat_model,
                temperature=self.settings.agent_temperature,
                api_key=self.settings.groq_api_key_str or "dummy",
                base_url=self.settings.groq_base_url,
            )
        else:
            logger.info("Using OpenAI chat provider with model: %s", self.settings.openai_chat_model)
            return ChatOpenAI(
                model=self.settings.openai_chat_model,
                temperature=self.settings.agent_temperature,
                api_key=self.settings.openai_api_key_str or "dummy",
            )

    def close(self) -> None:
        """Close owned resources."""
        if self._owns_retriever:
            self.retriever.close()

    def _build_graph(self) -> Any:
        """Compile the LangGraph state machine."""
        graph = StateGraph(AgentState)

        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("grade_relevance", self._grade_relevance_node)
        graph.add_node("rewrite_query", self._rewrite_query_node)
        graph.add_node("generate", self._generate_node)
        graph.add_node("cite", self._cite_node)

        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "grade_relevance")
        graph.add_conditional_edges(
            "grade_relevance",
            self._route_after_grade,
            {
                "generate": "generate",
                "rewrite": "rewrite_query",
            },
        )
        graph.add_edge("rewrite_query", "retrieve")
        graph.add_edge("generate", "cite")
        graph.add_edge("cite", END)

        return graph.compile()

    @traceable(name="retrieve_node", run_type="chain")
    def _retrieve_node(self, state: AgentState) -> dict[str, Any]:
        """Retrieve top-k chunks for the current search query."""
        query = state.get("search_query") or state["question"]
        attempts = state.get("retrieval_attempts", 0) + 1

        logger.info("Retrieval attempt %d for query: %s", attempts, query)
        response = self.retriever.search(
            query,
            k=self.settings.default_top_k,
            score_threshold=0.0,
        )

        return {
            "search_query": query,
            "retrieval_attempts": attempts,
            "documents": response.documents,
            "context": response.context,
            "citations": response.citations,
            "listing_ids": response.listing_ids,
        }

    @traceable(name="grade_relevance_node", run_type="chain")
    def _grade_relevance_node(self, state: AgentState) -> dict[str, Any]:
        """LLM grades whether retrieved context is relevant to the question."""
        if not state.get("documents"):
            return {"relevance_grade": RelevanceGrade.IRRELEVANT.value}

        prompt = GRADE_RELEVANCE_PROMPT.format(
            question=state["question"],
            context=state.get("context") or "",
        )
        result = self._llm.invoke([HumanMessage(content=prompt)])
        content = result.content if isinstance(result.content, str) else str(result.content)
        grade = _parse_relevance_grade(content)

        logger.info("Relevance grade: %s", grade.value)
        return {"relevance_grade": grade.value}

    def _route_after_grade(self, state: AgentState) -> Literal["generate", "rewrite"]:
        """Decide whether to answer, or rewrite and re-retrieve."""
        grade = state.get("relevance_grade", RelevanceGrade.IRRELEVANT.value)
        attempts = state.get("retrieval_attempts", 1)
        max_retries = self.settings.max_retrieval_retries

        if grade in (RelevanceGrade.RELEVANT.value, RelevanceGrade.PARTIAL.value):
            return "generate"

        if attempts <= max_retries:
            return "rewrite"

        logger.info(
            "Max retrieval retries reached (%d); generating with best available context",
            max_retries,
        )
        return "generate"

    @traceable(name="rewrite_query_node", run_type="chain")
    def _rewrite_query_node(self, state: AgentState) -> dict[str, Any]:
        """Rewrite the search query to improve retrieval on the next attempt."""
        prompt = REWRITE_QUERY_PROMPT.format(
            question=state["question"],
            search_query=state.get("search_query") or state["question"],
        )
        result = self._llm.invoke([HumanMessage(content=prompt)])
        rewritten = result.content if isinstance(result.content, str) else str(result.content)
        rewritten = rewritten.strip().strip('"').strip("'")

        history = list(state.get("rewritten_queries") or [])
        history.append(rewritten)

        logger.info("Rewritten query: %s", rewritten)
        return {
            "search_query": rewritten,
            "rewritten_queries": history,
        }

    @traceable(name="generate_node", run_type="chain")
    def _generate_node(self, state: AgentState) -> dict[str, Any]:
        """Generate a grounded answer from retrieved context."""
        context = state.get("context") or ""
        if not context.strip():
            return {
                "answer": "I don't know based on the available listings.",
                "refused": True,
            }

        prompt = GENERATE_ANSWER_PROMPT.format(
            question=state["question"],
            context=context,
        )
        result = self._llm.invoke([HumanMessage(content=prompt)])
        answer = result.content if isinstance(result.content, str) else str(result.content)
        answer = answer.strip()

        return {
            "answer": answer,
            "refused": _is_refusal(answer),
        }

    @traceable(name="cite_node", run_type="chain")
    def _cite_node(self, state: AgentState) -> dict[str, Any]:
        """Attach citations for listings referenced in the answer."""
        if state.get("refused"):
            return {"cited_listing_ids": [], "citations": []}

        available_ids = state.get("listing_ids") or []
        answer = state.get("answer") or ""
        cited_ids = _extract_cited_listing_ids(answer, available_ids)

        if not cited_ids:
            cited_ids = self._infer_citations_from_answer(answer, state.get("citations") or [])

        filtered = [
            citation
            for citation in (state.get("citations") or [])
            if citation.listing_id in cited_ids
        ]

        if not filtered and not state.get("refused"):
            filtered = list(state.get("citations") or [])[:3]
            cited_ids = [c.listing_id for c in filtered]

        return {
            "cited_listing_ids": cited_ids,
            "citations": filtered,
        }

    @staticmethod
    def _infer_citations_from_answer(
        answer: str,
        citations: list[Citation],
    ) -> list[str]:
        """Match listing names mentioned in the answer to citation IDs."""
        answer_lower = answer.lower()
        cited: list[str] = []
        for citation in citations:
            name = citation.listing_name.lower()
            if name and name in answer_lower:
                cited.append(citation.listing_id)
        return cited

    @traceable(name="rag_agent_run", run_type="chain")
    def run(self, question: str) -> AgentResponse:
        """
        Run the full self-correcting RAG pipeline for a question.

        Args:
            question: User natural-language question.

        Returns:
            AgentResponse with answer, citations, and metadata.
        """
        initial_state: AgentState = {
            "question": question,
            "search_query": question,
            "retrieval_attempts": 0,
            "rewritten_queries": [],
        }

        final_state = self._graph.invoke(initial_state)
        return self._to_response(final_state)

    def stream(self, question: str) -> Iterator[dict[str, Any]]:
        """
        Stream graph updates for a question.

        Yields:
            Dict updates per node completion.
        """
        initial_state: AgentState = {
            "question": question,
            "search_query": question,
            "retrieval_attempts": 0,
            "rewritten_queries": [],
        }

        for update in self._graph.stream(initial_state, stream_mode="updates"):
            yield update

    async def astream_tokens(self, question: str) -> AsyncIterator[str]:
        """
        Stream LLM tokens from the generate step.

        Note: Only the final generation is streamed; retrieval/grading run first.
        """
        state: AgentState = {
            "question": question,
            "search_query": question,
            "retrieval_attempts": 0,
            "rewritten_queries": [],
        }

        state.update(self._retrieve_node(state))
        state.update(self._grade_relevance_node(state))

        while self._route_after_grade(state) == "rewrite":
            state.update(self._rewrite_query_node(state))
            state.update(self._retrieve_node(state))
            state.update(self._grade_relevance_node(state))

        context = state.get("context") or ""
        if not context.strip():
            yield "I don't know based on the available listings."
            return

        prompt = GENERATE_ANSWER_PROMPT.format(
            question=question,
            context=context,
        )

        async for chunk in self._llm.astream([HumanMessage(content=prompt)]):
            if chunk.content:
                yield str(chunk.content)

    @staticmethod
    def _to_response(state: AgentState) -> AgentResponse:
        """Convert final graph state to a structured AgentResponse."""
        return AgentResponse(
            question=state["question"],
            answer=state.get("answer") or "",
            citations=list(state.get("citations") or []),
            cited_listing_ids=list(state.get("cited_listing_ids") or []),
            retrieval_attempts=state.get("retrieval_attempts") or 1,
            rewritten_queries=list(state.get("rewritten_queries") or []),
            refused=bool(state.get("refused")),
            documents=list(state.get("documents") or []),
        )


def get_rag_agent(settings: Settings | None = None) -> RAGAgent:
    """Factory for a configured RAGAgent instance."""
    return RAGAgent(settings)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Capstone RAG — self-correcting LangGraph agent",
    )
    parser.add_argument("question", nargs="?", help="Question to ask the agent")
    parser.add_argument("--stream", action="store_true", help="Stream graph node updates")
    parser.add_argument("--json", action="store_true", help="Output JSON response")
    return parser


def main() -> None:
    """CLI entry point for agent testing."""
    args = _build_parser().parse_args()
    settings = get_settings()
    setup_logging(settings.log_level)

    if not args.question:
        _build_parser().print_help()
        raise SystemExit(1)

    agent = RAGAgent(settings)

    try:
        if args.stream:
            print(f'Question: "{args.question}"\n')
            for update in agent.stream(args.question):
                for node_name, payload in update.items():
                    print(f"--- {node_name} ---")
                    if "answer" in payload:
                        print(payload["answer"])
                    elif "relevance_grade" in payload:
                        print(f"grade: {payload['relevance_grade']}")
                    elif "search_query" in payload and node_name == "rewrite_query":
                        print(f"rewritten: {payload['search_query']}")
                    else:
                        docs = payload.get("documents")
                        if docs is not None:
                            print(f"retrieved: {len(docs)} chunks")
            return

        with log_duration(logger, "agent run", extra={"question": args.question}):
            response = agent.run(args.question)

        if args.json:
            print(
                json.dumps(
                    {
                        "question": response.question,
                        "answer": response.answer,
                        "refused": response.refused,
                        "retrieval_attempts": response.retrieval_attempts,
                        "rewritten_queries": response.rewritten_queries,
                        "cited_listing_ids": response.cited_listing_ids,
                        "citations": [c.model_dump() for c in response.citations],
                    },
                    indent=2,
                    default=str,
                )
            )
            return

        print(f'Question: "{response.question}"\n')
        print(response.answer)
        print(f"\nRetrieval attempts: {response.retrieval_attempts}")
        if response.rewritten_queries:
            print(f"Rewritten queries: {response.rewritten_queries}")

        if response.citations:
            print("\n--- Citations ---")
            for citation in response.citations:
                loc = ", ".join(
                    filter(None, [citation.location.city, citation.location.country])
                )
                print(
                    f"- {citation.listing_name} ({citation.listing_id}) "
                    f"[{citation.score:.4f}] {loc}"
                )

    except Exception as exc:
        print(f"\nERROR: {exc}\n")
        raise SystemExit(1) from exc
    finally:
        agent.close()


if __name__ == "__main__":
    main()