"""
MongoDB Atlas Vector Search retrieval layer.

Supports both HuggingFace (free) and OpenAI embeddings.

Usage:
    python -m src.retrieval "quiet apartment near the beach with wifi"
    python -m src.retrieval "beach wifi" --k 10 --json
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import StructuredTool
from langsmith import traceable
from pydantic import BaseModel, Field
from pymongo.collection import Collection

from src.config import Settings, get_settings
from src.models import Citation, RetrievedDocument, RetrievalResponse, VectorSearchResult
from src.utils import get_logger, get_mongo_client, log_duration, setup_logging

logger = get_logger("retrieval")


def format_retrieval_context(documents: list[RetrievedDocument]) -> str:
    """Format retrieved documents into a numbered context block for the LLM."""
    if not documents:
        return ""

    blocks: list[str] = []
    for index, doc in enumerate(documents, start=1):
        location_parts = [
            part
            for part in (
                doc.location.neighbourhood,
                doc.location.city,
                doc.location.country,
            )
            if part
        ]
        location_str = ", ".join(location_parts) if location_parts else "Unknown"

        blocks.append(
            "\n".join(
                [
                    f"[{index}] {doc.listing_name} (listing_id={doc.listing_id}, score={doc.score:.4f})",
                    f"Source: {doc.source_type.value} / {doc.field_name}",
                    f"Location: {location_str}",
                    doc.text.strip(),
                ]
            )
        )

    return "\n\n---\n\n".join(blocks)


def build_citations(documents: list[RetrievedDocument]) -> list[Citation]:
    """Build deduplicated citations ordered by best chunk score per listing."""
    best_by_listing: dict[str, Citation] = {}

    for doc in documents:
        existing = best_by_listing.get(doc.listing_id)
        if existing is None or doc.score > existing.score:
            best_by_listing[doc.listing_id] = Citation(
                listing_id=doc.listing_id,
                listing_name=doc.listing_name,
                chunk_id=doc.chunk_id,
                score=doc.score,
                location=doc.location,
            )

    return sorted(best_by_listing.values(), key=lambda item: item.score, reverse=True)


class HuggingFaceEmbedder:
    """Local HuggingFace embedding model (free, no API calls)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _lazy_load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading HuggingFace model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_query(self, query: str) -> list[float]:
        model = self._lazy_load()
        return model.encode(query).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._lazy_load()
        return model.encode(texts).tolist()


class VectorRetriever:
    """Embed queries and search rag_chunks via MongoDB Atlas Vector Search."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = get_mongo_client(self.settings.mongodb_uri_str)
        self._vectors: Collection = self._client[self.settings.mongodb_db_name][
            self.settings.mongodb_vector_collection
        ]
        self._embedder = self._create_embedder()

    def _create_embedder(self) -> HuggingFaceEmbedder:
        """Create the appropriate embedder based on settings."""
        return HuggingFaceEmbedder(self.settings.hf_embedding_model)

    def close(self) -> None:
        """Close the MongoDB client connection."""
        self._client.close()

    def embed_query(self, query: str) -> list[float]:
        """Generate an embedding vector for a search query."""
        return self._embedder.embed_query(query)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        return self._embedder.embed_documents(texts)

    @traceable(name="vector_search", run_type="retriever")
    def search(
        self,
        query: str,
        k: int | None = None,
        *,
        score_threshold: float | None = None,
        listing_id: str | None = None,
    ) -> RetrievalResponse:
        """
        Run vector search and return structured documents, context, and citations.

        Args:
            query: Natural-language search query.
            k: Number of chunks to retrieve.
            score_threshold: Minimum vector search score (defaults to settings).
            listing_id: Optional filter to a single listing.

        Returns:
            RetrievalResponse with documents, formatted context, and citations.
        """
        k = k or self.settings.default_top_k
        threshold = (
            score_threshold
            if score_threshold is not None
            else self.settings.relevance_score_threshold
        )

        query_vector = self.embed_query(query)
        pipeline = self._build_search_pipeline(
            query_vector=query_vector,
            k=k,
            listing_id=listing_id,
        )

        raw_results: list[VectorSearchResult] = []
        for doc in self._vectors.aggregate(pipeline):
            raw_results.append(VectorSearchResult.from_mongo_document(doc))

        documents = [
            result.to_retrieved_document()
            for result in raw_results
            if result.score >= threshold
        ]

        citations = build_citations(documents)
        listing_ids = [citation.listing_id for citation in citations]

        return RetrievalResponse(
            query=query,
            documents=documents,
            context=format_retrieval_context(documents),
            citations=citations,
            listing_ids=listing_ids,
        )

    def _build_search_pipeline(
        self,
        *,
        query_vector: list[float],
        k: int,
        listing_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build the MongoDB aggregation pipeline for $vectorSearch."""
        vector_search: dict[str, Any] = {
            "index": self.settings.mongodb_vector_index_name,
            "path": "embedding",
            "queryVector": query_vector,
            "numCandidates": self.settings.vector_search_num_candidates,
            "limit": k,
        }

        if listing_id:
            vector_search["filter"] = {"listing_id": listing_id}

        return [
            {"$vectorSearch": vector_search},
            {
                "$project": {
                    "_id": 0,
                    "chunk_id": 1,
                    "listing_id": 1,
                    "listing_name": 1,
                    "text": 1,
                    "source_type": 1,
                    "field_name": 1,
                    "location": 1,
                    "review_score": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

    def as_langchain_retriever(self, k: int | None = None) -> BaseRetriever:
        """Return a LangChain retriever backed by this vector search service."""
        top_k = k or self.settings.default_top_k
        retriever_service = self

        class MongoAtlasRetriever(BaseRetriever):
            """LangChain retriever wrapper for MongoDB Atlas Vector Search."""

            def _get_relevant_documents(
                self,
                query: str,
                *,
                run_manager: Any = None,
            ) -> list[Document]:
                response = retriever_service.search(query, k=top_k)
                return [doc.to_langchain_document() for doc in response.documents]

        return MongoAtlasRetriever()

    def as_langchain_tool(self, k: int | None = None) -> StructuredTool:
        """Return a LangChain tool the agent graph can invoke for retrieval."""

        class SearchInput(BaseModel):
            query: str = Field(description="Natural-language search query for Airbnb listings")
            k: int | None = Field(
                default=None,
                description="Number of chunks to retrieve (optional)",
            )

        def search_listings(query: str, k: int | None = None) -> str:
            response = self.search(query, k=k or self.settings.default_top_k)
            if not response.has_results:
                return "No relevant listings found."
            return response.context

        return StructuredTool.from_function(
            func=search_listings,
            name="search_airbnb_listings",
            description=(
                "Search Airbnb listing descriptions and guest reviews using "
                "semantic vector search. Returns relevant listing context."
            ),
            args_schema=SearchInput,
        )


def get_vector_retriever(settings: Settings | None = None) -> VectorRetriever:
    """Factory for a configured VectorRetriever instance."""
    return VectorRetriever(settings)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Capstone RAG — MongoDB vector retrieval",
    )
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--k", type=int, default=None, help="Top-k chunks to retrieve")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Minimum vector search score",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON")
    return parser


def main() -> None:
    """CLI entry point for retrieval testing."""
    args = _build_parser().parse_args()
    settings = get_settings()
    setup_logging(settings.log_level)

    if not args.query:
        _build_parser().print_help()
        raise SystemExit(1)

    retriever = VectorRetriever(settings)

    try:
        with log_duration(logger, "vector retrieval", extra={"query": args.query}):
            response = retriever.search(
                args.query,
                k=args.k,
                score_threshold=args.threshold,
            )

        if args.json:
            print(
                json.dumps(
                    {
                        "query": response.query,
                        "listing_ids": response.listing_ids,
                        "citations": [c.model_dump() for c in response.citations],
                        "documents": [d.model_dump() for d in response.documents],
                    },
                    indent=2,
                    default=str,
                )
            )
            return

        print(f'Query: "{response.query}"')
        print(f"Results: {len(response.documents)} chunks, {len(response.citations)} listings\n")

        if not response.has_results:
            print("No results above threshold.")
            return

        for index, doc in enumerate(response.documents, 1):
            print(f"{index}. [{doc.score:.4f}] {doc.listing_name} ({doc.listing_id})")
            preview = doc.text[:180].replace("\n", " ")
            print(f"   {preview}...\n")

        print("--- Citations ---")
        for citation in response.citations:
            loc = ", ".join(
                filter(None, [citation.location.city, citation.location.country])
            )
            print(
                f"- {citation.listing_name} ({citation.listing_id}) "
                f"[{citation.score:.4f}] {loc}"
            )

    finally:
        retriever.close()


if __name__ == "__main__":
    main()