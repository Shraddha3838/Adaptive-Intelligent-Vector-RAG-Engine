"""Pydantic data models for ingestion, retrieval, and agent state."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ChunkSourceType(str, Enum):
    """Origin of a text chunk within a listing document."""

    LISTING = "listing"
    REVIEW = "review"


class ListingLocation(BaseModel):
    """Geographic metadata extracted from an Airbnb listing."""

    country: str | None = None
    country_code: str | None = None
    city: str | None = None
    state: str | None = None
    neighbourhood: str | None = None
    market: str | None = None

    @classmethod
    def from_address(cls, address: dict[str, Any] | None) -> ListingLocation:
        """Build location metadata from a listing address sub-document."""
        if not address:
            return cls()
        return cls(
            country=address.get("country"),
            country_code=address.get("country_code"),
            city=address.get("city"),
            state=address.get("state"),
            neighbourhood=address.get("neighbourhood"),
            market=address.get("market"),
        )


class TextSegment(BaseModel):
    """A raw text segment prior to chunking and embedding."""

    listing_id: str
    listing_name: str
    source_type: ChunkSourceType
    field_name: str
    text: str
    location: ListingLocation = Field(default_factory=ListingLocation)
    review_score: float | None = None
    review_date: str | None = None
    reviewer_name: str | None = None

    @field_validator("text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        """Normalize whitespace in segment text."""
        return " ".join(value.split())


class RagChunk(BaseModel):
    """A chunked, embeddable unit stored in the vector collection."""

    chunk_id: str
    listing_id: str
    listing_name: str
    text: str
    source_type: ChunkSourceType
    field_name: str
    chunk_index: int
    location: ListingLocation = Field(default_factory=ListingLocation)
    review_score: float | None = None
    review_date: str | None = None
    reviewer_name: str | None = None
    embedding: list[float] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ingest_run_id: str | None = None

    def to_mongo_document(self) -> dict[str, Any]:
        """Serialize to a MongoDB-ready document."""
        doc = self.model_dump(mode="python")
        doc["source_type"] = self.source_type.value
        return doc

    @classmethod
    def from_mongo_document(cls, doc: dict[str, Any]) -> RagChunk:
        """Deserialize from a MongoDB document."""
        data = dict(doc)
        data.pop("_id", None)
        if isinstance(data.get("source_type"), str):
            data["source_type"] = ChunkSourceType(data["source_type"])
        return cls.model_validate(data)


class IngestStats(BaseModel):
    """Summary statistics from an ingestion run."""

    listings_processed: int = 0
    listings_skipped: int = 0
    segments_extracted: int = 0
    chunks_created: int = 0
    chunks_written: int = 0
    embeddings_generated: int = 0
    errors: int = 0
    duration_seconds: float = 0.0
    ingest_run_id: str = ""


class VectorSearchResult(BaseModel):
    """A single result from Atlas Vector Search."""

    chunk_id: str
    listing_id: str
    listing_name: str
    text: str
    score: float
    source_type: ChunkSourceType
    field_name: str
    location: ListingLocation = Field(default_factory=ListingLocation)
    review_score: float | None = None

    @classmethod
    def from_mongo_document(cls, doc: dict[str, Any]) -> VectorSearchResult:
        """Build a search result from an aggregation output document."""
        source_type = doc.get("source_type", ChunkSourceType.LISTING.value)
        if isinstance(source_type, str):
            source_type = ChunkSourceType(source_type)

        location_data = doc.get("location") or {}
        location = (
            location_data
            if isinstance(location_data, ListingLocation)
            else ListingLocation.model_validate(location_data)
        )

        return cls(
            chunk_id=doc.get("chunk_id", ""),
            listing_id=str(doc.get("listing_id", "")),
            listing_name=doc.get("listing_name", ""),
            text=doc.get("text", ""),
            score=float(doc.get("score", 0.0)),
            source_type=source_type,
            field_name=doc.get("field_name", ""),
            location=location,
            review_score=doc.get("review_score"),
        )

    def to_retrieved_document(self) -> RetrievedDocument:
        """Convert a vector search hit to a retrieval document."""
        return RetrievedDocument(
            chunk_id=self.chunk_id,
            listing_id=self.listing_id,
            listing_name=self.listing_name,
            text=self.text,
            score=self.score,
            source_type=self.source_type,
            field_name=self.field_name,
            location=self.location,
            review_score=self.review_score,
        )


class RetrievedDocument(BaseModel):
    """A document returned by the retrieval layer for agent consumption."""

    chunk_id: str
    listing_id: str
    listing_name: str
    text: str
    score: float
    source_type: ChunkSourceType
    field_name: str
    location: ListingLocation = Field(default_factory=ListingLocation)
    review_score: float | None = None

    def to_langchain_document(self) -> Any:
        """Convert to a LangChain Document with rich metadata."""
        from langchain_core.documents import Document

        return Document(
            page_content=self.text,
            metadata={
                "chunk_id": self.chunk_id,
                "listing_id": self.listing_id,
                "listing_name": self.listing_name,
                "score": self.score,
                "source_type": self.source_type.value,
                "field_name": self.field_name,
                "city": self.location.city,
                "country": self.location.country,
                "review_score": self.review_score,
            },
        )


class Citation(BaseModel):
    """A source citation attached to an agent answer."""

    listing_id: str
    listing_name: str
    chunk_id: str
    score: float
    location: ListingLocation = Field(default_factory=ListingLocation)


class RetrievalResponse(BaseModel):
    """Structured output from a retrieval query."""

    query: str
    documents: list[RetrievedDocument] = Field(default_factory=list)
    context: str = ""
    citations: list[Citation] = Field(default_factory=list)
    listing_ids: list[str] = Field(default_factory=list)

    @property
    def has_results(self) -> bool:
        """Return True when at least one document was retrieved."""
        return len(self.documents) > 0


class RelevanceGrade(str, Enum):
    """LLM relevance assessment for retrieved context."""

    RELEVANT = "relevant"
    PARTIAL = "partial"
    IRRELEVANT = "irrelevant"


class AgentResponse(BaseModel):
    """Final structured response from the RAG agent."""

    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    cited_listing_ids: list[str] = Field(default_factory=list)
    retrieval_attempts: int = 1
    rewritten_queries: list[str] = Field(default_factory=list)
    refused: bool = False
    documents: list[RetrievedDocument] = Field(default_factory=list)
