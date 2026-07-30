"""
MongoDB ingestion pipeline: extract -> chunk -> embed -> write to rag_chunks.

Uses HuggingFace (free, local) for embeddings.

Usage:
    python -m src.ingest                     # ingest with default limit
    python -m src.ingest --limit 200         # ingest N listings
    python -m src.ingest --create-index      # create vector search index
    python -m src.ingest --search "query"    # sample vector search
    python -m src.ingest --force             # drop chunks and re-ingest
"""

from __future__ import annotations

import argparse
import hashlib
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Iterator

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import ReplaceOne
from pymongo.collection import Collection
from pymongo.errors import BulkWriteError

from src.config import Settings, get_settings
from src.constants import LISTING_TEXT_FIELDS
from src.models import (
    ChunkSourceType,
    IngestStats,
    ListingLocation,
    RagChunk,
    TextSegment,
    VectorSearchResult,
)
from src.retrieval import HuggingFaceEmbedder
from src.utils import get_logger, get_mongo_client, log_duration, setup_logging

logger = get_logger("ingest")

MIN_REVIEW_COMMENT_LENGTH = 40
MIN_SEGMENT_LENGTH = 20


class IngestionPipeline:
    """End-to-end pipeline for building the rag_chunks vector collection."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = get_mongo_client(self.settings.mongodb_uri_str)
        self._db = self._client[self.settings.mongodb_db_name]
        self._source: Collection = self._db[self.settings.mongodb_source_collection]
        self._vectors: Collection = self._db[self.settings.mongodb_vector_collection]
        self._embedder = HuggingFaceEmbedder(self.settings.hf_embedding_model)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def close(self) -> None:
        """Close the MongoDB client connection."""
        self._client.close()

    def _listing_has_text(self, listing: dict[str, Any]) -> bool:
        """Return True if the listing has at least one non-empty text field."""
        for field in LISTING_TEXT_FIELDS:
            value = listing.get(field)
            if isinstance(value, str) and len(value.strip()) >= MIN_SEGMENT_LENGTH:
                return True

        reviews = listing.get("reviews") or []
        for review in reviews:
            comment = (review or {}).get("comments", "")
            if isinstance(comment, str) and len(comment.strip()) >= MIN_REVIEW_COMMENT_LENGTH:
                return True

        return False

    def _extract_review_score(self, listing: dict[str, Any]) -> float | None:
        """Extract the overall review score from listing metadata."""
        scores = listing.get("review_scores") or {}
        rating = scores.get("review_scores_rating")
        if rating is None:
            return None
        try:
            return float(rating)
        except (TypeError, ValueError):
            return None

    def extract_text_segments(self, listing: dict[str, Any]) -> list[TextSegment]:
        """
        Extract embeddable text segments from a raw listing document.

        Strategy:
          - Listing fields (description, summary, space, neighborhood_overview)
            are concatenated into a single segment per listing for coherent context.
          - Each guest review comment becomes its own segment for granular retrieval.
        """
        listing_id = str(listing["_id"])
        listing_name = listing.get("name") or f"Listing {listing_id}"
        location = ListingLocation.from_address(listing.get("address"))
        review_score = self._extract_review_score(listing)
        segments: list[TextSegment] = []

        listing_parts: list[str] = []
        for field in LISTING_TEXT_FIELDS:
            value = listing.get(field)
            if isinstance(value, str) and len(value.strip()) >= MIN_SEGMENT_LENGTH:
                listing_parts.append(f"[{field}] {value.strip()}")

        if listing_parts:
            segments.append(
                TextSegment(
                    listing_id=listing_id,
                    listing_name=listing_name,
                    source_type=ChunkSourceType.LISTING,
                    field_name="listing_fields",
                    text="\n\n".join(listing_parts),
                    location=location,
                    review_score=review_score,
                )
            )

        for review in listing.get("reviews") or []:
            if not review:
                continue
            comment = review.get("comments", "")
            if not isinstance(comment, str):
                continue
            comment = comment.strip()
            if len(comment) < MIN_REVIEW_COMMENT_LENGTH:
                continue

            date_value = review.get("date")
            review_date = (
                date_value.isoformat()
                if hasattr(date_value, "isoformat")
                else str(date_value) if date_value else None
            )

            segments.append(
                TextSegment(
                    listing_id=listing_id,
                    listing_name=listing_name,
                    source_type=ChunkSourceType.REVIEW,
                    field_name="comments",
                    text=comment,
                    location=location,
                    review_score=review_score,
                    review_date=review_date,
                    reviewer_name=review.get("reviewer_name"),
                )
            )

        return segments

    def chunk_segments(self, segments: list[TextSegment]) -> list[RagChunk]:
        """Split text segments into sized chunks with stable identifiers."""
        chunks: list[RagChunk] = []
        now = datetime.now(timezone.utc)

        for segment in segments:
            if segment.source_type == ChunkSourceType.REVIEW:
                texts = [segment.text]
            else:
                texts = self._splitter.split_text(segment.text)

            for chunk_index, text in enumerate(texts):
                if len(text.strip()) < MIN_SEGMENT_LENGTH:
                    continue

                chunk_id = self._build_chunk_id(
                    segment.listing_id,
                    segment.source_type,
                    segment.field_name,
                    chunk_index,
                    text,
                )

                chunks.append(
                    RagChunk(
                        chunk_id=chunk_id,
                        listing_id=segment.listing_id,
                        listing_name=segment.listing_name,
                        text=text,
                        source_type=segment.source_type,
                        field_name=segment.field_name,
                        chunk_index=chunk_index,
                        location=segment.location,
                        review_score=segment.review_score,
                        review_date=segment.review_date,
                        reviewer_name=segment.reviewer_name,
                        created_at=now,
                        updated_at=now,
                    )
                )

        return chunks

    @staticmethod
    def _build_chunk_id(
        listing_id: str,
        source_type: ChunkSourceType,
        field_name: str,
        chunk_index: int,
        text: str,
    ) -> str:
        """Build a deterministic, idempotent chunk identifier."""
        fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        return f"{listing_id}:{source_type.value}:{field_name}:{chunk_index}:{fingerprint}"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts via HuggingFace."""
        if not texts:
            return []
        return self._embedder.embed_documents(texts)

    def write_chunks(
        self,
        chunks: list[RagChunk],
        ingest_run_id: str,
    ) -> int:
        """
        Upsert chunks into rag_chunks by chunk_id (idempotent).

        Returns:
            Number of chunks successfully written.
        """
        if not chunks:
            return 0

        now = datetime.now(timezone.utc)
        operations: list[ReplaceOne] = []

        for chunk in chunks:
            chunk.ingest_run_id = ingest_run_id
            chunk.updated_at = now
            doc = chunk.to_mongo_document()
            operations.append(
                ReplaceOne({"chunk_id": chunk.chunk_id}, doc, upsert=True)
            )

        written = 0
        for i in range(0, len(operations), self.settings.batch_size):
            batch = operations[i : i + self.settings.batch_size]
            try:
                result = self._vectors.bulk_write(batch, ordered=False)
                written += result.upserted_count + result.modified_count + result.matched_count
            except BulkWriteError as exc:
                details = exc.details or {}
                write_errors = details.get("writeErrors", [])
                written += details.get("nInserted", 0) + details.get("nModified", 0)
                logger.error(
                    "Bulk write partial failure: %d errors in batch",
                    len(write_errors),
                )

        return written

    def iter_listings(self, limit: int | None = None) -> Iterator[dict[str, Any]]:
        """
        Yield listings that contain embeddable text content.

        Listing IDs are collected first so the MongoDB cursor is not held open
        during slow embedding API calls (avoids CursorNotFound timeouts).
        """
        query = {
            "$or": [
                {field: {"$exists": True, "$type": "string", "$ne": ""}}
                for field in LISTING_TEXT_FIELDS
            ]
            + [{"reviews.comments": {"$exists": True, "$ne": ""}}]
        }

        id_cursor = self._source.find(query, {"_id": 1})
        if limit:
            id_cursor = id_cursor.limit(limit)
        listing_ids = [doc["_id"] for doc in id_cursor]

        for listing_id in listing_ids:
            listing = self._source.find_one({"_id": listing_id})
            if listing and self._listing_has_text(listing):
                yield listing

    def run(
        self,
        limit: int | None = None,
        *,
        force: bool = False,
    ) -> IngestStats:
        """
        Execute the full ingestion pipeline.

        Args:
            limit: Maximum number of listings to process.
            force: If True, delete existing chunks before ingesting.

        Returns:
            IngestStats summary of the run.
        """
        ingest_run_id = str(uuid.uuid4())
        effective_limit = limit or self.settings.ingest_listing_limit
        stats = IngestStats(ingest_run_id=ingest_run_id)
        start = time.perf_counter()

        logger.info(
            "Starting ingestion run %s (limit=%d, force=%s)",
            ingest_run_id,
            effective_limit,
            force,
        )

        if force:
            deleted = self._vectors.delete_many({}).deleted_count
            logger.info("Force mode: deleted %d existing chunks", deleted)

        pending_chunks: list[RagChunk] = []
        pending_texts: list[str] = []

        def flush_batch() -> None:
            nonlocal pending_chunks, pending_texts
            if not pending_texts:
                return

            embeddings = self.embed_texts(pending_texts)
            stats.embeddings_generated += len(embeddings)

            for chunk, embedding in zip(pending_chunks, embeddings):
                chunk.embedding = embedding

            written = self.write_chunks(pending_chunks, ingest_run_id)
            stats.chunks_written += written
            pending_chunks = []
            pending_texts = []

        try:
            for listing in self.iter_listings(limit=effective_limit):
                try:
                    segments = self.extract_text_segments(listing)
                    if not segments:
                        stats.listings_skipped += 1
                        continue

                    chunks = self.chunk_segments(segments)
                    stats.listings_processed += 1
                    stats.segments_extracted += len(segments)
                    stats.chunks_created += len(chunks)

                    for chunk in chunks:
                        pending_chunks.append(chunk)
                        pending_texts.append(chunk.text)

                        if len(pending_texts) >= self.settings.batch_size:
                            flush_batch()

                    if stats.listings_processed % 25 == 0:
                        logger.info(
                            "Progress: %d listings, %d chunks queued",
                            stats.listings_processed,
                            stats.chunks_created,
                        )

                except Exception:
                    stats.errors += 1
                    logger.exception(
                        "Failed to process listing %s",
                        listing.get("_id"),
                    )

            flush_batch()
        except Exception as exc:
            stats.duration_seconds = round(time.perf_counter() - start, 2)
            logger.error("Ingestion error: %s", exc)
            raise
        stats.duration_seconds = round(time.perf_counter() - start, 2)

        logger.info(
            "Ingestion complete: %d listings -> %d chunks in %.1fs",
            stats.listings_processed,
            stats.chunks_written,
            stats.duration_seconds,
        )
        return stats

    def ensure_vector_index(self, *, wait_until_ready: bool = True) -> str:
        """
        Create the Atlas Vector Search index on rag_chunks.embedding.

        Args:
            wait_until_ready: Block until index status is ACTIVE.

        Returns:
            Index name.
        """
        index_name = self.settings.mongodb_vector_index_name
        dimensions = self.settings.effective_embedding_dimensions

        existing_indexes = list(self._vectors.list_search_indexes())
        for idx in existing_indexes:
            if idx.get("name") == index_name:
                status = idx.get("status", "UNKNOWN")
                logger.info("Vector index '%s' already exists (status=%s)", index_name, status)
                if wait_until_ready and status != "READY":
                    self._wait_for_index_ready(index_name)
                return index_name

        definition = {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": dimensions,
                    "similarity": "cosine",
                },
                {"type": "filter", "path": "listing_id"},
                {"type": "filter", "path": "source_type"},
            ]
        }

        logger.info(
            "Creating vector search index '%s' (dims=%d)...",
            index_name,
            dimensions,
        )
        self._vectors.create_search_index(
            {"name": index_name, "definition": definition}
        )

        if wait_until_ready:
            self._wait_for_index_ready(index_name)

        return index_name

    def _wait_for_index_ready(
        self,
        index_name: str,
        timeout_seconds: int = 300,
        poll_interval: float = 5.0,
    ) -> None:
        """Poll until the vector search index reaches READY status."""
        deadline = time.time() + timeout_seconds

        while time.time() < deadline:
            for idx in self._vectors.list_search_indexes():
                if idx.get("name") == index_name:
                    status = idx.get("status", "UNKNOWN")
                    logger.info("Index '%s' status: %s", index_name, status)
                    if status == "READY":
                        return
            time.sleep(poll_interval)

        raise TimeoutError(
            f"Vector index '{index_name}' did not become READY within {timeout_seconds}s"
        )

    def sample_vector_search(
        self,
        query: str,
        k: int | None = None,
    ) -> list[VectorSearchResult]:
        """
        Run a sample $vectorSearch query to validate the pipeline.

        Args:
            query: Natural-language search query.
            k: Number of results (defaults to settings.default_top_k).

        Returns:
            Ranked list of VectorSearchResult objects.
        """
        k = k or self.settings.default_top_k
        query_embedding = self.embed_texts([query])[0]

        pipeline: list[dict[str, Any]] = [
            {
                "$vectorSearch": {
                    "index": self.settings.mongodb_vector_index_name,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": self.settings.vector_search_num_candidates,
                    "limit": k,
                }
            },
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

        results: list[VectorSearchResult] = []
        for doc in self._vectors.aggregate(pipeline):
            results.append(VectorSearchResult.from_mongo_document(doc))

        return results

    def get_collection_stats(self) -> dict[str, Any]:
        """Return basic statistics about source and vector collections."""
        return {
            "source_collection": self.settings.mongodb_source_collection,
            "source_count": self._source.estimated_document_count(),
            "vector_collection": self.settings.mongodb_vector_collection,
            "vector_count": self._vectors.estimated_document_count(),
        }


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Capstone RAG — MongoDB ingestion pipeline",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max listings to ingest (default: INGEST_LISTING_LIMIT from .env)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing chunks before ingesting",
    )
    parser.add_argument(
        "--create-index",
        action="store_true",
        help="Create the Atlas Vector Search index",
    )
    parser.add_argument(
        "--search",
        type=str,
        default=None,
        metavar="QUERY",
        help="Run a sample vector search query",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print collection statistics and exit",
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Run ingestion without creating index",
    )
    return parser


def main() -> None:
    """CLI entry point for the ingestion pipeline."""
    args = _build_parser().parse_args()
    settings = get_settings()
    setup_logging(settings.log_level)

    pipeline = IngestionPipeline(settings)

    try:
        if args.stats:
            stats = pipeline.get_collection_stats()
            print(f"Source: {stats['source_collection']} ({stats['source_count']:,} docs)")
            print(f"Vectors: {stats['vector_collection']} ({stats['vector_count']:,} chunks)")
            return

        should_ingest = not args.search and not args.create_index
        if args.ingest_only:
            should_ingest = True

        if should_ingest or (not args.search and not args.stats):
            with log_duration(logger, "ingestion pipeline"):
                result = pipeline.run(limit=args.limit, force=args.force)

            print("\n=== Ingestion Summary ===")
            print(f"  Run ID:              {result.ingest_run_id}")
            print(f"  Listings processed:  {result.listings_processed}")
            print(f"  Listings skipped:    {result.listings_skipped}")
            print(f"  Segments extracted:  {result.segments_extracted}")
            print(f"  Chunks created:      {result.chunks_created}")
            print(f"  Chunks written:      {result.chunks_written}")
            print(f"  Embeddings generated:{result.embeddings_generated}")
            print(f"  Errors:              {result.errors}")
            print(f"  Duration:            {result.duration_seconds}s")

        if args.create_index or (should_ingest and not args.ingest_only):
            with log_duration(logger, "vector index creation"):
                index_name = pipeline.ensure_vector_index()
            print(f"\nVector index ready: {index_name}")

        if args.search:
            query = args.search
            print(f'\n=== Vector Search: "{query}" ===\n')
            results = pipeline.sample_vector_search(query)
            if not results:
                print("No results returned. Ensure ingestion and index are complete.")
            for i, hit in enumerate(results, 1):
                location = hit.location
                loc_str = ", ".join(
                    filter(None, [location.city, location.country])
                )
                print(f"{i}. [{hit.score:.4f}] {hit.listing_name}")
                print(f"   ID: {hit.listing_id} | Type: {hit.source_type.value} | {loc_str}")
                preview = hit.text[:200].replace("\n", " ")
                print(f"   {preview}...\n")

    finally:
        pipeline.close()


if __name__ == "__main__":
    main()