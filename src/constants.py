"""Application-wide constants for the Capstone RAG system."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# MongoDB collections & index
# ---------------------------------------------------------------------------
DEFAULT_DB_NAME: str = "sample_airbnb"
SOURCE_COLLECTION: str = "listingsAndReviews"
VECTOR_COLLECTION: str = "rag_chunks"
VECTOR_INDEX_NAME: str = "rag_vector_index"

# ---------------------------------------------------------------------------
# Text fields extracted from Airbnb listings
# ---------------------------------------------------------------------------
LISTING_TEXT_FIELDS: tuple[str, ...] = (
    "description",
    "summary",
    "space",
    "neighborhood_overview",
)

REVIEW_COMMENTS_FIELD: str = "reviews.comments"

# ---------------------------------------------------------------------------
# Model providers
# ---------------------------------------------------------------------------
CHAT_PROVIDER_GROQ: str = "groq"
CHAT_PROVIDER_OPENAI: str = "openai"
EMBEDDING_PROVIDER_HUGGINGFACE: str = "huggingface"
EMBEDDING_PROVIDER_OPENAI: str = "openai"

# Groq models
DEFAULT_GROQ_MODEL: str = "llama-3.3-70b-versatile"
DEFAULT_GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"

# HuggingFace embedding model
DEFAULT_HF_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
HF_EMBEDDING_DIMENSIONS: int = 384

# OpenAI models (fallback)
DEFAULT_EMBEDDING_MODEL: str = "text-embedding-3-small"
DEFAULT_CHAT_MODEL: str = "gpt-4.1-mini"
EMBEDDING_DIMENSIONS: int = 1536

# ---------------------------------------------------------------------------
# Chunking defaults
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE: int = 800
DEFAULT_CHUNK_OVERLAP: int = 100
DEFAULT_INGEST_LIMIT: int = 200
DEFAULT_BATCH_SIZE: int = 50

# ---------------------------------------------------------------------------
# Retrieval defaults
# ---------------------------------------------------------------------------
DEFAULT_TOP_K: int = 5
DEFAULT_NUM_CANDIDATES: int = 50
DEFAULT_RELEVANCE_THRESHOLD: float = 0.7

# ---------------------------------------------------------------------------
# Agent defaults
# ---------------------------------------------------------------------------
MAX_RETRIEVAL_RETRIES: int = 2
DEFAULT_AGENT_TEMPERATURE: float = 0.0
REFUSAL_PHRASES: tuple[str, ...] = (
    "i don't know",
    "i do not know",
    "insufficient context",
    "not enough information",
    "cannot answer",
    "unable to answer",
    "no relevant",
)

# ---------------------------------------------------------------------------
# Evaluation thresholds (regression gate)
# ---------------------------------------------------------------------------
EVAL_RECALL_AT_K: int = 5
EVAL_RECALL_THRESHOLD: float = 0.70
EVAL_GROUNDEDNESS_THRESHOLD: float = 0.90
EVAL_REFUSAL_THRESHOLD: float = 1.00
EVAL_CITATION_THRESHOLD: float = 0.90

# ---------------------------------------------------------------------------
# LangSmith
# ---------------------------------------------------------------------------
DEFAULT_LANGSMITH_PROJECT: str = "capstone-rag"

# ---------------------------------------------------------------------------
# Application metadata
# ---------------------------------------------------------------------------
APP_NAME: str = "Capstone RAG"
APP_VERSION: str = "1.0.0"
APP_DESCRIPTION: str = (
    "Self-correcting retrieval-augmented generation agent over "
    "MongoDB Atlas Vector Search"
)