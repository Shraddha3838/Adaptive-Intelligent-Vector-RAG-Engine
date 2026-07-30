"""Centralized configuration management for the Capstone RAG system."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.constants import (
    CHAT_PROVIDER_GROQ,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DB_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GROQ_BASE_URL,
    DEFAULT_GROQ_MODEL,
    DEFAULT_HF_EMBEDDING_MODEL,
    DEFAULT_INGEST_LIMIT,
    DEFAULT_LANGSMITH_PROJECT,
    DEFAULT_NUM_CANDIDATES,
    DEFAULT_RELEVANCE_THRESHOLD,
    DEFAULT_TOP_K,
    EMBEDDING_DIMENSIONS,
    EVAL_CITATION_THRESHOLD,
    EVAL_GROUNDEDNESS_THRESHOLD,
    EVAL_RECALL_AT_K,
    EVAL_RECALL_THRESHOLD,
    EVAL_REFUSAL_THRESHOLD,
    HF_EMBEDDING_DIMENSIONS,
    MAX_RETRIEVAL_RETRIES,
    SOURCE_COLLECTION,
    VECTOR_COLLECTION,
    VECTOR_INDEX_NAME,
)


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Model Provider Selection ---
    chat_provider: str = Field(
        default=CHAT_PROVIDER_GROQ,
        description="Chat provider: 'groq' or 'openai'",
    )
    embedding_provider: str = Field(
        default="huggingface",
        description="Embedding provider: 'huggingface' or 'openai'",
    )

    # --- Groq (free chat) ---
    groq_api_key: SecretStr | None = Field(
        default=None,
        description="Groq API key (free chat inference)",
    )
    groq_base_url: str = Field(
        default=DEFAULT_GROQ_BASE_URL,
        description="Groq API base URL",
    )
    groq_chat_model: str = Field(
        default=DEFAULT_GROQ_MODEL,
        description="Groq chat model identifier",
    )

    # --- HuggingFace (free embeddings) ---
    huggingface_api_key: SecretStr | None = Field(
        default=None,
        description="HuggingFace API key (for embedding models)",
    )
    hf_embedding_model: str = Field(
        default=DEFAULT_HF_EMBEDDING_MODEL,
        description="HuggingFace embedding model name",
    )
    hf_embedding_dimensions: int = Field(
        default=HF_EMBEDDING_DIMENSIONS,
        description="HuggingFace embedding dimensions (384 for all-MiniLM-L6-v2)",
    )

    # --- OpenAI (fallback) ---
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI API key (fallback)",
    )
    openai_embedding_model: str = Field(
        default=DEFAULT_EMBEDDING_MODEL,
        description="OpenAI embedding model identifier",
    )
    openai_chat_model: str = Field(
        default=DEFAULT_CHAT_MODEL,
        description="OpenAI chat model identifier",
    )
    embedding_dimensions: int = Field(
        default=EMBEDDING_DIMENSIONS,
        description="Vector embedding dimensions (must match index)",
    )

    # --- MongoDB ---
    mongodb_uri: SecretStr = Field(..., description="MongoDB Atlas connection URI")
    mongodb_db_name: str = Field(
        default=DEFAULT_DB_NAME,
        description="MongoDB database name",
    )
    mongodb_source_collection: str = Field(
        default=SOURCE_COLLECTION,
        description="Source listings collection",
    )
    mongodb_vector_collection: str = Field(
        default=VECTOR_COLLECTION,
        description="Vector chunks collection",
    )
    mongodb_vector_index_name: str = Field(
        default=VECTOR_INDEX_NAME,
        description="Atlas Vector Search index name",
    )

    # --- LangSmith ---
    langsmith_api_key: SecretStr | None = Field(
        default=None,
        description="LangSmith API key (optional for local dev)",
    )
    langsmith_project: str = Field(
        default=DEFAULT_LANGSMITH_PROJECT,
        description="LangSmith project name",
    )
    langsmith_tracing: bool = Field(
        default=False,
        description="Enable LangSmith tracing",
    )

    # --- Ingestion ---
    ingest_listing_limit: int = Field(
        default=DEFAULT_INGEST_LIMIT,
        ge=1,
        description="Max listings to ingest (0 = all)",
    )
    chunk_size: int = Field(
        default=DEFAULT_CHUNK_SIZE,
        ge=100,
        description="Character chunk size for text splitting",
    )
    chunk_overlap: int = Field(
        default=DEFAULT_CHUNK_OVERLAP,
        ge=0,
        description="Character overlap between chunks",
    )
    batch_size: int = Field(
        default=DEFAULT_BATCH_SIZE,
        ge=1,
        description="Batch size for embedding writes",
    )

    # --- Retrieval ---
    default_top_k: int = Field(
        default=DEFAULT_TOP_K,
        ge=1,
        description="Default number of chunks to retrieve",
    )
    vector_search_num_candidates: int = Field(
        default=DEFAULT_NUM_CANDIDATES,
        ge=1,
        description="numCandidates for Atlas Vector Search",
    )
    relevance_score_threshold: float = Field(
        default=DEFAULT_RELEVANCE_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score for retrieved chunks",
    )

    # --- Agent ---
    max_retrieval_retries: int = Field(
        default=MAX_RETRIEVAL_RETRIES,
        ge=0,
        le=5,
        description="Max self-correction retrieval loops",
    )
    agent_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="LLM temperature for answer generation",
    )

    # --- Evaluation ---
    eval_recall_at_k: int = Field(
        default=EVAL_RECALL_AT_K,
        ge=1,
        description="k for recall@k metric",
    )
    eval_recall_threshold: float = Field(
        default=EVAL_RECALL_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum recall@k for regression gate",
    )
    eval_groundedness_threshold: float = Field(
        default=EVAL_GROUNDEDNESS_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum groundedness for regression gate",
    )
    eval_refusal_threshold: float = Field(
        default=EVAL_REFUSAL_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum refusal correctness for regression gate",
    )
    eval_citation_threshold: float = Field(
        default=EVAL_CITATION_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum citation correctness for regression gate",
    )

    # --- Application ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )
    app_env: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment",
    )

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, value: int, info) -> int:
        """Ensure chunk overlap is smaller than chunk size."""
        chunk_size = info.data.get("chunk_size", DEFAULT_CHUNK_SIZE)
        if value >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({value}) must be less than chunk_size ({chunk_size})"
            )
        return value

    @property
    def effective_embedding_dimensions(self) -> int:
        """Return the embedding dimensions based on the selected provider."""
        if self.embedding_provider == "huggingface":
            return self.hf_embedding_dimensions
        return self.embedding_dimensions

    @property
    def mongodb_uri_str(self) -> str:
        """Return the MongoDB URI as a plain string."""
        return self.mongodb_uri.get_secret_value()

    @property
    def langsmith_api_key_str(self) -> str | None:
        """Return the LangSmith API key as a plain string, if set."""
        if self.langsmith_api_key is None:
            return None
        return self.langsmith_api_key.get_secret_value()

    @property
    def groq_api_key_str(self) -> str | None:
        """Return the Groq API key as a plain string, if set."""
        if self.groq_api_key is None:
            return None
        return self.groq_api_key.get_secret_value()

    @property
    def huggingface_api_key_str(self) -> str | None:
        """Return the HuggingFace API key as a plain string, if set."""
        if self.huggingface_api_key is None:
            return None
        return self.huggingface_api_key.get_secret_value()

    @property
    def openai_api_key_str(self) -> str | None:
        """Return the OpenAI API key as a plain string, if set."""
        if self.openai_api_key is None:
            return None
        return self.openai_api_key.get_secret_value()

    def configure_langsmith_env(self) -> None:
        """Push LangSmith settings into process environment for SDK auto-discovery."""
        if self.langsmith_api_key_str:
            os.environ["LANGSMITH_API_KEY"] = self.langsmith_api_key_str
        os.environ["LANGSMITH_PROJECT"] = self.langsmith_project
        os.environ["LANGSMITH_TRACING"] = str(self.langsmith_tracing).lower()

    def is_production(self) -> bool:
        """Return True when running in production."""
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    settings = Settings()
    settings.configure_langsmith_env()
    return settings


def reset_settings_cache() -> None:
    """Clear the settings cache (useful in tests)."""
    get_settings.cache_clear()