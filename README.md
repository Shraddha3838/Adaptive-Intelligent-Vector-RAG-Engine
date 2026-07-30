# Capstone RAG — Self-Correcting RAG Agent

A production-ready, self-correcting Retrieval-Augmented Generation (RAG) system built on **LangGraph**, **MongoDB Atlas Vector Search**, **Groq** (free chat), and **HuggingFace** (free embeddings).

## Architecture

```
User Query → LangGraph Agent → MongoDB Atlas Vector Search → Relevance Grading
    ↺ (rewrite query if irrelevant, up to N retries)
    → Answer Generation → Citation Extraction → Response
```

The agent uses a **LangGraph state machine** with five nodes:
1. **Retrieve** — Embed query and search `rag_chunks` via Atlas Vector Search
2. **Grade Relevance** — LLM judges whether retrieved context is relevant
3. **Rewrite Query** — If irrelevant, rewrite the query and re-retrieve
4. **Generate** — Produce a grounded answer from the context
5. **Cite** — Attach source citations to the answer

## Features

- **Self-Correcting**: Automatically rewrites queries and retries when retrieval is insufficient
- **Free Stack**: Uses Groq (free LLM inference) + HuggingFace (local embeddings)
- **MongoDB Atlas Vector Search**: Semantic search over 8,627 Airbnb listing chunks
- **FastAPI Backend**: Production-ready REST API with Swagger docs
- **Streamlit Frontend**: Dark neon cyberpunk UI
- **LangSmith Tracing**: Optional observability integration
- **Session Management**: In-memory conversation tracking
- **Feedback Collection**: Rate and comment on agent responses
- **API Metrics**: Track queries, refusals, response times, and uptime

## Quick Start

### 1. Prerequisites

- Python 3.11+
- MongoDB Atlas cluster (free tier works)
- Groq API key (free) — [sign up](https://console.groq.com)
- (Optional) LangSmith API key for tracing — [sign up](https://smith.langchain.com)

### 2. Environment Setup

```bash
# Clone the repository
git clone <repo-url> && cd capstone-rag

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install sentence-transformers  # for local embeddings
```

### 3. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```ini
# Required: MongoDB Atlas
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority

# Required: Groq (free chat inference)
GROQ_API_KEY=gsk_your_groq_api_key
GROQ_CHAT_MODEL=llama-3.3-70b-versatile

# Required: HuggingFace (free embeddings)
HUGGINGFACE_API_KEY=hf_your_huggingface_token

# Optional: LangSmith tracing
# LANGSMITH_API_KEY=lsv2_pt_your_langsmith_key
LANGSMITH_TRACING=false
```

### 4. Ingest Data

The project uses MongoDB's `sample_airbnb` dataset. If you haven't loaded it:

```bash
# Load sample dataset into your Atlas cluster via MongoDB Atlas UI
# Then run the ingestion pipeline:
python -m src.ingest
```

This will:
- Extract text from 200 Airbnb listings
- Chunk into 800-character segments
- Generate embeddings via HuggingFace (all-MiniLM-L6-v2, 384 dims)
- Upsert into `rag_chunks` collection

### 5. Start the FastAPI Backend

```bash
uvicorn src.main:app --reload --port 8000
```

The API is now available at `http://localhost:8000`.

### 6. Start the Streamlit Frontend

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## API Endpoints

### Health Check

```bash
curl http://localhost:8000/api/health
```

Response: `{"status": "ok"}`

### Chat

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "user-123", "message": "Find me a quiet Airbnb near the beach with WiFi"}'
```

Response:
```json
{
  "answer": "Based on the available listings, I recommend...",
  "citations": [
    {
      "listing_id": "1104768",
      "listing_name": "2 Bdrm/2 Bath Family Suite Ocean View",
      "chunk_text": "Close to the beach. Malls everywhere...",
      "score": 0.865,
      "location": { "country": "United States", "market": "Oahu" }
    }
  ],
  "refused": false,
  "retrieved_docs": 5,
  "trace_id": ""
}
```

### Session History

```bash
curl http://localhost:8000/api/sessions/user-123
```

### Submit Feedback

```bash
curl -X POST http://localhost:8000/api/feedback \
  -H "Content-Type: application/json" \
  -d '{"session_id": "user-123", "message_index": 0, "rating": 5, "comment": "Great answer!"}'
```

### API Metrics

```bash
curl http://localhost:8000/api/metrics
```

### Swagger UI

Open `http://localhost:8000/docs` in your browser for interactive API documentation.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_api.py -v
```

## Project Structure

```
capstone-rag/
├── app.py                    # Streamlit frontend
├── src/
│   ├── api/                  # FastAPI layer (Phase 2)
│   │   ├── __init__.py
│   │   ├── dependencies.py   # Dependency injection (agent, sessions, metrics)
│   │   ├── main.py           # FastAPI app factory with CORS, lifespan, error handlers
│   │   ├── routes.py         # API endpoints: /health, /chat, /sessions, /feedback, /metrics
│   │   └── schemas.py        # Pydantic request/response models
│   ├── config.py             # Pydantic settings from .env
│   ├── constants.py          # App-wide constants
│   ├── graph.py              # LangGraph RAG agent (self-correcting workflow)
│   ├── ingest.py             # MongoDB ingestion pipeline
│   ├── main.py               # FastAPI entry point (re-exports src.api.main)
│   ├── models.py             # Domain models (Citation, AgentResponse, etc.)
│   ├── prompts.py            # LLM prompt templates
│   ├── retrieval.py          # Vector search + embedding layer
│   ├── routers/              # Legacy routers (backward compatible)
│   └── utils.py              # Logging, MongoDB client, helpers
├── tests/
│   └── test_api.py           # API endpoint tests
├── evals/                    # Evaluation framework
├── .env.example              # Environment template
└── requirements.txt
```

## Curl Examples

### Simple query
```bash
curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo", "message": "Show me apartments in Barcelona with great views"}' | jq
```

### Query that triggers refusal
```bash
curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo", "message": "What is the meaning of life?"}' | jq
```

### Check API health
```bash
curl -s http://localhost:8000/api/health | jq
```

## Postman Collection

1. Import the API URL: `http://localhost:8000`
2. Create a new request:
   - **GET /api/health** — No parameters
   - **POST /api/chat** — Body: `{"session_id": "test", "message": "your question"}`
   - **GET /api/sessions/{session_id}** — Path variable
   - **POST /api/feedback** — Body: `{"session_id": "test", "message_index": 0, "rating": 5}`
   - **GET /api/metrics** — No parameters
3. Set Content-Type header to `application/json` for POST requests

## Troubleshooting

### "API OFFLINE" in Streamlit
Ensure the FastAPI server is running on port 8000:
```bash
uvicorn src.main:app --reload --port 8000
```

### MongoDB connection fails
- Verify your `MONGODB_URI` in `.env` is correct
- Ensure your IP is whitelisted in MongoDB Atlas Network Access
- Check that the database user has read/write permissions

### "model_decommissioned" error
Update the Groq model in `.env`:
```ini
GROQ_CHAT_MODEL=llama-3.3-70b-versatile
```

### Sentence Transformers import error
```bash
pip install sentence-transformers
```

## License

MIT