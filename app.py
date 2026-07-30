"""
Capstone RAG — Dark Neon Streamlit Frontend

A stunning cyberpunk-themed UI for the self-correcting RAG agent.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

import requests
import streamlit as st

# ── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Capstone RAG — Neon",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom Dark Neon CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background: #0a0a12;
        color: #e0e0ff;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
    }
    .main > div { background: #0a0a12; }

    h1, h2, h3 {
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: 0.05em;
    }
    h1 {
        background: linear-gradient(135deg, #00f0ff, #a855f7, #ff00aa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800 !important;
        text-shadow: 0 0 30px rgba(168, 85, 247, 0.3);
    }
    h2 { color: #00f0ff !important; border-bottom: 1px solid #1a1a3e; padding-bottom: 0.5rem; }
    h3 { color: #a855f7 !important; }

    .neon-card {
        background: linear-gradient(135deg, #12122a, #1a0a2e);
        border: 1px solid #2a1a4e;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 0 20px rgba(168, 85, 247, 0.1), inset 0 0 20px rgba(168, 85, 247, 0.03);
        transition: all 0.3s ease;
    }
    .neon-card:hover {
        border-color: #a855f7;
        box-shadow: 0 0 30px rgba(168, 85, 247, 0.2), inset 0 0 30px rgba(168, 85, 247, 0.05);
    }

    .badge-success {
        background: linear-gradient(135deg, #00ff88, #00cc66);
        color: #000;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .badge-warning {
        background: linear-gradient(135deg, #ffaa00, #ff8800);
        color: #000;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .badge-refused {
        background: linear-gradient(135deg, #ff00aa, #cc0066);
        color: #fff;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
    }

    .metric-box {
        background: linear-gradient(135deg, #0d0d20, #1a0a2e);
        border: 1px solid #2a1a4e;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00f0ff, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-label {
        color: #8888aa;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.25rem;
    }

    .user-message {
        background: linear-gradient(135deg, #1a1a4e, #2a1a3e);
        border: 1px solid #3a2a6e;
        border-radius: 16px 16px 4px 16px;
        padding: 1rem 1.25rem;
        margin: 0.5rem 0;
        color: #e0e0ff;
    }
    .assistant-message {
        background: linear-gradient(135deg, #0d0d20, #1a0a2e);
        border: 1px solid #2a1a4e;
        border-radius: 16px 16px 16px 4px;
        padding: 1rem 1.25rem;
        margin: 0.5rem 0;
        color: #e0e0ff;
        border-left: 3px solid #a855f7;
    }

    .stTextInput > div > div > input {
        background: #0d0d20 !important;
        border: 1px solid #2a1a4e !important;
        border-radius: 12px !important;
        color: #e0e0ff !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #a855f7 !important;
        box-shadow: 0 0 15px rgba(168, 85, 247, 0.2) !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #a855f7, #7c3aed) !important;
        border: none !important;
        border-radius: 12px !important;
        color: #fff !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        padding: 0.5rem 2rem !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .stButton > button:hover {
        box-shadow: 0 0 25px rgba(168, 85, 247, 0.4) !important;
        transform: translateY(-2px);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #08081a, #0d0d20) !important;
        border-right: 1px solid #1a1a3e;
    }
    section[data-testid="stSidebar"] .stMarkdown h3 { color: #a855f7 !important; }

    .streamlit-expanderHeader {
        background: #0d0d20 !important;
        border: 1px solid #2a1a4e !important;
        border-radius: 12px !important;
        color: #e0e0ff !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    .streamlit-expanderContent {
        background: #0a0a12 !important;
        border: 1px solid #2a1a4e !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: #0d0d20;
        border-bottom: 1px solid #2a1a4e;
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8888aa;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
    }
    .stTabs [aria-selected="true"] {
        color: #a855f7 !important;
        border-bottom: 2px solid #a855f7 !important;
    }

    .footer-glow {
        text-align: center;
        padding: 2rem;
        color: #444466;
        font-size: 0.8rem;
        border-top: 1px solid #1a1a3e;
        margin-top: 3rem;
    }

    ::-webkit-scrollbar { width: 8px; background: #0a0a12; }
    ::-webkit-scrollbar-thumb { background: #2a1a4e; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #a855f7; }

    @keyframes glow-pulse {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
    .glow-pulse { animation: glow-pulse 2s ease-in-out infinite; }
</style>
""", unsafe_allow_html=True)

# ── API Configuration ──────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
API_HEALTH = f"{API_BASE}/api/health"
API_CHAT = f"{API_BASE}/api/chat"
API_METRICS = f"{API_BASE}/api/metrics"


def get_session_id() -> str:
    """Get or create a persistent session ID for this browser session."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id


def check_api_health() -> bool:
    try:
        resp = requests.get(API_HEALTH, timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def chat_agent(session_id: str, message: str) -> dict[str, Any] | None:
    try:
        resp = requests.post(
            API_CHAT,
            json={"session_id": session_id, "message": message},
            timeout=120,
        )
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


def get_metrics_data() -> dict[str, Any] | None:
    try:
        resp = requests.get(API_METRICS, timeout=5)
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


# ── Session State ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_healthy" not in st.session_state:
    st.session_state.api_healthy = False

session_id = get_session_id()


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚀 CAPSTONE RAG")
    st.markdown("*Self-Correcting RAG Agent*")
    st.markdown("---")

    api_ok = check_api_health()
    st.session_state.api_healthy = api_ok
    if api_ok:
        st.markdown('<span class="badge-success">● API ONLINE</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-refused">● API OFFLINE</span>', unsafe_allow_html=True)
        st.info("Start: `uvicorn src.main:app --reload --port 8000`")

    st.markdown("---")

    metrics_data = get_metrics_data() if api_ok else None
    if metrics_data:
        st.markdown("### 📊 API Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Queries", metrics_data.get("total_queries", 0))
        with col2:
            st.metric("Avg Time", f"{metrics_data.get('avg_response_time_ms', 0):.0f}ms")

    st.markdown("---")
    st.markdown("### 💡 Try These")
    example_queries = [
        "Find me a quiet apartment near the beach with wifi",
        "What are the best-reviewed properties in Barcelona?",
        "Show me apartments in Tokyo with amazing city views",
        "Are there any treehouse stays available?",
        "Compare apartments in Paris and Rome for a romantic getaway",
        "What properties have swimming pools?",
    ]
    for q in example_queries:
        if st.button(q, key=f"ex_{q[:20]}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()

    st.markdown("---")
    st.markdown("### 🔧 System Info")
    st.markdown("**Chat:** Groq (llama-3.3-70b-versatile)")
    st.markdown("**Embeddings:** HuggingFace (all-MiniLM-L6-v2)")
    st.markdown("**Vector Store:** MongoDB Atlas")
    st.markdown("**Dimensions:** 384")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main Content ────────────────────────────────────────────────────────────
st.markdown("# 🚀 Capstone RAG")
st.markdown(
    '<p style="color: #8888aa; margin-top: -0.5rem;">'
    "Self-correcting RAG agent · MongoDB Vector Search · LangGraph · Groq + HuggingFace"
    "</p>",
    unsafe_allow_html=True,
)

# ── Metrics Row ─────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
metrics_display = {
    "Recall@5": {"value": "≥ 0.70", "desc": "Target threshold"},
    "Groundedness": {"value": "≥ 0.90", "desc": "Citation grounding"},
    "Refusal Acc.": {"value": "1.00", "desc": "Correct refusal"},
    "Citation Acc.": {"value": "≥ 0.90", "desc": "Citation accuracy"},
}
for (label, info), col in zip(metrics_display.items(), [col1, col2, col3, col4]):
    with col:
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-value">{info["value"]}</div>'
            f'<div class="metric-label">{label}<br>{info["desc"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("---")

# ── Chat Messages ───────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-message">'
            f'<strong style="color: #00f0ff;">🧑 YOU</strong><br>{msg["content"]}'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        answer = msg.get("content", "")
        citations = msg.get("citations", [])
        retrieval_attempts = msg.get("retrieval_attempts", 1)
        refused = msg.get("refused", False)
        rewritten_queries = msg.get("rewritten_queries", [])
        duration = msg.get("duration_ms", 0)

        badge = (
            '<span class="badge-refused">REFUSED</span>'
            if refused
            else '<span class="badge-success">ANSWERED</span>'
        )

        citation_text = ""
        if citations:
            lines = []
            for c in citations:
                loc = c.get("location", {}) or {}
                parts = [loc.get("city", ""), loc.get("country", "")]
                loc_str = f" — {', '.join(filter(None, parts))}" if any(parts) else ""
                lines.append(
                    f"📌 **{c['listing_name']}** (ID: `{c['listing_id']}`)"
                    f"{loc_str} — score: `{c.get('score', 0):.4f}`"
                )
            citation_text = "\n".join(lines)

        rewrite_text = ""
        if rewritten_queries:
            rewrite_text = (
                "🔄 **Query Rewrites:**\n"
                + "\n".join(f"  `{i+1}. {q}`" for i, q in enumerate(rewritten_queries))
            )

        st.markdown(
            f'<div class="assistant-message">'
            f'<strong style="color: #a855f7;">🤖 RAG AGENT</strong> {badge}'
            f'<br><br>{answer}'
            f'{"<hr style=\"border-color: #2a1a4e; margin: 0.75rem 0;\">" if citation_text or rewrite_text else ""}'
            f'{citation_text}'
            f'{"<br>" if citation_text and rewrite_text else ""}'
            f'{rewrite_text}'
            f'<br><span style="color: #666688; font-size: 0.75rem;">'
            f'⚡ {duration:.0f}ms · {retrieval_attempts} attempt{"s" if retrieval_attempts > 1 else ""}'
            f'</span></div>',
            unsafe_allow_html=True,
        )

# ── Input ───────────────────────────────────────────────────────────────────
st.markdown("### 💬 Ask a Question")
col_input, col_button = st.columns([5, 1])
with col_input:
    user_question = st.text_input(
        "Question",
        placeholder="e.g., Find me a quiet apartment near the beach with wifi",
        label_visibility="collapsed",
        key="user_input",
    )
with col_button:
    send = st.button("🚀 ASK", use_container_width=True)

if send and user_question:
    if not st.session_state.api_healthy:
        st.error("API is not running. Start: `uvicorn src.main:app --reload --port 8000`")
    else:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.spinner("🧠 Reasoning..."):
            result = chat_agent(session_id, user_question)
        if result:
            st.session_state.messages.append({
                "role": "assistant",
                "content": result.get("answer", ""),
                "citations": result.get("citations", []),
                "retrieval_attempts": result.get("retrieval_attempts", 1),
                "rewritten_queries": result.get("rewritten_queries", []),
                "refused": result.get("refused", False),
                "duration_ms": result.get("duration_ms", 0),
            })
        else:
            st.error("Agent returned no response. Check the API logs.")
        st.rerun()

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer-glow">'
    "Capstone RAG · Built with LangGraph · MongoDB Atlas Vector Search · Groq · HuggingFace"
    "</div>",
    unsafe_allow_html=True,
)