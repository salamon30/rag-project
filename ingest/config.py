import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION = "documents"

# bge-small has a 512-token window, so 500-token chunks are embedded in full
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# bge v1.5 convention: queries get this prefix, passages do not
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CHUNK_TOKENS = 500
CHUNK_OVERLAP = 75

LLM_BASE_URL = "https://integrate.api.nvidia.com/v1"
LLM_MODEL = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
TOP_K = 5

# Two-stage retrieval: the bi-encoder fetches RERANK_CANDIDATES chunks, a
# cross-encoder re-scores each (question, chunk) pair and the best TOP_K are kept.
# Set RERANK_MODEL=none to use vector search only. MiniLM matched bge-reranker-base
# on hit@5 and page_hit@5 in eval at ~6x lower latency (0.7 s vs 4.3 s per question).
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_CANDIDATES = 50
# Thinking on gives better grounded answers and keeps [n] citations (eval showed
# thinking-off answers dropped citations and misread context). Reasoning can still
# exhaust the budget, so the generator falls back to thinking-off once if needed.
LLM_THINKING = os.getenv("LLM_THINKING", "1") == "1"
LLM_MAX_TOKENS = 8000
