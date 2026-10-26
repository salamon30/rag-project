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
# Thinking on gives better grounded answers and keeps [n] citations (eval showed
# thinking-off answers dropped citations and misread context). Reasoning can still
# exhaust the budget, so the generator falls back to thinking-off once if needed.
LLM_THINKING = os.getenv("LLM_THINKING", "1") == "1"
LLM_MAX_TOKENS = 8000
