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
# Cap on chunks per paper in the final top-k (0 = no cap). Without a cap the reranker
# can fill all k slots from one paper, which starves questions that need two papers.
# Split multi-paper questions into one search per paper (one extra LLM call per question).
# Hard set: multi-paper correctness 3.7 -> 5.0, original set unchanged at 5.0. DECOMPOSE=0 turns it off.
DECOMPOSE = os.getenv("DECOMPOSE", "1") == "1"
MAX_PER_SOURCE = int(os.getenv("MAX_PER_SOURCE", "0"))
# A kept split shares the context between papers: with k = 5, each of two papers got
# only two or three passages and the page with the figure was often not among them.
# So a split question gets this many passages in total instead of TOP_K.
MULTI_QUERY_K = int(os.getenv("MULTI_QUERY_K", "8"))
# Chunks end where the token window ends, often mid-sentence. Each hit is extended with
# the start of the chunk that follows it in the paper (this many characters; 0 = off).
NEIGHBOR_CHARS = int(os.getenv("NEIGHBOR_CHARS", "600"))
# When the first answer is a refusal, let the LLM propose one more search from what the
# passages revealed (for chained questions), then answer again. Measured and left off:
# in 70 questions it ran 8 times, 7 of them on unanswerable questions (refusal kept, two
# extra LLM calls each) and once on a chained question that it did not fix. FOLLOWUP=1 turns it on.
FOLLOWUP = os.getenv("FOLLOWUP", "0") == "1"
FOLLOWUP_K = 3
# Thinking on gives better grounded answers and keeps [n] citations (eval showed
# thinking-off answers dropped citations and misread context). Reasoning can still
# exhaust the budget, so the generator falls back to thinking-off once if needed.
LLM_THINKING = os.getenv("LLM_THINKING", "1") == "1"
LLM_MAX_TOKENS = 8000
