from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION = "documents"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_TOKENS = 500
CHUNK_OVERLAP = 75
