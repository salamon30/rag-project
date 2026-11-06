"""UI: streamlit run app/streamlit_app.py"""
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # streamlit runs this file as a script

import streamlit as st  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from app.limits import DailyCounter  # noqa: E402
from ingest import config  # noqa: E402
from ingest.store import get_collection  # noqa: E402
from rag.pipeline import RagPipeline  # noqa: E402

# Public demo only: cap new questions per day and per visitor. 0 = no limits (local use).
DEMO_DAILY_LIMIT = int(os.getenv("DEMO_DAILY_LIMIT", "0"))
DEMO_SESSION_LIMIT = int(os.getenv("DEMO_SESSION_LIMIT", "5"))
REPO_URL = "https://github.com/salamon30/rag-project"

st.set_page_config(page_title="Medical AI Paper QA", page_icon="📄", layout="wide")


def paper_list():
    """(arXiv id, file name) pairs, read from the download script so the list lives in one place."""
    script = (ROOT / "scripts" / "download_data.sh").read_text()
    return re.findall(r"^(\d{4}\.\d{4,5}) (\w+)$", script, re.M)


@st.cache_resource(show_spinner="First start: downloading the 10 papers and building the search index "
                                "(about 3 minutes, only once)...")
def ensure_index():
    """Hosted copies start without data/ and chroma_db/ (both are git-ignored): build them once."""
    if get_collection(config.CHROMA_DIR, config.COLLECTION).count() > 0:
        return
    config.DATA_DIR.mkdir(exist_ok=True)
    for arxiv_id, name in paper_list():
        target = config.DATA_DIR / f"{name}.pdf"
        if not target.exists():
            request = urllib.request.Request(f"https://arxiv.org/pdf/{arxiv_id}",
                                             headers={"User-Agent": "rag-project/0.1"})
            with urllib.request.urlopen(request, timeout=120) as response:
                target.write_bytes(response.read())
            time.sleep(3)  # arXiv asks for a pause between downloads
    import subprocess
    subprocess.run([sys.executable, "-m", "ingest.run", "--reset"], cwd=ROOT, check=True)


@st.cache_resource(show_spinner="Loading embedding model and vector store...")
def get_pipeline():
    ensure_index()
    return RagPipeline()


@st.cache_resource
def answer_cache():
    # Shared by all visitors: a repeated question (e.g. an example button) costs no API call
    return {}


@st.cache_resource
def daily_counter():
    return DailyCounter(Path("/tmp/rag_demo_usage.json"), DEMO_DAILY_LIMIT)


def eval_summary():
    report = ROOT / "eval" / "report.md"
    if not report.exists():
        return None
    rows = re.findall(r"^\| ([^|]+) \| ([0-9.]+) \|$", report.read_text(), re.M)
    return {k.strip(): v for k, v in rows}


with st.sidebar:
    st.header("Settings")
    k = st.slider("Chunks to retrieve (k)", 1, 10, config.TOP_K)
    show_text = st.toggle("Show retrieved text", value=True)
    st.divider()
    st.caption(f"Embeddings: `{config.EMBED_MODEL}`")
    st.caption(f"Reranker: `{config.RERANK_MODEL}`")
    st.caption(f"Query decomposition: {'on' if config.DECOMPOSE else 'off'}")
    st.caption(f"LLM: `{config.LLM_MODEL}`")
    summary = eval_summary()
    if summary:
        st.subheader("Last evaluation")
        for name in ("hit@5", "MRR@10", "faithfulness (1-5)", "correctness (1-5)"):
            if name in summary:
                st.metric(name, summary[name])

st.title("Medical AI Paper QA")
st.write("Ask questions about 10 papers on medical AI and retrieval-augmented generation. "
         "Answers use only the retrieved passages and cite them as [n].")
if DEMO_DAILY_LIMIT:
    left = daily_counter().remaining()
    st.caption(f"Public demo: {left} new question{'' if left == 1 else 's'} left today, "
               f"up to {DEMO_SESSION_LIMIT} per visitor. The example questions are always free. "
               f"An answer takes about 30 seconds. For unlimited use, [run it locally]({REPO_URL}).")

ensure_index()

examples = [
    "How did Med-PaLM 2 perform on MedQA?",
    "What three prompting techniques are combined in Medprompt?",
    "How large is the dataset used to develop MedSAM?",
]
cols = st.columns(len(examples))
for col, ex in zip(cols, examples, strict=True):
    if col.button(ex, use_container_width=True):
        st.session_state.question = ex

# ?q=... in the URL pre-fills and runs a question (shareable links)
if "question" not in st.session_state and st.query_params.get("q"):
    st.session_state.question = st.query_params["q"]

question = st.text_input("Your question", key="question",
                         placeholder="e.g. What is the MIRAGE benchmark?")

if question:
    answers = answer_cache()
    key = (question.strip().lower(), k)
    result = answers.get(key)
    if result is None:
        if DEMO_DAILY_LIMIT and question not in examples:  # examples do not count against the limits
            asked = st.session_state.get("asked", 0)
            if asked >= DEMO_SESSION_LIMIT:
                st.warning(f"You have used the {DEMO_SESSION_LIMIT} questions of this demo session. "
                           f"Thanks for trying it! The code and setup are on [GitHub]({REPO_URL}).")
                st.stop()
            if not daily_counter().take():
                st.warning("Today's question budget for the public demo is used up; it resets at midnight UTC. "
                           f"You can still click the example questions, or [run it locally]({REPO_URL}).")
                st.stop()
            st.session_state.asked = asked + 1
        try:
            pipe = get_pipeline()
            with st.spinner("Retrieving and generating..."):
                result = pipe.ask(question, k)
        except Exception as e:
            st.error(f"Could not answer: {e}")
            st.stop()
        answers[key] = result

    st.subheader("Answer")
    st.markdown(result.answer)
    if len(result.queries) > 1:
        st.caption("Split into one search per paper: " + " · ".join(f"“{q}”" for q in result.queries))

    st.subheader("Sources")
    for i, h in enumerate(result.hits, start=1):
        if h.vector_rank is not None and h.score != h.vector_score:
            label = (f"[{i}] {h.source} · page {h.page} · relevance {h.score:.2f}"
                     f" · vector rank {h.vector_rank}")
        else:
            label = f"[{i}] {h.source} · page {h.page} · relevance score {h.score:.3f}"
        if show_text:
            with st.expander(label):
                st.text(h.text)
        else:
            st.write(label)
