"""UI: streamlit run app/streamlit_app.py"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # streamlit runs this file as a script

import streamlit as st  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from ingest import config  # noqa: E402
from rag.pipeline import RagPipeline  # noqa: E402

st.set_page_config(page_title="Medical AI Paper QA", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model and vector store...")
def get_pipeline():
    return RagPipeline()


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

examples = [
    "How did Med-PaLM 2 perform on MedQA?",
    "What three prompting techniques are combined in Medprompt?",
    "How large is the dataset used to develop MedSAM?",
]
cols = st.columns(len(examples))
for col, ex in zip(cols, examples):
    if col.button(ex, use_container_width=True):
        st.session_state.question = ex

# ?q=... in the URL pre-fills and runs a question (shareable links)
if "question" not in st.session_state and st.query_params.get("q"):
    st.session_state.question = st.query_params["q"]

question = st.text_input("Your question", key="question",
                         placeholder="e.g. What is the MIRAGE benchmark?")

if question:
    try:
        pipe = get_pipeline()
        with st.spinner("Retrieving and generating..."):
            result = pipe.ask(question, k)
    except Exception as e:
        st.error(f"Could not answer: {e}")
        st.stop()

    st.subheader("Answer")
    st.markdown(result.answer)

    st.subheader("Sources")
    for i, h in enumerate(result.hits, start=1):
        label = f"[{i}] {h.source} · page {h.page} · similarity {h.score:.3f}"
        if show_text:
            with st.expander(label):
                st.text(h.text)
        else:
            st.write(label)
