# Container for the Streamlit demo. Hugging Face Spaces builds this file and serves port 7860.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Spaces run the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# CPU-only torch keeps the image about 2 GB smaller than the default CUDA build
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# Build the index into the image: download the 10 papers, chunk, embed, and cache the reranker,
# so the first visitor does not wait for model downloads
RUN ./scripts/download_data.sh \
 && python -m ingest.run --reset \
 && python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

EXPOSE 7860
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true", "--client.toolbarMode=minimal", "--browser.gatherUsageStats=false"]
