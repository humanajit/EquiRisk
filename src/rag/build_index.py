"""
src/rag/build_index.py

Builds a per-ticker FAISS index over a small corpus assembled from:
  - recent news headlines/descriptions (from the processed feature table)
  - latest computed stats (risk label, volatility, sentiment trend)
  - a plain-language "risk explanation" blurb

This runs after ML inference (predict.py) so the corpus can include the
freshly computed risk score/label -- the chat should be able to explain
*why* a stock got its current label, not just recite news.

Each ticker gets its own small FAISS index (rather than one giant index
across all 150 tickers) since queries are always scoped to one company
in this dashboard -- keeps retrieval fast and avoids cross-ticker noise.
"""

import io
import logging

import faiss
import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer

from src.utils.s3_io import read_hive_partitioned_parquet_s3, put_bytes, put_json, vectorstore_key

logger = logging.getLogger("equirisk.rag.build_index")
logging.basicConfig(level=logging.INFO)

_embedder = None  # lazy-loaded, shared across tickers in one run


def _load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _get_embedder(model_name: str) -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(model_name)
    return _embedder


def _chunk_text(text: str, chunk_size_tokens: int, overlap_tokens: int) -> list:
    """Simple word-count-based chunking (approximates tokens closely
    enough for this corpus size -- headlines/blurbs are short, so exact
    tokenization isn't critical here). Swap in a real tokenizer-based
    splitter if you find chunks running noticeably over/under target."""
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(chunk_size_tokens - overlap_tokens, 1)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_size_tokens])
        if chunk.strip():
            chunks.append(chunk)
        if start + chunk_size_tokens >= len(words):
            break
    return chunks


def build_ticker_corpus(ticker_df: pd.DataFrame, ticker: str) -> list:
    """Assembles the raw text documents for one ticker: recent headlines
    (most recent 20 rows' worth), plus one summary doc describing the
    latest risk stats in plain language. Returns a list of raw text
    strings, pre-chunking."""
    docs = []

    recent = ticker_df.sort_values("date").tail(20)
    for _, row in recent.iterrows():
        if row.get("headlines"):
            for headline in row["headlines"]:
                if headline:
                    docs.append(f"[{row['date']}] {headline}")

    latest = ticker_df.sort_values("date").iloc[-1]
    summary = (
        f"As of {latest['date']}, {ticker} has a risk label of {latest.get('risk_label', 'unknown')}. "
        f"20-day volatility is {latest.get('volatility_20d', float('nan')):.4f}, "
        f"60-day volatility is {latest.get('volatility_60d', float('nan')):.4f}. "
        f"Recent average news sentiment (3-day) is {latest.get('sentiment_3d_avg', 0.0):.3f} "
        f"(range -1 very negative to +1 very positive), based on {int(latest.get('article_count', 0))} "
        f"recent articles."
    )
    docs.append(summary)
    return docs


def build_index_for_ticker(ticker: str, ticker_df: pd.DataFrame, config: dict) -> None:
    rag_config = config["rag"]
    bucket = config["s3"]["bucket"]

    docs = build_ticker_corpus(ticker_df, ticker)
    chunks = []
    for doc in docs:
        chunks.extend(_chunk_text(doc, rag_config["chunk_size_tokens"], rag_config["chunk_overlap_tokens"]))

    if not chunks:
        logger.warning(f"No chunks generated for {ticker} -- skipping index build")
        return

    embedder = _get_embedder(rag_config["embedding_model"])
    embeddings = embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])  # inner product on normalized vecs = cosine sim
    index.add(embeddings.astype("float32"))

    index_buf = io.BytesIO()
    faiss.write_index(index, faiss.PyCallbackIOWriter(index_buf.write))
    put_bytes(vectorstore_key(ticker, "index.faiss"), index_buf.getvalue(), bucket)

    put_json(vectorstore_key(ticker, "chunks.json").replace(".json", "") + ".json",
             {"chunks": chunks}, bucket)

    logger.info(f"Built index for {ticker}: {len(chunks)} chunks")


def refresh_all_indices(config_path: str = "config/config.yaml") -> None:
    """Main RAG entrypoint, called by the orchestrator after ML
    inference completes. Rebuilds every ticker's index from scratch --
    simpler and safer for a course project than incremental updates,
    and the data volume per ticker is small enough that full rebuild is
    fast."""
    config = _load_config(config_path)
    bucket = config["s3"]["bucket"]
    processed_prefix = config["s3"]["paths"]["processed_features"]

    full_df = read_hive_partitioned_parquet_s3(processed_prefix, bucket, partition_col="ticker")

    tickers = full_df["ticker"].unique().tolist()
    logger.info(f"Refreshing RAG indices for {len(tickers)} tickers")

    for ticker in tickers:
        ticker_df = full_df[full_df["ticker"] == ticker]
        try:
            build_index_for_ticker(ticker, ticker_df, config)
        except Exception as e:
            logger.error(f"Failed to build index for {ticker}: {e}")

    logger.info("RAG index refresh complete")


if __name__ == "__main__":
    refresh_all_indices()