"""Embedding and the vector index.

RAG step two: convert every chunk to a vector so that "similar
meaning" becomes "nearby in space." The workhorse is a sentence
transformer (all-MiniLM-L6-v2: small, fast, free, runs locally).
Vectors are L2-normalized at encode time, which makes cosine
similarity a plain dot product at query time.

A TF-IDF fallback exists so the pipeline runs anywhere; it matches on
word overlap rather than meaning, which is exactly the difference
embeddings were invented to fix. Comparing the two on the same
question is the single most instructive experiment in this project.

Index artifacts live in data/rag/ (gitignored like all data layers):
chunks.parquet, embeddings.npy, manifest.json, and tfidf.pkl when the
fallback is used.
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from .corpus import REPO, build_chunks

OUT_DIR = REPO / "data" / "rag"

SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SbertEmbedder:
    name = "sbert"

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(SBERT_MODEL)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        )


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self, fitted=None) -> None:
        self.vectorizer = fitted

    def fit(self, texts: list[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(max_features=4096, stop_words="english")
        self.vectorizer.fit(texts)

    def encode(self, texts: list[str]) -> np.ndarray:
        x = self.vectorizer.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return x / norms


def get_embedder():
    try:
        return SbertEmbedder()
    except ImportError:
        print("sentence-transformers not installed; falling back to TF-IDF")
        return TfidfEmbedder()


def build_index() -> None:
    chunks = build_chunks()
    texts = [f"{c.section}\n{c.text}" for c in chunks]
    emb = get_embedder()
    if isinstance(emb, TfidfEmbedder):
        emb.fit(texts)
    vectors = emb.encode(texts)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "source": [c.source for c in chunks],
            "section": [c.section for c in chunks],
            "text": [c.text for c in chunks],
        }
    ).to_parquet(OUT_DIR / "chunks.parquet", index=False)
    np.save(OUT_DIR / "embeddings.npy", vectors)
    manifest = {"embedder": emb.name, "n_chunks": len(chunks), "dim": int(vectors.shape[1])}
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if emb.name == "tfidf":
        with open(OUT_DIR / "tfidf.pkl", "wb") as f:
            pickle.dump(emb.vectorizer, f)
    print(f"indexed {len(chunks)} chunks from the corpus ({emb.name}, dim {vectors.shape[1]})")


def load_index():
    manifest = json.loads((OUT_DIR / "manifest.json").read_text())
    chunks = pd.read_parquet(OUT_DIR / "chunks.parquet")
    vectors = np.load(OUT_DIR / "embeddings.npy")
    if manifest["embedder"] == "sbert":
        emb = SbertEmbedder()
    else:
        with open(OUT_DIR / "tfidf.pkl", "rb") as f:
            emb = TfidfEmbedder(fitted=pickle.load(f))
    return chunks, vectors, emb
