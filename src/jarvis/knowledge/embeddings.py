"""
JARVIS-PRIME Local Embedding Engine
=====================================

Provides text embeddings without requiring cloud APIs.

Tiered approach (auto-selects best available):
    1. sentence-transformers (if installed) — Best quality, ~80MB model
    2. TF-IDF vectorizer (always available) — Good quality, zero downloads

Both produce vectors usable with ChromaDB for semantic search.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any


class TFIDFEmbedder:
    """
    Lightweight TF-IDF embedding engine.
    No external dependencies — uses pure Python + basic math.

    Produces fixed-dimension vectors via hashing trick,
    giving reasonable semantic search without any model downloads.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self._idf: dict[str, float] = {}
        self._doc_count = 0
        self._fitted = False

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenizer: lowercase, split on non-alpha, remove stopwords."""
        tokens = re.findall(r'[a-z0-9]+', text.lower())
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'shall',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'and',
            'but', 'or', 'not', 'no', 'if', 'then', 'than', 'so',
            'this', 'that', 'these', 'those', 'it', 'its',
        }
        return [t for t in tokens if t not in stopwords and len(t) > 1]

    def _hash_token(self, token: str) -> int:
        """Hash a token to a dimension index."""
        h = hashlib.md5(token.encode()).hexdigest()
        return int(h[:8], 16) % self.dim

    def fit(self, documents: list[str]) -> None:
        """Fit IDF values from a corpus of documents."""
        self._doc_count = len(documents)
        df: dict[str, int] = {}

        for doc in documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                df[token] = df.get(token, 0) + 1

        self._idf = {
            token: math.log((self._doc_count + 1) / (count + 1)) + 1
            for token, count in df.items()
        }
        self._fitted = True

    def embed(self, text: str) -> list[float]:
        """
        Embed a single text into a fixed-dimension vector.
        Uses TF-IDF with hashing trick for dimensionality reduction.
        """
        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * self.dim

        tf = Counter(tokens)
        total = len(tokens)

        vector = [0.0] * self.dim
        for token, count in tf.items():
            tf_val = count / total
            idf_val = self._idf.get(token, 1.0)
            idx = self._hash_token(token)
            # Use sign hash to reduce collisions
            sign = 1 if int(hashlib.md5(token.encode()).hexdigest()[8:12], 16) % 2 == 0 else -1
            vector[idx] += sign * tf_val * idf_val

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]

        return vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        return [self.embed(t) for t in texts]


class EmbeddingEngine:
    """
    Unified embedding interface.
    Automatically selects the best available backend.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self._backend_name = "tfidf"
        self._tfidf = TFIDFEmbedder(dim=dim)
        self._st_model: Any = None

        # Try to import sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
            self._backend_name = "sentence-transformers"
            self.dim = 384  # all-MiniLM-L6-v2 output dim
        except ImportError:
            pass

    @property
    def backend(self) -> str:
        return self._backend_name

    def embed(self, text: str) -> list[float]:
        """Embed a single text."""
        if self._st_model is not None:
            vec = self._st_model.encode(text, show_progress_bar=False)
            return vec.tolist()
        return self._tfidf.embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        if self._st_model is not None:
            vecs = self._st_model.encode(texts, show_progress_bar=False)
            return vecs.tolist()
        return self._tfidf.embed_batch(texts)

    def fit_tfidf(self, documents: list[str]) -> None:
        """Fit TF-IDF on a corpus (only needed for tfidf backend)."""
        self._tfidf.fit(documents)

    def stats(self) -> dict[str, Any]:
        return {
            "backend": self._backend_name,
            "dimension": self.dim,
            "model": "all-MiniLM-L6-v2" if self._st_model else "tfidf-hash",
        }
