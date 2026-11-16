"""
embedders.py  -  turns text into vectors (embeddings).

Two backends, one interface:
  SentenceTransformerEmbedder   a real semantic model that runs on your computer (default: BAAI/bge-small-en-v1.5).
                                The first use downloads the model once (about 130 MB); after that it works offline.
  HashingEmbedder               a keyword-based fallback with no download. For tests and offline work only:
                                it matches words, it does not understand meaning.

Every embedder returns unit-length vectors, so cosine similarity = dot product.
The index remembers which embedder built it, and the retriever always uses the same one for queries.
"""
import math
import re

import numpy as np

import rag_config as config


class EmbeddingSetupError(Exception):
    """The embedding model could not be loaded. The message says what to do."""


def approx_token_count(text: str) -> int:
    """
    Conservative token estimate for BERT-style tokenizers when no real tokenizer is available:
    every word and punctuation mark counts once, times 1.25 for word-piece splitting.
    """
    return math.ceil(len(re.findall(r"\w+|[^\w\s]", text)) * 1.25)


def default_prefixes(model_name: str):
    """(query_prefix, document_prefix) recommended for well-known retrieval models."""
    m = model_name.lower()
    if "bge" in m and "m3" not in m:
        return "Represent this sentence for searching relevant passages: ", ""
    if "e5" in m:
        return "query: ", "passage: "
    return "", ""


class Embedder:
    backend = "base"
    name = ""
    dimension = 0
    max_tokens = 512
    query_prefix = ""
    doc_prefix = ""

    def _encode(self, texts) -> np.ndarray:
        raise NotImplementedError

    def encode_documents(self, texts) -> np.ndarray:
        return self._encode([self.doc_prefix + t for t in texts])

    def encode_queries(self, texts) -> np.ndarray:
        return self._encode([self.query_prefix + t for t in texts])

    def count_tokens(self, text: str) -> int:
        return approx_token_count(text)

    def describe(self) -> dict:
        return {"backend": self.backend, "model": self.name, "dimension": int(self.dimension),
                "max_tokens": int(self.max_tokens), "query_prefix": self.query_prefix,
                "doc_prefix": self.doc_prefix}


class SentenceTransformerEmbedder(Embedder):
    backend = "sentence-transformers"

    def __init__(self, model_name: str, batch_size: int = 16, query_prefix=None, doc_prefix=None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbeddingSetupError(
                "sentence-transformers is not installed. Run:  pip install -r requirements.txt   "
                "(or set EMBEDDING_BACKEND=hashing in .env for the no-download fallback).") from e
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise EmbeddingSetupError(
                f"Could not load the embedding model '{model_name}'. The first use needs an internet connection "
                "(the model downloads once, about 130 MB, and is then cached). Check your connection, or set "
                f"EMBEDDING_MODEL in .env to a model name or a local folder. Details: {e}") from e
        q_default, d_default = default_prefixes(model_name)
        self.name = model_name
        self.batch_size = batch_size
        self.query_prefix = q_default if query_prefix is None else query_prefix
        self.doc_prefix = d_default if doc_prefix is None else doc_prefix
        self.dimension = int(self.model.get_sentence_embedding_dimension())
        self.max_tokens = int(self.model.max_seq_length)

    def _encode(self, texts) -> np.ndarray:
        vectors = self.model.encode(list(texts), batch_size=self.batch_size, show_progress_bar=False,
                                    normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vectors, dtype=np.float32)

    def count_tokens(self, text: str) -> int:
        try:
            return len(self.model.tokenizer(text, add_special_tokens=False, truncation=False,
                                            verbose=False)["input_ids"])
        except Exception:
            return approx_token_count(text)


class HashingEmbedder(Embedder):
    backend = "hashing"

    def __init__(self, dimension: int = 4096):
        from sklearn.feature_extraction.text import HashingVectorizer
        self._vectorizer = HashingVectorizer(n_features=dimension, ngram_range=(1, 2), alternate_sign=False,
                                             norm=None, stop_words="english", lowercase=True)
        self.name = f"hashing-{dimension}"
        self.dimension = dimension
        self.max_tokens = 512

    def _encode(self, texts) -> np.ndarray:
        matrix = self._vectorizer.transform(list(texts)).astype(np.float32)
        matrix.data = np.log1p(matrix.data)                      # sub-linear term frequency
        dense = matrix.toarray()
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (dense / norms).astype(np.float32)


def get_embedder(backend: str = None, model: str = None) -> Embedder:
    """Build the embedder named in the settings (or the one given)."""
    backend = (backend or config.EMBEDDING_BACKEND).lower()
    if backend == "hashing":
        return HashingEmbedder()
    if backend in ("sentence-transformers", "sentence_transformers", "st"):
        return SentenceTransformerEmbedder(model or config.EMBEDDING_MODEL, config.EMBEDDING_BATCH_SIZE,
                                           config.EMBEDDING_QUERY_PREFIX, config.EMBEDDING_DOC_PREFIX)
    raise EmbeddingSetupError(f"EMBEDDING_BACKEND must be 'sentence-transformers' or 'hashing', got '{backend}'.")
