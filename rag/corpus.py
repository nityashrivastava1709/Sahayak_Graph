"""
corpus.py  -  reads the Layer 1 scheme table and produces the chunks to index.

Reads  data/processed/schemes_clean.csv  (written by Layer 1 step F). Never changes it.
`data_hash` is a fingerprint of the text that gets indexed; if Layer 1 data changes, the hash changes and
validate_index.py tells you the vector index is out of date.
"""
import hashlib
import json
import re

import pandas as pd

import chunker
import rag_config as config


class CorpusError(Exception):
    """schemes_clean.csv is missing or malformed. The message says what to do."""


REQUIRED_COLUMNS = ["scheme_id", "scheme_name", "level", "state", "primary_category", "categories",
                    "official_link", "details", "benefits", "eligibility", "application", "documents", "tags"]


def load_schemes() -> list:
    path = config.SCHEMES_CSV
    if not path.exists():
        raise CorpusError(f"{path} not found. Run Layer 1 steps A-F first.")
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise CorpusError(f"{path.name} is missing columns {missing}. Found: {list(df.columns)}")
    ids = df["scheme_id"].str.strip().tolist()
    bad = [i for i in ids if not re.fullmatch(r"S\d{3}", i)]
    if bad:
        raise CorpusError(f"scheme_id values not in the form S001: {bad[:5]}")
    if len(set(ids)) != len(ids):
        raise CorpusError("scheme_id values in schemes_clean.csv are not unique.")
    schemes = df[REQUIRED_COLUMNS].to_dict("records")
    for s in schemes:
        s["scheme_id"] = s["scheme_id"].strip()
        if not s["scheme_name"].strip():
            raise CorpusError(f"{s['scheme_id']} has no scheme_name.")
    return schemes


def data_hash(schemes: list) -> str:
    payload = json.dumps([[s[c] for c in REQUIRED_COLUMNS] for s in schemes], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_chunks(schemes: list, count, max_tokens: int = None, overlap_tokens: int = None,
                 strategy: str = None) -> list:
    max_tokens = max_tokens or config.CHUNK_MAX_TOKENS
    overlap_tokens = config.CHUNK_OVERLAP_TOKENS if overlap_tokens is None else overlap_tokens
    strategy = strategy or config.CHUNK_STRATEGY
    chunks = []
    for scheme in schemes:
        chunks.extend(chunker.chunk_scheme(scheme, count, max_tokens, overlap_tokens, strategy))
    return chunks
