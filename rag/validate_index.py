"""
validate_index.py  -  integrity checks on the vector index.

Run:  python validate_index.py

PASS = requirement met   WARN = worth knowing, not an error   FAIL = rebuild with  python build_index.py

Checks: the index matches the CURRENT Layer 1 data (detects a stale index), every scheme has chunks,
chunk ids and numbering are consistent, no text was lost or cut off, metadata is complete, vectors
have the right size and length, and the cosine metric is in use.
"""
import re
import sys

import numpy as np

import chunker
import corpus
import embedders
import rag_config as config
import vector_store


def _words(text):
    return set(re.findall(r"\w+", text.lower()))


def run_checks(collection, schemes, embedder=None) -> list:
    out = []
    add = lambda ok, text, warn=False: out.append(("PASS" if ok else ("WARN" if warn else "FAIL"), text))
    meta = dict(collection.metadata or {})
    data = collection.get(include=["documents", "metadatas", "embeddings"])
    ids, docs, metas, vectors = data["ids"], data["documents"], data["metadatas"], data["embeddings"]

    add(meta.get("hnsw:space") == "cosine", f"distance metric is cosine (found: {meta.get('hnsw:space')})")
    add(len(ids) == len(set(ids)) and len(ids) > 0, f"{len(ids)} chunks stored, all ids unique")

    # ---- is the index built from the current Layer 1 data?
    current_hash = corpus.data_hash(schemes)
    add(meta.get("data_hash") == current_hash,
        "index matches the current schemes_clean.csv (a mismatch means Layer 1 data changed: rebuild)")

    # ---- every scheme has chunks, numbering is contiguous
    by_scheme = {}
    for i, m in zip(ids, metas):
        by_scheme.setdefault(m["scheme_id"], []).append((m["chunk_index"], i, m))
    scheme_ids = {s["scheme_id"] for s in schemes}
    missing = sorted(scheme_ids - set(by_scheme))
    extra = sorted(set(by_scheme) - scheme_ids)
    add(not missing, f"every scheme has at least one chunk{'' if not missing else ': missing ' + str(missing[:8])}")
    add(not extra, f"no chunks for schemes that are not in schemes_clean.csv{'' if not extra else ': ' + str(extra[:8])}")
    bad_numbering = []
    for sid, items in by_scheme.items():
        idx = sorted(i for i, _, _ in items)
        n_declared = {m["n_chunks"] for _, _, m in items}
        if idx != list(range(len(items))) or n_declared != {len(items)}:
            bad_numbering.append(sid)
    add(not bad_numbering, f"chunk numbering is contiguous and n_chunks is correct{'' if not bad_numbering else ': ' + str(bad_numbering[:8])}")

    # ---- metadata and text
    required = ["scheme_id", "scheme_name", "section", "sections", "state", "level", "primary_category", "tokens"]
    incomplete = [i for i, m in zip(ids, metas) if any(m.get(k) in (None, "") for k in required)]
    add(not incomplete, f"metadata complete on every chunk{'' if not incomplete else ': ' + str(incomplete[:5])}")
    no_link = [i for i, m in zip(ids, metas) if not m.get("official_link")]
    add(not no_link, f"every chunk carries the official link ({len(no_link)} without)", warn=True)
    empty = [i for i, d in zip(ids, docs) if not (d or "").strip()]
    add(not empty, f"no empty chunk text{'' if not empty else ': ' + str(empty[:5])}")
    max_tokens = int(meta.get("chunk_max_tokens", config.CHUNK_MAX_TOKENS))
    over = [i for i, m in zip(ids, metas) if m["tokens"] > max_tokens]
    add(not over, f"no chunk exceeds {max_tokens} tokens{'' if not over else ': ' + str(over[:5])}")

    # ---- no text lost: every word of every section is in some chunk of its scheme
    chunk_words = {}
    for sid, items in by_scheme.items():
        chunk_words[sid] = set()
    for i, d, m in zip(ids, docs, metas):
        chunk_words[m["scheme_id"]] |= _words(d)
    lost = []
    for s in schemes:
        have = chunk_words.get(s["scheme_id"], set())
        text = " ".join(str(s[c]) for _, c, _ in chunker.SECTIONS) + " " + str(s["tags"])
        if _words(text) - have:
            lost.append(s["scheme_id"])
    add(not lost, f"no text lost: every word of every section is in a chunk{'' if not lost else ': ' + str(lost[:8])}")

    # ---- vectors
    V = np.asarray(vectors, dtype=np.float32)
    dim = int(meta.get("dimension", V.shape[1] if V.ndim == 2 else 0))
    add(V.ndim == 2 and V.shape == (len(ids), dim), f"vectors have the expected shape {(len(ids), dim)} (found {tuple(V.shape)})")
    norms = np.linalg.norm(V, axis=1)
    add(bool(np.all(np.isfinite(V))) and bool(np.all(np.abs(norms - 1) < 0.02)),
        "all vectors are finite and unit length")

    # ---- exact comparison with a fresh chunking of the current data (needs the same tokenizer)
    try:
        embedder = embedder or embedders.get_embedder(meta.get("embedding_backend"), meta.get("embedding_model"))
    except embedders.EmbeddingSetupError as e:
        add(False, f"could not load the embedding model for the exact comparison ({str(e)[:90]}...)", warn=True)
        return out
    expected = corpus.build_chunks(schemes, embedder.count_tokens, int(meta.get("chunk_max_tokens", config.CHUNK_MAX_TOKENS)),
                                   int(meta.get("chunk_overlap_tokens", config.CHUNK_OVERLAP_TOKENS)),
                                   meta.get("chunk_strategy", config.CHUNK_STRATEGY))
    stored = dict(zip(ids, docs))
    exp = {c.chunk_id: c.text for c in expected}
    add(set(stored) == set(exp), f"the stored chunk ids equal a fresh chunking of the current data "
                                 f"({len(stored)} stored, {len(exp)} expected)")
    differing = [i for i in exp if i in stored and stored[i] != exp[i]]
    add(not differing, f"stored chunk texts equal a fresh chunking{'' if not differing else ': ' + str(differing[:5])}")
    if (meta.get("chunk_max_tokens"), meta.get("chunk_strategy")) != (config.CHUNK_MAX_TOKENS, config.CHUNK_STRATEGY):
        add(False, "the index was built with different chunk settings than the current .env", warn=True)
    return out


def main():
    try:
        schemes = corpus.load_schemes()
    except corpus.CorpusError as e:
        raise SystemExit(f"DATA PROBLEM: {e}")
    try:
        collection = vector_store.get_collection(vector_store.open_client())
    except vector_store.IndexMissingError as e:
        raise SystemExit(str(e))
    checks = run_checks(collection, schemes)
    for status, text in checks:
        print(f"{status}  {text}")
    statuses = [s for s, _ in checks]
    print(f"\n{statuses.count('PASS')} passed, {statuses.count('WARN')} warnings, {statuses.count('FAIL')} failed")
    if statuses.count("FAIL"):
        print("Rebuild with  python build_index.py  and run this again.")
        return 1
    print("The index is consistent with the Layer 1 data. Next: python test_layer3.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
