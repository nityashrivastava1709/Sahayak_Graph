"""
test_layer3.py  -  proves the vector search behaves correctly.

Run (after  python build_index.py):   python test_layer3.py

Checks (nothing is hard-coded to your data, so it keeps working when Layer 1 data changes):
  * the index is consistent (validate_index.py) and uses the COSINE metric
  * chunking is deterministic
  * searching with a scheme's own words finds that scheme (recall / MRR)
  * a chunk's own text finds that chunk
  * every filter (state, level, category, section, scheme ids, min_score) returns only what it should
  * scores are ordered and within 0..1; schemes are grouped correctly; supporting text can be fetched for
    schemes found by the knowledge graph; context blocks cite chunk ids; bad input is rejected clearly
  * a few natural-language questions about well-known schemes (WARN only: depends on the embedding model)
"""
import json
import random
import sys
import tempfile
from pathlib import Path

import chromadb  # noqa: F401  (import check)

import chunker
import corpus
import embedders
import evaluate_retrieval
import rag_config as config
import validate_index
import vector_store
from retriever import Retriever, format_context

results = []


def check(name, ok, detail="", warn=False):
    status = "PASS" if ok else ("WARN" if warn else "FAIL")
    results.append(status)
    print(f"{status}  {name}" + (f"   {detail}" if detail and not ok else ""))


def metric_is_cosine() -> bool:
    """Chroma's default metric is NOT cosine. Prove that ours is: orthogonal vectors must be at distance 1."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        client = vector_store.open_client(Path(tmp))
        col = vector_store.create_collection(client, name="metric_check")
        vector_store.upsert(col, ["a", "b", "c"], [[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]], ["a", "b", "c"],
                            [{"x": 1}, {"x": 2}, {"x": 3}])
        r = col.query(query_embeddings=[[1.0, 0.0]], n_results=3)
        d = dict(zip(r["ids"][0], r["distances"][0]))
        return abs(d["a"]) < 1e-3 and abs(d["b"] - 1.0) < 1e-3 and abs(d["c"] - 0.2929) < 5e-3


def main():
    try:
        schemes = corpus.load_schemes()
    except corpus.CorpusError as e:
        raise SystemExit(f"DATA PROBLEM: {e}")
    try:
        r = Retriever()
    except vector_store.IndexMissingError as e:
        raise SystemExit(str(e))
    except embedders.EmbeddingSetupError as e:
        raise SystemExit(f"EMBEDDING PROBLEM: {e}")
    count_before = r.collection.count()
    print(f"Index: {r.meta.get('embedding_model')} | {count_before} chunks | {r.meta.get('chunk_strategy')} chunks\n")

    # ---------------- index and metric ----------------
    failed_checks = [t for s, t in validate_index.run_checks(r.collection, schemes, r.embedder) if s == "FAIL"]
    check("validate_index.py has no FAIL", not failed_checks, "; ".join(failed_checks))
    check("the cosine metric works as expected (orthogonal vectors are at distance 1)", metric_is_cosine())
    check("the retriever uses the index's own query prefix", r.embedder.query_prefix == r.meta.get("query_prefix", r.embedder.query_prefix))
    try:
        Retriever(embedder=embedders.HashingEmbedder(dimension=64))
        check("an embedder with the wrong dimension is rejected", False)
    except vector_store.IndexMismatchError:
        check("an embedder with the wrong dimension is rejected", True)
    a = corpus.build_chunks(schemes, r.embedder.count_tokens, int(r.meta["chunk_max_tokens"]), int(r.meta["chunk_overlap_tokens"]), r.meta["chunk_strategy"])
    b = corpus.build_chunks(schemes, r.embedder.count_tokens, int(r.meta["chunk_max_tokens"]), int(r.meta["chunk_overlap_tokens"]), r.meta["chunk_strategy"])
    check("chunking is deterministic (same ids and text every time)", [(c.chunk_id, c.text) for c in a] == [(c.chunk_id, c.text) for c in b])

    # ---------------- retrieval quality ----------------
    print()
    report = evaluate_retrieval.evaluate(r, schemes)
    for kind, floor_fail, floor_warn, key in (("name", 0.6, 0.85, "recall@5"), ("benefits", 0.5, 0.75, "recall@10"),
                                              ("eligibility", 0.4, 0.6, "recall@10")):
        got = report[kind][key]
        check(f"queries from the scheme {kind} find the scheme ({key} = {got:.2f}, MRR {report[kind]['mrr']:.2f})",
              got >= floor_fail, f"below {floor_fail}")
        if got < floor_warn:
            check(f"   {kind} {key} is at least {floor_warn}", False, f"{got:.2f}; try another embedding model or chunk size", warn=True)

    rng = random.Random(config.SEED)
    chunks = rng.sample(a, min(30, len(a)))
    top3 = sum(c.chunk_id in {h["chunk_id"] for h in r.search(c.text[:600], k=3)} for c in chunks)
    check(f"the first 600 characters of a chunk find that chunk in the top 3 ({top3}/{len(chunks)})",
          top3 >= 0.95 * len(chunks))

    # ---------------- scores ----------------
    print()
    hits = r.search("scholarship for students", k=10)
    scores = [h["score"] for h in hits]
    check("results are sorted best first", scores == sorted(scores, reverse=True))
    check("scores are between 0 and 1 and equal 1 - distance",
          all(0 <= h["score"] <= 1 for h in hits) and all(abs(h["score"] - max(0, 1 - h["distance"])) < 1e-3 for h in hits))
    check("k is respected", len(r.search("pension", k=3)) == 3 and len(r.search("pension", k=1)) == 1)
    check("hits are JSON-serialisable", bool(json.dumps(hits)))
    check("min_score removes weak hits", all(h["score"] >= 0.2 for h in r.search("scholarship for students", k=20, min_score=0.2)))

    # ---------------- filters ----------------
    print()
    states = r.known_states()
    sample_states = [s for s in states if s != "Central"][:6]
    ok_in = ok_only = True
    for st in sample_states:
        h1 = r.search("scheme for farmers and students", k=20, state=st)
        h2 = r.search("scheme for farmers and students", k=20, state=st, include_central=False)
        ok_in &= bool(h1) and all(h["state"] in (st, "Central") for h in h1)
        ok_only &= bool(h2) and all(h["state"] == st for h in h2)
    check(f"state filter returns only that state or Central ({len(sample_states)} states tried)", ok_in)
    check("state filter with include_central=False returns only that state", ok_only)
    check("a state code works (MP = Madhya Pradesh)", all(h["state"] in ("Madhya Pradesh", "Central") for h in r.search("farmers", k=10, state="MP")))
    check("level filter", all(h["level"] == "Central" for h in r.search("scheme", k=15, level="Central")) and
          all(h["level"] == "State" for h in r.search("scheme", k=15, level="State")))
    cats = r.known_categories()
    cat_ok = all(h["primary_category"] and cats[0] in (r.collection.get(ids=[h["chunk_id"]], include=["metadatas"])["metadatas"][0]["categories"])
                 for h in r.search("scheme", k=10, categories=[cats[0]]))
    check(f"category filter returns only that category ('{cats[0]}')", cat_ok)
    for section in chunker.SECTION_NAMES:
        hs = r.search("what is needed", k=10, sections=[section])
        check(f"section filter '{section}' returns only chunks containing that section",
              bool(hs) and all(section in h["sections"] for h in hs))
    if r.meta.get("chunk_strategy") == "merged":
        check("documents chunks really contain the documents text",
              all("Documents required:" in h["text"] for h in r.search("documents", k=10, sections=["documents"])))
    ids = [s["scheme_id"] for s in schemes[:5]]
    hs = r.search("scheme", k=30, scheme_ids=ids)
    check("scheme_ids filter returns only those schemes", bool(hs) and {h["scheme_id"] for h in hs} <= set(ids))
    combo = r.search("documents", k=10, state="Rajasthan", sections=["documents"], categories=[cats[0]])
    check("filters combine (state + section + category)",
          all(h["state"] in ("Rajasthan", "Central") and "documents" in h["sections"] for h in combo))
    check("empty scheme_ids list matches nothing", r.search("scheme", scheme_ids=[]) == [])
    check("a state with no schemes returns an empty list, not an error",
          r.search("scheme", state="Lakshadweep", include_central=False) == [] or all(h["state"] == "Lakshadweep" for h in r.search("scheme", state="Lakshadweep", include_central=False)))

    # ---------------- schemes, supporting text, context ----------------
    print()
    sr = r.search_schemes("scholarship for girls", k=5, per_scheme=2)
    sids = [s["scheme_id"] for s in sr]
    check("search_schemes: each scheme once, at most k, best first",
          len(sids) == len(set(sids)) <= 5 and [s["score"] for s in sr] == sorted((s["score"] for s in sr), reverse=True))
    check("search_schemes: chunks belong to their scheme and respect per_scheme",
          all(0 < len(s["chunks"]) <= 2 and all(c["scheme_id"] == s["scheme_id"] for c in s["chunks"]) for s in sr))
    check("search_schemes: a scheme's score is its best chunk's score", all(s["score"] == s["chunks"][0]["score"] for s in sr))
    want = [schemes[0]["scheme_id"], schemes[7]["scheme_id"], "S999"]
    cf = r.chunks_for_schemes("who can apply and income limit", want, per_scheme=2)
    check("chunks_for_schemes: every requested id is present, unknown ids are empty",
          list(cf) == want and cf["S999"] == [] and all(cf[w] for w in want[:2]))
    check("chunks_for_schemes: per_scheme limit and ownership",
          all(len(v) <= 2 and all(h["scheme_id"] == k for h in v) for k, v in cf.items()))
    one = schemes[0]["scheme_id"]
    allc = r.get_scheme_chunks(one)
    n_expected = len(r.collection.get(where={"scheme_id": one}, include=[])["ids"])
    check("get_scheme_chunks returns every chunk in reading order",
          len(allc) == n_expected and [c["chunk_index"] for c in allc] == list(range(n_expected)))
    ctx = format_context(r.search("water conservation", k=3), max_chars=700)
    check("format_context cites chunk ids and respects the size limit", ctx.startswith("[") and len(ctx) <= 700 and "-c" in ctx)
    check("format_context of nothing is empty", format_context([]) == "")

    # ---------------- bad input ----------------
    print()
    for label, kwargs in [("empty query", dict(query="  ")), ("k = 0", dict(query="x", k=0)), ("k as text", dict(query="x", k="5")),
                          ("unknown state", dict(query="x", state="Atlantis")), ("unknown section", dict(query="x", sections=["nope"])),
                          ("bad level", dict(query="x", level="Local")), ("huge query", dict(query="x" * 5000))]:
        try:
            r.search(**kwargs)
            check(f"{label} is rejected with a clear error", False)
        except ValueError:
            check(f"{label} is rejected with a clear error", True)
    check("searching never changes the index", r.collection.count() == count_before)

    # ---------------- natural-language spot checks (informative) ----------------
    print()
    spot = [("fellowship for women scientists to pursue a Ph.D.", "WISE"),
            ("financial support for girl child education and savings", "Ladli Laxmi"),
            ("awards for water conservation and river rejuvenation", "Water Awards"),
            ("training for registered construction workers", "Kaushal Vikas")]
    for query, name_part in spot:
        targets = {s["scheme_id"] for s in schemes if name_part.lower() in s["scheme_name"].lower()}
        if not targets:
            continue
        top = [s["scheme_id"] for s in r.search_schemes(query, k=5)]
        check(f"'{query}' finds a '{name_part}' scheme in the top 5", bool(targets & set(top)),
              f"top 5: {top}", warn=True)

    print(f"\n{results.count('PASS')} passed, {results.count('WARN')} warnings, {results.count('FAIL')} failed")
    if results.count("FAIL"):
        print("Some checks failed. If you changed Layer 1 data or settings, rebuild with  python build_index.py.")
        return 1
    print("Layer 3 is complete: search returns the right chunks, with metadata and working filters.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
