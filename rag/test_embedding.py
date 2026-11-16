"""
test_embedding.py  -  run this BEFORE building the index. It checks the embedding model itself.

Run:  python test_embedding.py
The first run with the default model downloads it once (about 130 MB) and needs internet.

Checks: the model loads; vectors have the right size and unit length; the same text always gives the same
vector; batch and single encoding agree; chunks of the configured size fit the model; and (for the real model)
that meaning is understood: a question about "a woman whose husband died" finds the widow-pension text.
"""
import sys

import numpy as np

import embedders
import rag_config as config

results = []


def report(name, ok, detail="", warn=False):
    status = "PASS" if ok else ("WARN" if warn else "FAIL")
    results.append(status)
    print(f"[{status}] {name}" + (f"  -  {detail}" if detail else ""))


print(f"Backend: {config.EMBEDDING_BACKEND}   Model: {config.EMBEDDING_MODEL if config.EMBEDDING_BACKEND != 'hashing' else '(keyword fallback)'}")
try:
    e = embedders.get_embedder()
except embedders.EmbeddingSetupError as err:
    report("model loads", False, str(err))
    sys.exit(1)
info = e.describe()
report("model loads", True, f"{info['model']}, {info['dimension']} dimensions, input limit {info['max_tokens']} tokens")

docs = ["Monthly pension for widows aged above 40 whose family income is below the limit.",
        "Subsidy on certified seeds for small farmers owning up to five acres of land.",
        "Scholarship for students pursuing higher education in science and technology."]
V = e.encode_documents(docs)
report("vectors have the right shape", V.shape == (3, info["dimension"]), str(V.shape))
report("vectors are unit length", bool(np.allclose(np.linalg.norm(V, axis=1), 1.0, atol=1e-3)))
report("same text, same vector", bool(np.allclose(e.encode_documents(docs[:1]), V[:1], atol=1e-4)))
report("batch equals one-by-one", bool(np.allclose(np.vstack([e.encode_documents([d]) for d in docs]), V, atol=1e-3)))
report("token counter works", e.count_tokens("monthly pension for widows") >= 3, str(e.count_tokens("monthly pension for widows")))
room = e.max_tokens - (config.CHUNK_MAX_TOKENS + 60)
report("configured chunks fit the model's input limit", room >= 0,
       f"limit {e.max_tokens}, chunks up to {config.CHUNK_MAX_TOKENS} + header ~60" + ("" if room >= 0 else
       f"; lower CHUNK_MAX_TOKENS to {max(100, e.max_tokens - 120)}"))
if info["backend"] == "sentence-transformers":
    print(f"Query prefix : {e.query_prefix!r}\nDoc prefix   : {e.doc_prefix!r}")

if info["backend"] == "hashing":
    queries = {"pension for widows": 0, "subsidy for farmers seeds": 1, "scholarship for science students": 2}
    label = "keyword match finds the right text (fallback backend: words, not meaning)"
else:
    queries = {"financial help for a woman whose husband has died": 0,
               "support for growers to buy good quality planting material": 1,
               "money to study at college": 2}
    label = "meaning is understood (paraphrased questions find the right text)"
Q = e.encode_queries(list(queries))
best = [int(np.argmax(Q[i] @ V.T)) for i in range(len(queries))]
report(label, best == list(queries.values()), f"picked {best}, wanted {list(queries.values())}")

print(f"\n{results.count('PASS')} passed, {results.count('WARN')} warnings, {results.count('FAIL')} failed")
if results.count("FAIL"):
    sys.exit(1)
print("The embedding model is ready. Next: python build_index.py")
