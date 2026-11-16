"""
evaluate_retrieval.py  -  a quick quality measurement of the index.

Run:  python evaluate_retrieval.py

For every scheme three queries are made from its OWN text, and we check where the right scheme ranks:
    name         the scheme's name
    benefits     the first 30 words of its benefits
    eligibility  the first 30 words of its eligibility text
Reported: recall@1, @3, @5, @10 (share of queries where the right scheme is in the top n) and MRR
(average of 1/rank). Saved to data/processed/rag_retrieval_report.json.

This is a sanity check, not a final benchmark: the queries reuse the schemes' own words, and generic
eligibility sentences ("The applicant should be a woman") are shared by many schemes, so eligibility
scores are naturally lower. Use the numbers to compare settings (model, chunk size) against each other.
"""
import json
import sys

import corpus
import rag_config as config
from retriever import Retriever

CUTOFFS = (1, 3, 5, 10)


def first_words(text: str, n: int = 30) -> str:
    return " ".join(str(text).split()[:n])


def make_queries(scheme: dict) -> list:
    queries = [("name", scheme["scheme_name"])]
    for kind, column in (("benefits", "benefits"), ("eligibility", "eligibility")):
        snippet = first_words(scheme[column])
        if len(snippet.split()) >= 4:
            queries.append((kind, snippet))
    return queries


def evaluate(retriever, schemes) -> dict:
    ranks = {"name": [], "benefits": [], "eligibility": []}
    for s in schemes:
        for kind, query in make_queries(s):
            found = [r["scheme_id"] for r in retriever.search_schemes(query, k=max(CUTOFFS))]
            ranks[kind].append(found.index(s["scheme_id"]) + 1 if s["scheme_id"] in found else None)
    report = {}
    for kind, rs in ranks.items():
        n = len(rs)
        if not n:
            continue
        report[kind] = {"queries": n,
                        **{f"recall@{c}": round(sum(1 for r in rs if r is not None and r <= c) / n, 3) for c in CUTOFFS},
                        "mrr": round(sum(1 / r for r in rs if r) / n, 3)}
    return report


def main():
    schemes = corpus.load_schemes()
    retriever = Retriever()
    print(f"Index: {retriever.meta.get('embedding_model')}  |  {retriever.count} chunks  |  "
          f"{retriever.meta.get('chunk_strategy')} chunks (max {retriever.meta.get('chunk_max_tokens')} tokens)\n")
    report = evaluate(retriever, schemes)
    print(f"{'query type':12} {'queries':>8} " + " ".join(f"{'R@' + str(c):>7}" for c in CUTOFFS) + f" {'MRR':>7}")
    for kind, m in report.items():
        print(f"{kind:12} {m['queries']:>8} " + " ".join(f"{m['recall@' + str(c)]:>7.2f}" for c in CUTOFFS) + f" {m['mrr']:>7.2f}")
    out = {"index": {k: v for k, v in retriever.meta.items() if k in ("embedding_model", "chunk_strategy", "chunk_max_tokens", "n_chunks")},
           "results": report}
    config.RETRIEVAL_REPORT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved {config.RETRIEVAL_REPORT}")


if __name__ == "__main__":
    sys.exit(main())
