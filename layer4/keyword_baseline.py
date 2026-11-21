"""
keyword_baseline.py  -  BASELINE 1: conventional keyword search (BM25) over the scheme text.

    from keyword_baseline import KeywordBaseline
    result = KeywordBaseline().run("widow pension in Uttar Pradesh", k=10, state="UP")

It ranks whole schemes by how well the words of the question match the scheme's name, description, benefits,
eligibility, application and documents text (the name counts triple). It does not understand meaning and does
not check eligibility. Output follows the common format in l4_common.py; `documents` and `steps` are None
because a keyword search does not provide them. `evidence` is the best-matching sentence of the scheme.
"""
import re
import time

import l4_config as config

config.ensure_layer3()
import chunker  # noqa: E402  (Layer 3)
import corpus  # noqa: E402  (Layer 3)
import rag_config  # noqa: E402  (Layer 3)

import l4_common  # noqa: E402
from bm25 import BM25, tokenize  # noqa: E402

SYSTEM_NAME = "keyword"
_TEXT_COLUMNS = [column for _, column, _ in chunker.SECTIONS]


def _scheme_text(s: dict) -> str:
    return " ".join([s["scheme_name"]] * 3 + [s["primary_category"], s["tags"]] + [str(s[c]) for c in _TEXT_COLUMNS])


class KeywordBaseline:
    name = SYSTEM_NAME

    def __init__(self, schemes=None):
        self.schemes = schemes if schemes is not None else corpus.load_schemes()
        self.bm25 = BM25().fit([_scheme_text(s) for s in self.schemes])

    def _best_sentence(self, scheme: dict, query_terms: set) -> dict:
        best, best_hits = None, 0
        for _, column, label in chunker.SECTIONS:
            for sentence in chunker.split_sentences(str(scheme[column])):
                hits = len(query_terms & set(tokenize(sentence)))
                if hits > best_hits:
                    best, best_hits = {"chunk_id": None, "text": f"{label}: {sentence}"}, hits
        return best

    def run(self, query: str, k: int = 10, state=None, include_central: bool = True) -> dict:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be a whole number >= 1")
        wanted_state = None
        if state:
            wanted_state = rag_config.canonical_state(state)
            if wanted_state is None:
                raise ValueError(f"Unknown state '{state}'. Use a full name such as 'Madhya Pradesh' or a code such as 'MP'.")
        started = time.perf_counter()
        allowed = {i for i, s in enumerate(self.schemes)
                   if wanted_state is None or s["state"] == wanted_state or (include_central and s["state"] == "Central")}
        ranked = [(i, sc) for i, sc in self.bm25.top(query, k=len(self.schemes)) if i in allowed][:k]
        terms = set(tokenize(query))
        entries = []
        for rank, (i, score) in enumerate(ranked, start=1):
            s = self.schemes[i]
            best = self._best_sentence(s, terms)
            entries.append(l4_common.scheme_entry(
                s["scheme_id"], s["scheme_name"], rank, score=score, evidence=[best] if best else [],
                official_link=s["official_link"], state=s["state"]))
        meta = {"latency_s": round(time.perf_counter() - started, 4), "method": "bm25",
                "n_candidates": len(allowed), "state_filter": wanted_state}
        return l4_common.make_result(SYSTEM_NAME, query, entries, meta=meta)

    def run_profile(self, profile: dict, k: int = 10) -> dict:
        """Ask the standard question built from a structured profile (the state becomes the filter)."""
        return self.run(l4_common.profile_to_query(profile), k=k, state=l4_common.profile_state(profile))
