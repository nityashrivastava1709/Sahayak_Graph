"""
vector_rag_baseline.py  -  BASELINE 2: vector-only RAG followed by ONE LLM call.

    from vector_rag_baseline import VectorRAGBaseline
    result = VectorRAGBaseline().run("I am a widow in Uttar Pradesh ...", k=10, state="UP")

Steps (this is the "standard RAG" the full SahayakGraph is compared against):
  1. find the best chunks with the Layer 3 vector search (optionally reordered by a cross-encoder reranker)
  2. group them by scheme, put the text in a prompt
  3. ONE Gemini call: "which of these schemes does this person qualify for, which documents, which steps?"
No knowledge graph, no verification, no second chance: whatever the LLM says is the answer.

The LLM may name a scheme that does not exist or that it was not shown. These are NOT silently removed:
  * ids that do not exist    -> listed in result["invalid_ids"]
  * real ids that were not retrieved -> kept, with retrieved = False
so the evaluation can count them as hallucinations. If the LLM call fails, the retrieval order is returned
with meta["llm_failed"] = True (so a run never crashes, and the failure is visible).
"""
import hashlib
import time

import l4_config as config

config.ensure_layer3()
import corpus  # noqa: E402  (Layer 3)
import rag_config  # noqa: E402  (Layer 3)
from retriever import Retriever, format_context  # noqa: E402  (Layer 3)

import l4_common  # noqa: E402
import llm_client  # noqa: E402
import reranker as rr  # noqa: E402

SYSTEM_PROMPT = """You help people in India find government welfare schemes. You are given a person's question and passages taken from official scheme descriptions. Each passage starts with [chunk id] and the scheme name; the scheme id is the part of the chunk id before '-c' (for example S012 in S012-c1).

Rules:
1. Use ONLY the passages. Do not use outside knowledge.
2. Recommend a scheme only if the passages suggest the person qualifies or could qualify. Skip schemes the person clearly does not qualify for (for example a different state or gender).
3. For each recommended scheme give: its scheme id exactly as in the chunk id, one sentence saying why it fits, the documents the passages say are needed, and the application steps the passages describe. If the passages do not say, return an empty list. Never guess.
4. List the best matches first and respect the maximum number of schemes stated in the question.
5. "answer" is a short plain-language summary for the person (at most 120 words)."""

RAG_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "schemes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "scheme_id": {"type": "string"},
                    "reason": {"type": "string"},
                    "required_documents": {"type": "array", "items": {"type": "string"}},
                    "application_steps": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["scheme_id", "reason", "required_documents", "application_steps"],
            },
        },
    },
    "required": ["answer", "schemes"],
}


class VectorRAGBaseline:
    def __init__(self, retriever=None, schemes=None, reranker=None, retrieve_k: int = 10, per_scheme: int = 2,
                 chunk_pool: int = 40):
        self.retriever = retriever or Retriever()
        schemes = schemes if schemes is not None else corpus.load_schemes()
        self.by_id = {s["scheme_id"]: s for s in schemes}
        self.reranker = reranker
        self.retrieve_k, self.per_scheme, self.chunk_pool = retrieve_k, per_scheme, chunk_pool
        self.name = "vector_rag+rerank" if reranker is not None else "vector_rag"

    # ------------------------------------------------------------------ steps
    def _retrieve(self, query, state):
        hits = self.retriever.search(query, k=self.chunk_pool, state=state)
        info = None
        if self.reranker is not None and hits:
            hits, info = rr.safe_rerank(self.reranker, query, hits)
        return l4_common.group_hits_by_scheme(hits, self.retrieve_k, self.per_scheme), info

    def _retrieval_only(self, groups, k):
        entries = []
        for rank, g in enumerate(groups[:k], start=1):
            entries.append(l4_common.scheme_entry(
                g["scheme_id"], g["scheme_name"], rank, score=g["score"],
                evidence=[{"chunk_id": c["chunk_id"], "text": c["text"]} for c in g["chunks"]],
                official_link=g["official_link"], state=g["state"]))
        return entries

    def _from_llm(self, parsed, groups, k):
        retrieved = {g["scheme_id"]: g for g in groups}
        entries, invalid, seen = [], [], set()
        for item in parsed["schemes"]:
            sid = str(item.get("scheme_id", "")).strip().upper()
            if not sid or sid in seen:
                continue
            seen.add(sid)
            scheme = self.by_id.get(sid)
            if scheme is None:
                invalid.append(sid)
                continue
            if len(entries) >= k:
                continue
            g = retrieved.get(sid)
            clean = lambda items: [str(x).strip() for x in items if str(x).strip()]
            entries.append(l4_common.scheme_entry(
                sid, scheme["scheme_name"], len(entries) + 1, score=g["score"] if g else None,
                reason=str(item.get("reason", "")).strip() or None,
                documents=clean(item.get("required_documents", [])), steps=clean(item.get("application_steps", [])),
                evidence=[{"chunk_id": c["chunk_id"], "text": c["text"]} for c in g["chunks"]] if g else [],
                retrieved=g is not None, official_link=scheme["official_link"], state=scheme["state"]))
        return entries, invalid

    # ------------------------------------------------------------------ public
    def run(self, query: str, k: int = 10, state=None, use_llm: bool = True) -> dict:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be a whole number >= 1")
        if state:
            canonical = rag_config.canonical_state(state)
            if canonical is None:
                raise ValueError(f"Unknown state '{state}'. Use a full name such as 'Madhya Pradesh' or a code such as 'MP'.")
            state = canonical
        started = time.perf_counter()
        groups, rerank_info = self._retrieve(query, state)
        meta = {"method": "vector_search" + ("+rerank" if self.reranker is not None else ""),
                "state_filter": state, "n_context_schemes": len(groups),
                "n_context_chunks": sum(len(g["chunks"]) for g in groups), "llm_used": False,
                "llm_failed": False, "model": None, "rerank": rerank_info}
        answer, invalid = None, []

        entries = self._retrieval_only(groups, k)
        if use_llm and groups:
            chunks = [c for g in groups for c in g["chunks"]]
            context = format_context(chunks, max_chars=40000)
            user = (f"PERSON'S QUESTION:\n{query}\n\nPASSAGES:\n{context}\n\n"
                    f"Recommend at most {k} schemes.")
            key = hashlib.md5((query + "|" + ",".join(c["chunk_id"] for c in chunks) + f"|{k}").encode()).hexdigest()
            parsed = llm_client.cached_llm_json("vector_rag", key, SYSTEM_PROMPT, user, RAG_SCHEMA)
            meta["llm_used"], meta["model"] = True, config.GEMINI_MODEL
            if parsed is None:
                meta["llm_failed"] = True                      # keep the retrieval order, flagged
            else:
                entries, invalid = self._from_llm(parsed, groups, k)
                answer = str(parsed.get("answer", "")).strip() or None
        meta["latency_s"] = round(time.perf_counter() - started, 4)
        return l4_common.make_result(self.name, query, entries, answer=answer, invalid_ids=invalid, meta=meta)

    def run_profile(self, profile: dict, k: int = 10, use_llm: bool = True) -> dict:
        """Ask the standard question built from a structured profile (the state becomes the filter)."""
        return self.run(l4_common.profile_to_query(profile), k=k, state=l4_common.profile_state(profile), use_llm=use_llm)
