"""
retriever.py  -  searches the vector index. This is Layer 3's public interface.

    from retriever import Retriever
    r = Retriever()
    hits = r.search("pension for widows", k=5, state="Uttar Pradesh")
    r.search_schemes(...)            best schemes (grouped chunks)
    r.chunks_for_schemes(...)        supporting text for schemes the graph already found
    r.get_scheme_chunks("S012")      every chunk of one scheme, in order
    format_context(hits)             text block with chunk ids, ready for an LLM prompt

Each hit is a plain dict (JSON-serialisable):
    chunk_id, scheme_id, scheme_name, sections, text, score (0-1, higher = more similar), distance,
    state, level, primary_category, official_link, chunk_index, n_chunks, tokens

The retriever reads which embedder built the index and uses the same one for queries.
"""
import chunker
import embedders
import rag_config as config
import vector_store

MAX_QUERY_CHARS = 2000
MAX_K = 200


class Retriever:
    def __init__(self, embedder=None, client=None, collection=None):
        self.client = client or vector_store.open_client()
        self.collection = collection or vector_store.get_collection(self.client)
        self.meta = dict(self.collection.metadata or {})
        self.embedder = embedder or embedders.get_embedder(self.meta.get("embedding_backend"),
                                                           self.meta.get("embedding_model"))
        if "query_prefix" in self.meta:                       # use exactly what the index was built with
            self.embedder.query_prefix = self.meta["query_prefix"]
        built_dim = self.meta.get("dimension")
        if built_dim is not None and int(built_dim) != int(self.embedder.dimension):
            raise vector_store.IndexMismatchError(
                f"The index was built with {self.meta.get('embedding_model')} ({built_dim} dimensions) but the "
                f"embedder in use produces {self.embedder.dimension}. Rebuild with  python build_index.py")
        self.count = self.collection.count()

    # ------------------------------------------------------------------ checks
    @staticmethod
    def _check_query(query):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if len(query) > MAX_QUERY_CHARS:
            raise ValueError(f"query is too long ({len(query)} characters; the limit is {MAX_QUERY_CHARS})")

    @staticmethod
    def _check_k(k):
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_K:
            raise ValueError(f"k must be a whole number between 1 and {MAX_K}")
        return k

    def _where(self, state, include_central, level, categories, sections, scheme_ids):
        """Validated filter, or the string 'NOTHING' when the filters can match nothing."""
        if state:
            canonical = config.canonical_state(state) or ("Central" if str(state).strip().lower() == "central" else None)
            if canonical is None:
                raise ValueError(f"Unknown state '{state}'. Use a full name such as 'Madhya Pradesh' or a code such as 'MP'.")
            state = canonical
        if level:
            lvl = str(level).strip().lower()
            if lvl not in ("central", "state"):
                raise ValueError("level must be 'Central' or 'State'")
            level = lvl.capitalize()
        if sections:
            bad = [s for s in sections if s not in chunker.SECTION_NAMES]
            if bad:
                raise ValueError(f"Unknown section(s) {bad}. Choose from {chunker.SECTION_NAMES}")
        if scheme_ids is not None and len(scheme_ids) == 0:
            return "NOTHING"
        if categories is not None and len(categories) == 0:
            categories = None
        if sections is not None and len(sections) == 0:
            sections = None
        return vector_store.build_where(state, include_central, level, categories, sections, scheme_ids)

    # ------------------------------------------------------------------ results
    @staticmethod
    def _hit(chunk_id, text, meta, distance=None):
        hit = {"chunk_id": chunk_id, "scheme_id": meta["scheme_id"], "scheme_name": meta["scheme_name"],
               "sections": str(meta["sections"]).split("+"), "text": text,
               "state": meta["state"], "level": meta["level"], "primary_category": meta["primary_category"],
               "official_link": meta["official_link"], "chunk_index": meta["chunk_index"],
               "n_chunks": meta["n_chunks"], "tokens": meta["tokens"]}
        if distance is not None:
            hit["distance"] = round(float(distance), 4)
            hit["score"] = round(max(0.0, min(1.0, 1.0 - float(distance))), 4)
        return hit

    def _run(self, query, n, where):
        vector = self.embedder.encode_queries([query])[0]
        res = self.collection.query(query_embeddings=[[float(x) for x in vector]],
                                    n_results=max(1, min(n, self.count)), where=where,
                                    include=["documents", "metadatas", "distances"])
        return [self._hit(i, d, m, dist) for i, d, m, dist in
                zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])]

    # ------------------------------------------------------------------ public API
    def search(self, query, k=5, state=None, include_central=True, level=None, categories=None,
               sections=None, scheme_ids=None, min_score=None) -> list:
        """Top-k chunks for a query, best first. Filters are optional; see vector_store.build_where."""
        self._check_query(query)
        k = self._check_k(k)
        where = self._where(state, include_central, level, categories, sections, scheme_ids)
        if where == "NOTHING" or self.count == 0:
            return []
        hits = self._run(query, k, where)
        if min_score is not None:
            hits = [h for h in hits if h["score"] >= min_score]
        return hits

    def search_schemes(self, query, k=5, per_scheme=2, **filters) -> list:
        """
        Top-k SCHEMES for a query. Chunks are grouped by scheme; a scheme's score is its best chunk's score.
        Each result: scheme_id, scheme_name, score, state, level, primary_category, official_link,
        chunks (best `per_scheme` chunks).
        """
        k = self._check_k(k)
        if not isinstance(per_scheme, int) or per_scheme < 1:
            raise ValueError("per_scheme must be a whole number >= 1")
        hits = self.search(query, k=min(MAX_K, max(k * per_scheme * 4, 30)), **filters)
        grouped = {}
        for h in hits:                                               # hits are already best-first
            entry = grouped.setdefault(h["scheme_id"], {
                "scheme_id": h["scheme_id"], "scheme_name": h["scheme_name"], "score": h["score"],
                "state": h["state"], "level": h["level"], "primary_category": h["primary_category"],
                "official_link": h["official_link"], "chunks": []})
            if len(entry["chunks"]) < per_scheme:
                entry["chunks"].append(h)
        return list(grouped.values())[:k]

    def chunks_for_schemes(self, query, scheme_ids, per_scheme=2) -> dict:
        """
        Supporting text for schemes that were found another way (for example by the knowledge graph):
        the chunks of each given scheme that best match the query. Returns {scheme_id: [hits]};
        every requested id is present (an unknown id maps to an empty list).
        """
        self._check_query(query)
        ids = list(dict.fromkeys(scheme_ids or []))
        result = {sid: [] for sid in ids}
        if not ids or self.count == 0:
            return result
        where = {"scheme_id": {"$in": ids}}
        total = len(self.collection.get(where=where, include=[])["ids"])
        if total == 0:
            return result
        for h in self._run(query, total, where):
            if len(result[h["scheme_id"]]) < per_scheme:
                result[h["scheme_id"]].append(h)
        return result

    def get_scheme_chunks(self, scheme_id) -> list:
        """Every chunk of one scheme, in reading order (no score)."""
        res = self.collection.get(where={"scheme_id": scheme_id}, include=["documents", "metadatas"])
        hits = [self._hit(i, d, m) for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]
        return sorted(hits, key=lambda h: h["chunk_index"])

    def known_states(self) -> list:
        metas = self.collection.get(include=["metadatas"])["metadatas"]
        return sorted({m["state"] for m in metas})

    def known_categories(self) -> list:
        metas = self.collection.get(include=["metadatas"])["metadatas"]
        return sorted({c.strip() for m in metas for c in m["categories"].split("|") if c.strip()})


def format_context(hits, max_chars: int = 3000) -> str:
    """Turn hits into a text block for an LLM prompt. Every passage starts with its chunk id so answers can cite it."""
    blocks, used = [], 0
    for h in hits:
        header = f"[{h['chunk_id']}] {h['scheme_name']} ({', '.join(h['sections'])})"
        text = h["text"].strip()
        room = max_chars - used - len(header) - 4
        if room < 80:
            break
        if len(text) > room:
            text = text[:room].rsplit(" ", 1)[0] + " ..."
        block = f"{header}\n{text}"
        blocks.append(block)
        used += len(block) + 2
    return "\n\n".join(blocks)
