"""
chunker.py  -  cuts each scheme's text into chunks that are good to embed and to quote.

A scheme has five text sections:  Overview (details + tags), Benefits, Eligibility, How to apply,
Documents required. Two strategies (CHUNK_STRATEGY in .env):

  merged   (default)  consecutive sections of one scheme are packed into chunks of up to
                      CHUNK_MAX_TOKENS tokens, each section labelled inside the chunk
                      ("Eligibility: ..."). This gives the plan's 300-500 token chunks even though
                      most sections are short. A section longer than the limit is split at sentence
                      boundaries into pieces that overlap by CHUNK_OVERLAP_TOKENS.
  section             one chunk per section (shorter chunks, one clear topic each).

Rules that hold for both:
  * a chunk never mixes two schemes
  * no text is lost: every word of every section ends up in at least one chunk
  * each chunk (plus its header) fits the embedding model's input limit
  * chunk ids are stable: S012-c0, S012-c1, ...

Every chunk carries metadata (scheme id, name, sections, state, level, category, official link, ...)
so searches can be filtered, and every answer can point back to its source.

The text that is EMBEDDED is  header + chunk text , where the header repeats the scheme name,
category and state ("Contextual chunk header"), so a chunk about "income limit" still knows which
scheme it belongs to. The text that is STORED and quoted is the plain chunk text.
"""
import re
from dataclasses import dataclass, field

# (section name, column in schemes_clean.csv, label used inside merged chunks)
SECTIONS = [
    ("overview", "details", "Overview"),
    ("benefits", "benefits", "Benefits"),
    ("eligibility", "eligibility", "Eligibility"),
    ("application", "application", "How to apply"),
    ("documents", "documents", "Documents required"),
]
SECTION_NAMES = [s[0] for s in SECTIONS]
STRATEGIES = ("merged", "section")

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'\u201c\u2018])")


# ---------------------------------------------------------------------------------------
# splitting text
# ---------------------------------------------------------------------------------------
def split_sentences(text: str) -> list:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_BREAK.split(text) if s.strip()]


def split_by_words(text: str, count, max_tokens: int, overlap_tokens: int) -> list:
    """Fallback for text with no sentence breaks (for example a long list of documents): word windows."""
    words = text.split()
    if not words:
        return []
    per_word = max(count(text) / len(words), 0.1)
    size = max(1, int(max_tokens / per_word))
    overlap_words = min(int(overlap_tokens / per_word), size - 1)
    pieces, i = [], 0
    while i < len(words):
        j = min(len(words), i + size)
        piece = " ".join(words[i:j])
        while count(piece) > max_tokens and j - i > 1:           # shrink until it fits
            j -= max(1, (j - i) // 10)
            piece = " ".join(words[i:j])
        pieces.append(piece)
        if j >= len(words):
            break
        i = max(i + 1, j - overlap_words)
    return pieces


def split_text(text: str, count, max_tokens: int, overlap_tokens: int) -> list:
    """
    Pack sentences into pieces of at most `max_tokens`. When a new piece starts, the last sentences of the
    previous piece (up to `overlap_tokens`) are repeated at its start, so a requirement that falls on a
    boundary is whole in at least one piece.
    """
    sentences = split_sentences(text)
    pieces, current, cur = [], [], 0
    for s in sentences:
        t = count(s)
        if t > max_tokens:                                       # one huge "sentence"
            if current:
                pieces.append(" ".join(current))
            current, cur = [], 0
            pieces.extend(split_by_words(s, count, max_tokens, overlap_tokens))
            continue
        if current and cur + t > max_tokens:
            pieces.append(" ".join(current))
            tail, tail_tokens = [], 0
            for prev in reversed(current):
                pt = count(prev)
                if tail_tokens + pt > overlap_tokens:
                    break
                tail.insert(0, prev)
                tail_tokens += pt
            while tail and tail_tokens + t > max_tokens:
                tail_tokens -= count(tail.pop(0))
            current, cur = tail, tail_tokens
        current.append(s)
        cur += t
    if current:
        pieces.append(" ".join(current))
    # safety net: the additive estimate can be slightly off for real tokenizers
    final = []
    for p in pieces:
        final.extend(split_by_words(p, count, max_tokens, overlap_tokens) if count(p) > max_tokens else [p])
    return final


# ---------------------------------------------------------------------------------------
# chunks
# ---------------------------------------------------------------------------------------
@dataclass
class Chunk:
    chunk_id: str
    scheme_id: str
    scheme_name: str
    sections: list
    index: int
    text: str
    tokens: int
    header: str
    header_tokens: int
    metadata: dict = field(default_factory=dict)

    @property
    def embed_text(self) -> str:
        return f"{self.header}\n{self.text}"


@dataclass
class _Unit:
    section: str
    text: str
    tokens: int
    piece: int


def category_flag(category: str) -> str:
    """'Agriculture,Rural & Environment' -> 'cat_agriculture_rural_environment' (a boolean metadata key)."""
    return "cat_" + re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_")


def header_for(scheme: dict) -> str:
    where = ("Central scheme" if str(scheme.get("level", "")).strip().lower() == "central"
             else f"State: {scheme.get('state', '')}")
    return f"{scheme['scheme_name']}. Category: {scheme.get('primary_category', '')}. {where}."


def _section_text(scheme: dict, name: str, column: str) -> str:
    text = re.sub(r"\s+", " ", str(scheme.get(column, "") or "")).strip()
    if name == "overview":
        tags = re.sub(r"\s+", " ", str(scheme.get("tags", "") or "")).strip()
        if tags:
            text = f"{text} Tags: {tags}".strip()
    return text


def _metadata(scheme: dict, sections: list, index: int, tokens: int) -> dict:
    meta = {
        "scheme_id": scheme["scheme_id"],
        "scheme_name": scheme["scheme_name"],
        "section": sections[0],
        "sections": "+".join(sections),
        "chunk_index": index,
        "n_chunks": 0,                                    # filled in once all chunks of the scheme are known
        "state": str(scheme.get("state", "")),
        "level": str(scheme.get("level", "")),
        "primary_category": str(scheme.get("primary_category", "")),
        "categories": str(scheme.get("categories", "")),
        "official_link": str(scheme.get("official_link", "")),
        "tokens": tokens,
    }
    for s in sections:
        meta[f"sec_{s}"] = True
    for c in [x.strip() for x in str(scheme.get("categories", "")).split("|") if x.strip()]:
        meta[category_flag(c)] = True
    return meta


def chunk_scheme(scheme: dict, count, max_tokens: int, overlap_tokens: int, strategy: str = "merged") -> list:
    """All chunks of one scheme. `count` is a function text -> number of tokens."""
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    if max_tokens < 100 or not (0 <= overlap_tokens <= max_tokens // 2):
        raise ValueError("CHUNK_MAX_TOKENS must be >= 100 and CHUNK_OVERLAP_TOKENS between 0 and half of it")
    labelled = strategy == "merged"

    units = []
    for name, column, label in SECTIONS:
        raw = _section_text(scheme, name, column)
        if not raw:
            continue
        label_cost = count(f"{label}: ") if labelled else 0
        for i, piece in enumerate(split_text(raw, count, max_tokens - label_cost, overlap_tokens)):
            text = f"{label}: {piece}" if labelled else piece
            units.append(_Unit(name, text, count(text), i))

    groups, current = [], []
    for u in units:
        if strategy == "section":
            groups.append([u])
            continue
        if current and count("\n".join(x.text for x in current + [u])) > max_tokens:
            groups.append(current)
            current = []
        current.append(u)
    if current:
        groups.append(current)

    header = header_for(scheme)
    header_tokens = count(header)
    chunks = []
    for i, group in enumerate(groups):
        text = "\n".join(u.text for u in group)
        sections = list(dict.fromkeys(u.section for u in group))
        meta = _metadata(scheme, sections, i, count(text))
        chunks.append(Chunk(chunk_id=f"{scheme['scheme_id']}-c{i}", scheme_id=scheme["scheme_id"],
                            scheme_name=scheme["scheme_name"], sections=sections, index=i, text=text,
                            tokens=meta["tokens"], header=header, header_tokens=header_tokens, metadata=meta))
    for c in chunks:
        c.metadata["n_chunks"] = len(chunks)
    return chunks


def summarize(chunks: list) -> dict:
    if not chunks:
        return {"chunks": 0}
    tokens = sorted(c.tokens for c in chunks)
    return {"chunks": len(chunks), "schemes": len({c.scheme_id for c in chunks}),
            "tokens_min": tokens[0], "tokens_median": tokens[len(tokens) // 2], "tokens_max": tokens[-1],
            "tokens_mean": round(sum(tokens) / len(tokens), 1),
            "multi_section_chunks": sum(len(c.sections) > 1 for c in chunks)}
