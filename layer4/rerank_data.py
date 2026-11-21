"""
rerank_data.py  -  the building blocks for the reranker's training data. Plain Python, no PyTorch, no LLM.

The data answers one question: "given a person's question, can the model tell the chunk that answers it from chunks
that do not?"

  split_schemes     schemes are split into train / validation / test BY SCHEME (never by question), so the model is
                    always tested on schemes it has never seen
  clean_questions   keeps LLM-written questions that are usable and do not give the scheme's name away
  build_group       one question + its source chunk (the POSITIVE) + negatives from OTHER schemes:
                      hard negatives   chunks a first-stage search ranks high but that belong to another scheme
                      random negatives chunks picked at random from other schemes
                    chunks of the SAME scheme are never used as negatives (they may also answer the question)
  build_eval_item   one question + the candidates the first-stage search returns over the WHOLE index, with the
                    right scheme marked; this is exactly the situation the reranker faces in production
"""
import json
import random
import re

GENERIC_NAME_WORDS = {
    "scheme", "yojana", "yojna", "programme", "program", "mission", "fund", "assistance", "benefit", "benefits",
    "scholarship", "scholarships", "pension", "fellowship", "award", "awards", "welfare", "development", "national",
    "state", "government", "central", "support", "financial", "relief", "training", "education", "scholar",
    "women", "woman", "girl", "child", "children", "farmer", "farmers", "student", "students", "rural", "urban",
}
STYLES = ("situation", "short", "formal")


# ---------------------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------------------
def read_jsonl(path) -> list:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------------------
# splitting
# ---------------------------------------------------------------------------------------
def split_schemes(scheme_ids, val_fraction: float, test_fraction: float, seed: int) -> dict:
    """{'train': [...], 'val': [...], 'test': [...]}: disjoint, deterministic, each non-empty when possible."""
    ids = sorted(set(scheme_ids))
    if len(ids) < 3:
        raise ValueError("need at least 3 schemes to make train / validation / test splits")
    if not (0 < val_fraction < 0.5 and 0 < test_fraction < 0.5 and val_fraction + test_fraction < 0.8):
        raise ValueError("VAL_FRACTION and TEST_FRACTION must each be between 0 and 0.5 (and sum to under 0.8)")
    shuffled = ids[:]
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, round(len(ids) * val_fraction))
    n_test = max(1, round(len(ids) * test_fraction))
    return {"val": sorted(shuffled[:n_val]), "test": sorted(shuffled[n_val:n_val + n_test]),
            "train": sorted(shuffled[n_val + n_test:])}


def split_of(scheme_id, splits) -> str:
    for name in ("train", "val", "test"):
        if scheme_id in splits[name]:
            return name
    raise KeyError(scheme_id)


# ---------------------------------------------------------------------------------------
# questions
# ---------------------------------------------------------------------------------------
def _norm(text) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def leaks_name(question: str, scheme_name: str) -> bool:
    """
    True if the question gives the scheme away: it contains the full name, or two of the name's distinctive capitalised
    words (like "Ladli" and "Laxmi"), or the name's acronym (like "WISE"). Real users rarely know the name, and a
    model that learns to match names would not generalise to them.
    """
    q = _norm(question)
    name = _norm(scheme_name)
    if name and f" {name} " in f" {q} ":
        return True
    q_words = set(q.split())
    original_words = re.findall(r"[A-Za-z]+", scheme_name)
    proper = [w.lower() for w in original_words if w[0].isupper() and len(w) >= 4 and w.lower() not in GENERIC_NAME_WORDS]
    if len(proper) >= 2 and sum(w in q_words for w in proper) >= 2:
        return True
    acronyms = [w.lower() for w in original_words if w.isupper() and len(w) >= 3]
    return any(a in q_words for a in acronyms)


def clean_questions(raw_items, scheme_name, max_n: int = 5):
    """
    Keep usable questions. `raw_items` = [{'text', 'style'}]. Returns (kept, dropped) where dropped counts the reasons.
    """
    kept, seen = [], set()
    dropped = {"too_short_or_long": 0, "leaks_name": 0, "duplicate": 0}
    for item in raw_items or []:
        text = re.sub(r"\s+", " ", str(item.get("text", ""))).strip()
        if len(text) < 8 or len(text) > 400 or len(text.split()) < 2:
            dropped["too_short_or_long"] += 1
        elif leaks_name(text, scheme_name):
            dropped["leaks_name"] += 1
        elif _norm(text) in seen:
            dropped["duplicate"] += 1
        else:
            seen.add(_norm(text))
            style = item.get("style") if item.get("style") in STYLES else "other"
            kept.append({"text": text, "style": style})
    return kept[:max_n], dropped


# ---------------------------------------------------------------------------------------
# training groups and evaluation items
# ---------------------------------------------------------------------------------------
def build_group(question: dict, pool: dict, search_fn, rng: random.Random, n_neg: int, n_hard: int):
    """
    question  {'qid','question','chunk_id','scheme_id','split'}
    pool      {chunk_id: {'chunk_id','scheme_id','text'}}: the chunks this group may draw from (ONE split only)
    search_fn(query_text, k) -> list of {'chunk_id','scheme_id','text'}, best first, limited to the same pool
    Returns a group dict, or None if no negatives could be found.
    """
    positive = pool[question["chunk_id"]]
    others = {cid: c for cid, c in pool.items() if c["scheme_id"] != question["scheme_id"]}
    negatives, used = [], set()
    for hit in search_fn(question["question"], k=max(30, 3 * n_hard)):
        if len(negatives) >= n_hard:
            break
        if hit["chunk_id"] in others and hit["chunk_id"] not in used:
            used.add(hit["chunk_id"])
            negatives.append({"chunk_id": hit["chunk_id"], "text": others[hit["chunk_id"]]["text"], "kind": "hard"})
    remaining = sorted(cid for cid in others if cid not in used)
    rng.shuffle(remaining)
    for cid in remaining[:max(0, n_neg - len(negatives))]:
        negatives.append({"chunk_id": cid, "text": others[cid]["text"], "kind": "random"})
    if not negatives:
        return None
    return {"qid": question["qid"], "query": question["question"], "scheme_id": question["scheme_id"],
            "split": question["split"], "positive": {"chunk_id": positive["chunk_id"], "text": positive["text"]},
            "negatives": negatives}


def build_eval_item(question: dict, search_fn, n_candidates: int) -> dict:
    """The first-stage candidates over the whole index for this question, in first-stage order."""
    hits = search_fn(question["question"], k=n_candidates)
    return {"qid": question["qid"], "query": question["question"], "relevant_scheme_id": question["scheme_id"],
            "split": question["split"],
            "candidates": [{"chunk_id": h["chunk_id"], "scheme_id": h["scheme_id"], "text": h["text"],
                            "first_stage_rank": i, "first_stage_score": h.get("score")} for i, h in enumerate(hits, start=1)]}


def audit_groups(groups: list, splits: dict) -> list:
    """Problems that would make training data invalid (used by the tests and by make_training_data.py)."""
    problems = []
    for g in groups:
        split = g["split"]
        pool_schemes = set(splits[split])
        if g["scheme_id"] not in pool_schemes:
            problems.append(f"{g['qid']}: positive scheme is not in the {split} split")
        for n in g["negatives"]:
            sid = n["chunk_id"].split("-c")[0]
            if sid == g["scheme_id"]:
                problems.append(f"{g['qid']}: a negative comes from the question's own scheme")
            if sid not in pool_schemes:
                problems.append(f"{g['qid']}: negative {n['chunk_id']} is outside the {split} split (data leakage)")
        ids = [n["chunk_id"] for n in g["negatives"]]
        if len(ids) != len(set(ids)):
            problems.append(f"{g['qid']}: duplicate negatives")
        if g["positive"]["chunk_id"] in ids:
            problems.append(f"{g['qid']}: the positive is also a negative")
    return problems
