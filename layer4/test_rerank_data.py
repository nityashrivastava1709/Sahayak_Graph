"""
test_rerank_data.py  -  tests for the training-data logic and the promotion decision. No model, LLM or index needed.

Run:  python test_rerank_data.py
These tests guard the things that would silently ruin a reranker: test schemes leaking into training,
negatives taken from the question's own scheme, questions that give the scheme's name away.
"""
import random
import sys
import tempfile
from pathlib import Path

import rerank_data as rd

passed = failed = 0


def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
    else:
        failed += 1
        print(f"FAIL  {name}   {detail}")


# ---- splitting by scheme
ids = [f"S{i:03d}" for i in range(1, 91)]
sp = rd.split_schemes(ids, 0.15, 0.15, 42)
check("train / val / test are disjoint and cover every scheme",
      set(sp["train"]) | set(sp["val"]) | set(sp["test"]) == set(ids) and not (set(sp["train"]) & set(sp["val"]))
      and not (set(sp["train"]) & set(sp["test"])) and not (set(sp["val"]) & set(sp["test"])))
check("sizes follow the fractions", len(sp["val"]) == 14 and len(sp["test"]) == 14 and len(sp["train"]) == 62, str({k: len(v) for k, v in sp.items()}))
check("deterministic for the same seed, different for another", rd.split_schemes(ids, 0.15, 0.15, 42) == sp and rd.split_schemes(ids, 0.15, 0.15, 7) != sp)
check("input order does not matter", rd.split_schemes(list(reversed(ids)), 0.15, 0.15, 42) == sp)
small = rd.split_schemes(["S001", "S002", "S003", "S004"], 0.15, 0.15, 1)
check("tiny sets still get a validation and a test scheme", len(small["val"]) >= 1 and len(small["test"]) >= 1 and len(small["train"]) >= 1)
for label, args in [("fewer than 3 schemes", (["S001", "S002"], 0.15, 0.15, 1)), ("fractions too big", (ids, 0.5, 0.5, 1)), ("zero fraction", (ids, 0.0, 0.15, 1))]:
    try:
        rd.split_schemes(*args)
        check(f"rejects {label}", False)
    except ValueError:
        check(f"rejects {label}", True)
check("split_of finds the right split", rd.split_of(sp["val"][0], sp) == "val" and rd.split_of(sp["train"][0], sp) == "train")

# ---- questions that give the scheme away
leak = [("what do I get under Ladli Laxmi Yojana", "Ladli Laxmi Yojana", True), ("can I apply for WISE", "WISE Fellowship for Ph.D.", True),
        ("Kaushal Vikas training for my husband", "Kaushal Vikas Yojana", True),
        ("my daughter was born in 2010, help with her studies", "Ladli Laxmi Yojana", False), ("I am a widow, can I get a pension?", "Widow Pension Scheme", False),
        ("kaushal training for a construction worker", "Kaushal Vikas Yojana", False), ("fellowship for women who want a PhD", "WISE Fellowship for Ph.D.", False)]
for q, name, want in leak:
    check(f"leak check: '{q[:40]}...' vs '{name}' -> {want}", rd.leaks_name(q, name) == want)

raw = [{"text": "my husband died, can I get a pension?", "style": "situation"}, {"text": "widow pension", "style": "short"},
       {"text": "My husband died, can I get a pension?", "style": "formal"},                  # duplicate (case)
       {"text": "hi", "style": "short"}, {"text": "x" * 500, "style": "formal"},
       {"text": "Tell me about Ladli Laxmi Yojana benefits", "style": "formal"}, {"text": "who can apply for support for girls", "style": "weird-style"}]
kept, dropped = rd.clean_questions(raw, "Ladli Laxmi Yojana", max_n=5)
check("clean_questions keeps usable ones", [k["text"] for k in kept] == ["my husband died, can I get a pension?", "widow pension", "who can apply for support for girls"], str(kept))
check("it counts why questions were dropped", dropped == {"too_short_or_long": 2, "leaks_name": 1, "duplicate": 1}, str(dropped))
check("an unknown style becomes 'other'", kept[-1]["style"] == "other" and kept[0]["style"] == "situation")
check("max_n limits the result", len(rd.clean_questions(raw, "Zzz", max_n=2)[0]) == 2)
check("nothing in, nothing out", rd.clean_questions([], "x")[0] == [] and rd.clean_questions(None, "x")[0] == [])

# ---- groups: positives, hard and random negatives
pool = {f"S{s:03d}-c{c}": {"chunk_id": f"S{s:03d}-c{c}", "scheme_id": f"S{s:03d}", "text": f"text of scheme {s} part {c}"}
        for s in range(1, 9) for c in range(2)}
question = {"qid": "S001-c0-q0", "question": "a question", "chunk_id": "S001-c0", "scheme_id": "S001", "split": "train"}
search = lambda text, k: [{"chunk_id": "S001-c1", "scheme_id": "S001", "text": "own scheme"}, {"chunk_id": "S005-c0", "scheme_id": "S005", "text": ""},
                          {"chunk_id": "S003-c1", "scheme_id": "S003", "text": ""}, {"chunk_id": "S099-c0", "scheme_id": "S099", "text": "not in the pool"},
                          {"chunk_id": "S005-c0", "scheme_id": "S005", "text": ""}, {"chunk_id": "S004-c0", "scheme_id": "S004", "text": ""}]
g = rd.build_group(question, pool, search, random.Random(1), n_neg=7, n_hard=3)
neg_ids = [n["chunk_id"] for n in g["negatives"]]
check("the positive is the question's source chunk", g["positive"] == {"chunk_id": "S001-c0", "text": pool["S001-c0"]["text"]})
check("no negative comes from the question's own scheme (not even S001-c1)", not any(n.startswith("S001") for n in neg_ids), str(neg_ids))
check("hard negatives come first, in search order, without duplicates", neg_ids[:3] == ["S005-c0", "S003-c1", "S004-c0"] and all(n["kind"] == "hard" for n in g["negatives"][:3]), str(neg_ids))
check("chunks outside the pool are never used", "S099-c0" not in neg_ids)
check("random negatives fill up to n_neg", len(neg_ids) == 7 and all(n["kind"] == "random" for n in g["negatives"][3:]))
check("no duplicate negatives and the positive is not among them", len(set(neg_ids)) == len(neg_ids) and "S001-c0" not in neg_ids)
check("negative texts come from the pool", all(n["text"] == pool[n["chunk_id"]]["text"] for n in g["negatives"]))
check("same random seed, same group", rd.build_group(question, pool, search, random.Random(1), 7, 3) == g)
check("a different seed changes only the random negatives", [n["chunk_id"] for n in rd.build_group(question, pool, search, random.Random(2), 7, 3)["negatives"]][:3] == neg_ids[:3])
tiny_pool = {k: v for k, v in pool.items() if v["scheme_id"] == "S001"}
check("no other scheme in the pool -> no group", rd.build_group(question, tiny_pool, lambda t, k: [], random.Random(1), 7, 3) is None)
few = {k: v for k, v in pool.items() if v["scheme_id"] in ("S001", "S002")}
check("fewer possible negatives than asked -> uses what exists", len(rd.build_group(question, few, lambda t, k: [], random.Random(1), 7, 3)["negatives"]) == 2)

# ---- the leakage audit catches planted mistakes
splits = {"train": ["S001", "S002", "S003"], "val": ["S004"], "test": ["S005"]}
good = {"qid": "q1", "scheme_id": "S001", "split": "train", "positive": {"chunk_id": "S001-c0", "text": ""},
        "negatives": [{"chunk_id": "S002-c0", "text": "", "kind": "hard"}, {"chunk_id": "S003-c1", "text": "", "kind": "random"}]}
check("a clean group passes the audit", rd.audit_groups([good], splits) == [])
own = dict(good, negatives=[{"chunk_id": "S001-c1", "text": "", "kind": "hard"}])
leaky = dict(good, negatives=[{"chunk_id": "S005-c0", "text": "", "kind": "hard"}])
dup = dict(good, negatives=good["negatives"] + [good["negatives"][0]])
pos_neg = dict(good, negatives=[{"chunk_id": "S001-c0", "text": "", "kind": "hard"}])
wrong_split = dict(good, scheme_id="S004")
check("audit catches a negative from the question's own scheme", any("own scheme" in p for p in rd.audit_groups([own], splits)))
check("audit catches a test-scheme chunk inside a training group", any("leakage" in p for p in rd.audit_groups([leaky], splits)))
check("audit catches duplicate negatives", any("duplicate" in p for p in rd.audit_groups([dup], splits)))
check("audit catches the positive used as a negative", any("also a negative" in p for p in rd.audit_groups([pos_neg], splits)))
check("audit catches a positive from another split", any("not in the train split" in p for p in rd.audit_groups([wrong_split], splits)))

# ---- evaluation items
ev = rd.build_eval_item(question, lambda t, k: [{"chunk_id": "S002-c0", "scheme_id": "S002", "text": "a", "score": 0.9}, {"chunk_id": "S001-c0", "scheme_id": "S001", "text": "b", "score": 0.5}][:k], 2)
check("eval item keeps first-stage order and ranks", [c["first_stage_rank"] for c in ev["candidates"]] == [1, 2] and ev["candidates"][0]["chunk_id"] == "S002-c0")
check("eval item names the right scheme", ev["relevant_scheme_id"] == "S001" and ev["candidates"][1]["first_stage_score"] == 0.5)

# ---- files
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    path = Path(tmp) / "a" / "x.jsonl"
    rows = [{"q": "विधवा पेंशन ₹5,000", "n": 1}, {"q": "line\nbreak", "n": 2}]
    rd.write_jsonl(path, rows)
    check("jsonl round trip keeps unicode and newlines", rd.read_jsonl(path) == rows)
    check("a missing file reads as empty", rd.read_jsonl(Path(tmp) / "nope.jsonl") == [])

# ---- the promotion decision (evaluate_reranker.decide_active)
from evaluate_reranker import decide_active
mk = lambda v, t: {"val": {"mrr": v}, "test": {"mrr": t}}
d = decide_active(mk(0.5, 0.5), mk(0.6, 0.6), mk(0.7, 0.7))
check("fine-tuned wins on val and test -> active, and recommended when it beats first-stage", d["active"] == "finetuned" and d["rerank_recommended"] is True)
d = decide_active(mk(0.5, 0.5), mk(0.6, 0.6), mk(0.7, 0.55))
check("fine-tuned loses on test -> keep pretrained", d["active"] == "pretrained")
d = decide_active(mk(0.5, 0.5), mk(0.6, 0.6), mk(0.55, 0.7))
check("fine-tuned loses on validation -> keep pretrained", d["active"] == "pretrained")
d = decide_active(mk(0.5, 0.5), mk(0.6, 0.6), mk(0.6, 0.6))
check("a tie is not enough to promote", d["active"] == "pretrained")
d = decide_active(mk(0.8, 0.8), mk(0.2, 0.2), mk(0.3, 0.3))
check("rerank NOT recommended when the first-stage order is better", d["active"] == "finetuned" and d["rerank_recommended"] is False and "skip" in d["reason"][-1])
d = decide_active(mk(0.5, 0.5), mk(0.6, 0.6), None)
check("no fine-tuned model -> pretrained", d["active"] == "pretrained" and d["rerank_recommended"] is True)
check("--force overrides", decide_active(mk(0.5, 0.5), mk(0.6, 0.6), mk(0.7, 0.7), force="pretrained")["active"] == "pretrained")
try:
    decide_active(mk(0.5, 0.5), mk(0.6, 0.6), None, force="finetuned")
    check("--force finetuned without a model is an error", False)
except ValueError:
    check("--force finetuned without a model is an error", True)
check("recommendation compares test MRR with first-stage test MRR (not validation)", decide_active(mk(0.9, 0.5), mk(0.6, 0.6), None)["rerank_recommended"] is True)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
