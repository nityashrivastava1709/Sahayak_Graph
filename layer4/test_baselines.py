"""
test_baselines.py  -  proves both baselines take a question and return schemes in the SAME output format.

Run (needs your Layer 1 data, the Layer 3 index and the Layer 3 folder; NO Gemini key or internet):  python test_baselines.py
The Gemini call is replaced by a scripted fake that behaves well, behaves badly (invents a scheme, repeats one, names one it
was never shown), or fails, so every branch of the vector-RAG baseline is exercised.
"""
import json
import re
import sys
import tempfile
import time
from pathlib import Path

from google.genai import types

import l4_common
import l4_config as config
import llm_client
from keyword_baseline import KeywordBaseline
from vector_rag_baseline import VectorRAGBaseline

time.sleep = lambda s: None
config.GEMINI_API_KEY = "fake-key-for-test"
tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
config.LLM_CACHE_DIR = Path(tmp.name) / "cache"
config.LLM_FAILURE_LOG = Path(tmp.name) / "failures.log"
passed = failed = 0


def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
    else:
        failed += 1
        print(f"FAIL  {name}   {detail}")


def R(text):
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=[types.Part(text=text)]), finish_reason=types.FinishReason.STOP)])


class FakeGemini:
    """mode: normal | bad | fail | forbidden"""

    def __init__(self, mode="normal"):
        self.mode, self.calls, self.last_user, self.last_system = mode, 0, "", ""

        class Models:
            pass
        self.models = Models()
        self.models.generate_content = self._generate

    def _generate(self, model, contents, config):
        self.calls += 1
        self.last_user, self.last_system = contents, config.system_instruction
        if self.mode == "forbidden":
            raise AssertionError("the LLM must not be called in this mode")
        if self.mode == "fail":
            return R("this is not json")
        ids = list(dict.fromkeys(re.findall(r"\[(S\d{3})-c\d+\]", contents)))
        k = int(re.search(r"at most (\d+) schemes", contents).group(1))
        pick = lambda sid, why: {"scheme_id": sid, "reason": why, "required_documents": ["Aadhaar Card", " "], "application_steps": ["Apply online"]}
        if self.mode == "normal":
            items = [pick(s, "fits the question") for s in ids[:min(3, k)]]
        else:
            unseen = next(f"S{n:03d}" for n in range(1, 91) if f"S{n:03d}" not in ids)
            items = [pick(ids[0].lower(), "lower-case id"), pick("S999", "invented"), pick(unseen, "never shown"), pick(ids[0], "duplicate")] + [pick(s, "x") for s in ids[1:4]]
        return R(json.dumps({"answer": "You may qualify for these schemes.", "schemes": items}))


def install(mode):
    fake = FakeGemini(mode)
    llm_client._client = fake
    return fake


kb = KeywordBaseline()
valid_ids = {s["scheme_id"] for s in kb.schemes}
rag = VectorRAGBaseline()
states = sorted({s["state"] for s in kb.schemes if s["state"] != "Central"})
queries = ["scholarship for girls", "pension for widows", "fellowship for women scientists to pursue a Ph.D.", "farmer seeds subsidy",
           "training for construction workers", "financial help for pregnant women", "education loan for students",
           "support for fishermen in lean season", "disabled persons assistance", "income certificate documents needed",
           "I am a farmer with 2 acres, what subsidy can I get?", "zzzqqq nothing matches this"]

# ---- BOTH baselines follow the same format (the Layer 4 "done when")
install("normal")
formats_ok, same_keys, serialisable = True, True, True
for q in queries:
    a, b = kb.run(q, k=5), rag.run(q, k=5)
    for res in (a, b):
        problems = l4_common.validate_result(res, valid_ids)
        if problems:
            formats_ok = False
            print("   format problem:", res["system"], q, problems[:2])
        try:
            json.dumps(res)
        except Exception:
            serialisable = False
    for sa, sb in zip(a["schemes"], b["schemes"]):
        same_keys &= set(sa) == set(sb)
    same_keys &= set(a) == set(b) and set(a["meta"]) >= {"latency_s"} and set(b["meta"]) >= {"latency_s"}
check(f"both baselines follow the common format on {len(queries)} questions", formats_ok)
check("both return exactly the same set of keys", same_keys)
check("results are JSON-serialisable", serialisable)
check("the two systems are named differently", kb.run("pension").get("system") == "keyword" and rag.run("pension")["system"] == "vector_rag")
check("a question nothing matches gives an empty valid keyword result", kb.run("zzzqqq xxyyzz")["schemes"] == [])

# ---- vector RAG, well-behaved LLM
fake = install("normal")
r = rag.run("what monthly pension can an old widow get", k=5)          # a new question, so the LLM is really called
check("at most k schemes, ranks 1..n", 0 < len(r["schemes"]) <= 5 and [s["rank"] for s in r["schemes"]] == list(range(1, len(r["schemes"]) + 1)))
check("every scheme was retrieved (grounded), with its retrieval score and evidence", all(s["retrieved"] and s["score"] is not None and s["evidence"] for s in r["schemes"]))
check("evidence chunk ids belong to their scheme", all(e["chunk_id"].startswith(s["scheme_id"] + "-c") for s in r["schemes"] for e in s["evidence"]))
check("documents and steps are cleaned lists of strings (blank entries removed)", all(s["documents"] == ["Aadhaar Card"] and s["steps"] == ["Apply online"] for s in r["schemes"]))
check("reason and answer come from the LLM", all(s["reason"] == "fits the question" for s in r["schemes"]) and r["answer"] == "You may qualify for these schemes.")
check("meta records what happened", r["meta"]["llm_used"] and not r["meta"]["llm_failed"] and r["meta"]["n_context_chunks"] > 0 and r["meta"]["model"] == config.GEMINI_MODEL)
check("the prompt contains the question, passages with chunk ids, and the limit", "what monthly pension can an old widow get" in fake.last_user and "PASSAGES:" in fake.last_user
      and re.search(r"\[S\d{3}-c\d+\]", fake.last_user) and "at most 5 schemes" in fake.last_user)
check("the system prompt tells the LLM to use only the passages", "Use ONLY the passages" in fake.last_system and "Never guess" in fake.last_system)
calls_before = fake.calls
rag.run("what monthly pension can an old widow get", k=5)
check("an identical question is answered from the cache (no second LLM call)", fake.calls == calls_before)
check("k limits the schemes the LLM may return", len(rag.run("scholarship", k=2)["schemes"]) <= 2)

# ---- vector RAG, badly-behaved LLM: nothing is hidden
install("bad")
r = rag.run("scholarship for students", k=10)
ids = [s["scheme_id"] for s in r["schemes"]]
check("an invented scheme id is reported in invalid_ids, not silently dropped", r["invalid_ids"] == ["S999"], str(r["invalid_ids"]))
check("a real scheme that was never shown is kept but marked retrieved = False, with no score",
      any(not s["retrieved"] and s["score"] is None and s["evidence"] == [] for s in r["schemes"]))
check("ids are normalised (lower-case 's012' -> 'S012') and duplicates removed", len(ids) == len(set(ids)) and all(re.fullmatch(r"S\d{3}", i) for i in ids))
check("the result is still in the common format", l4_common.validate_result(r, valid_ids) == [], str(l4_common.validate_result(r, valid_ids)))

# ---- vector RAG, failing LLM: never crashes, never hides the failure
install("fail")
r = rag.run("scholarship for students", k=4)
check("when the LLM fails the retrieval order is returned and flagged", r["meta"]["llm_failed"] is True and len(r["schemes"]) > 0 and r["answer"] is None)
check("...without invented documents", all(s["documents"] is None and s["steps"] is None for s in r["schemes"]))
check("...in the common format", l4_common.validate_result(r, valid_ids) == [])

# ---- no LLM at all
install("forbidden")
r = rag.run("scholarship for students", k=4, use_llm=False)
check("use_llm=False never calls the LLM and returns retrieval order", r["meta"]["llm_used"] is False and len(r["schemes"]) == 4)
check("retrieval-only scores are sorted best first", [s["score"] for s in r["schemes"]] == sorted((s["score"] for s in r["schemes"]), reverse=True))

# ---- state filter and profiles
install("normal")
st = states[0]
check(f"the state filter keeps {st} and Central schemes (keyword)", all(s["state"] in (st, "Central") for s in kb.run("scheme for farmers and students", k=30, state=st)["schemes"]))
check(f"the state filter keeps {st} and Central schemes (vector RAG)", all(s["state"] in (st, "Central") for s in rag.run("scheme for farmers and students", k=10, state=st)["schemes"]))
check("a state code works for both", kb.run("farmers", state="MP")["meta"]["state_filter"] == "Madhya Pradesh" and rag.run("farmers", state="MP")["meta"]["state_filter"] == "Madhya Pradesh")
prof = {"age": 34, "gender": "female", "state": "MP", "occupation": "farmer", "income": "2,50,000", "caste": ["SC"], "owns_land": True, "bpl_status": "no", "land_holding": 3}
q = l4_common.profile_to_query(prof)
check("profile_to_query is deterministic and complete", q == l4_common.profile_to_query(dict(reversed(list(prof.items())))) and
      all(t in q for t in ["I am 34 years old", "I am a woman", "farmer", "Madhya Pradesh", "Rs 250000", "SC category", "own agricultural land",
                           "not below the poverty line", "3 acres", "Which government welfare schemes"]), q)
check("an empty profile still gives a valid question", l4_common.profile_to_query({}).startswith("Which government"))
check("profile_state returns the state as typed", l4_common.profile_state(prof) == "MP" and l4_common.profile_state({}) is None)
check("run_profile works for both baselines and filters by the profile's state",
      all(s["state"] in ("Madhya Pradesh", "Central") for s in kb.run_profile(prof, k=20)["schemes"]) and
      l4_common.validate_result(rag.run_profile(prof, k=5), valid_ids) == [])

# ---- reranker hook (a fake reranker: no PyTorch needed here)
class Reverse:
    model_path = "fake-reverse"

    def rerank(self, query, candidates, text_key="text", top_k=None):
        out = []
        for new_rank, c in enumerate(reversed(candidates), start=1):
            item = dict(c)
            item.update({"rerank_score": 1 - new_rank / 100, "original_rank": candidates.index(c) + 1, "rank": new_rank})
            out.append(item)
        return out


class Broken:
    model_path = "fake-broken"

    def rerank(self, *a, **k):
        raise RuntimeError("model crashed")


plain = rag.run("scholarship for students", k=5, use_llm=False)
reranked = VectorRAGBaseline(reranker=Reverse()).run("scholarship for students", k=5, use_llm=False)
check("with a reranker the system is named vector_rag+rerank and the order changes",
      reranked["system"] == "vector_rag+rerank" and [s["scheme_id"] for s in reranked["schemes"]] != [s["scheme_id"] for s in plain["schemes"]])
check("meta says the reranker was used", reranked["meta"]["rerank"]["used"] is True and reranked["meta"]["method"] == "vector_search+rerank")
fallback = VectorRAGBaseline(reranker=Broken()).run("scholarship for students", k=5, use_llm=False)
check("a crashing reranker falls back to the original order, flagged in meta",
      fallback["meta"]["rerank"]["used"] is False and "crashed" in fallback["meta"]["rerank"]["reason"]
      and [s["scheme_id"] for s in fallback["schemes"]] == [s["scheme_id"] for s in plain["schemes"]])
check("a reranked result is still in the common format", l4_common.validate_result(reranked, valid_ids) == [])

# ---- bad input
for label, call in [("empty query", lambda b: b.run("  ")), ("k = 0", lambda b: b.run("x", k=0)), ("k as text", lambda b: b.run("x", k="3")),
                    ("unknown state", lambda b: b.run("x", state="Atlantis"))]:
    for name, b in (("keyword", kb), ("vector", rag)):
        try:
            call(b)
            check(f"{name}: {label} is rejected", False)
        except ValueError:
            check(f"{name}: {label} is rejected", True)

# ---- the format validator itself catches mistakes
good = rag.run("pension for widows", k=3)
bad = json.loads(json.dumps(good))
bad["schemes"][0]["rank"] = 5
check("validate_result catches a wrong rank", any("rank" in p for p in l4_common.validate_result(bad, valid_ids)))
bad = json.loads(json.dumps(good)); bad["schemes"].append(dict(bad["schemes"][0], rank=len(bad["schemes"]) + 1))
check("validate_result catches a duplicate scheme", any("twice" in p for p in l4_common.validate_result(bad, valid_ids)))
bad = json.loads(json.dumps(good)); bad["schemes"][0]["scheme_id"] = "S999"
check("validate_result catches a scheme that does not exist", any("does not exist" in p for p in l4_common.validate_result(bad, valid_ids)))
bad = json.loads(json.dumps(good)); del bad["meta"]
check("validate_result catches a missing key", any("meta" in p for p in l4_common.validate_result(bad)))
bad = json.loads(json.dumps(good)); bad["schemes"][0]["documents"] = "Aadhaar"
check("validate_result catches documents that are not a list", any("documents" in p for p in l4_common.validate_result(bad)))
check("group_hits_by_scheme groups best-first and caps chunks per scheme",
      [(g["scheme_id"], len(g["chunks"])) for g in l4_common.group_hits_by_scheme(
          [{"scheme_id": "S001", "scheme_name": "a", "chunk_id": "S001-c0"}, {"scheme_id": "S002", "scheme_name": "b", "chunk_id": "S002-c0"},
           {"scheme_id": "S001", "scheme_name": "a", "chunk_id": "S001-c1"}, {"scheme_id": "S001", "scheme_name": "a", "chunk_id": "S001-c2"}], k=5, per_scheme=2)]
      == [("S001", 2), ("S002", 1)])

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
