"""
test_bm25.py  -  tests for the keyword search (BM25) and the keyword baseline.

Run:  python test_bm25.py
The first part needs nothing. The second part needs your Layer 1 data (schemes_clean.csv) and Layer 3 folder.
"""
import math
import sys

import bm25

passed = failed = 0


def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
    else:
        failed += 1
        print(f"FAIL  {name}   {detail}")


# ---- tokenizing and stemming
check("lower-case, stop words removed", bm25.tokenize("The Farmers of India are in the field") == ["farmer", "india", "field"])
check("plurals reduced", [bm25.stem(w) for w in ["schemes", "families", "benefits", "boxes", "churches"]] == ["scheme", "family", "benefit", "box", "church"])
check("words that merely end in s are left alone", [bm25.stem(w) for w in ["class", "status", "analysis"]] == ["class", "status", "analysis"])
check("short words are never cut", [bm25.stem(w) for w in ["gas", "bus", "is"]] == ["gas", "bus", "is"])
check("punctuation and unicode", bm25.tokenize("Ph.D. fellowship, ₹5,000!") == ["ph", "fellowship", "000"])
check("empty text", bm25.tokenize("") == [] and bm25.tokenize(None) == [])

# ---- ranking on a tiny corpus
docs = ["monthly pension for widows", "subsidy on seeds for small farmers", "scholarship for students and pension for retired teachers"]
m = bm25.BM25().fit(docs)
top = m.top("pension widow")
check("the most specific document ranks first", [i for i, _ in top] == [0, 2], str(top))
check("documents with no query word are left out", all(i != 1 for i, _ in top))
check("a word nobody has gives no results", m.top("spaceship") == [])
check("only stop words gives no results", m.top("the of and") == [])
check("empty corpus", bm25.BM25().fit([]).scores("anything") == [] and bm25.BM25().fit([]).top("x") == [])
check("repeating a query word changes nothing (each word counts once)", m.top("pension widow") == m.top("pension pension widow widow widow"))
check("deterministic", m.top("pension widow") == bm25.BM25().fit(docs).top("pension widow"))

# ---- the formula
two = bm25.BM25().fit(["apple banana", "apple cherry cherry"])
idf = math.log(1 + (2 - 2 + 0.5) / (2 + 0.5))
want = idf * 1 * 2.5 / (1 + 1.5 * (1 - 0.75 + 0.75 * 2 / 2.5))
check("matches the BM25 formula worked out by hand", abs(two.scores("apple")[0] - want) < 1e-9, f"{two.scores('apple')[0]} vs {want}")

# ---- the three ideas behind BM25
rare = bm25.BM25().fit(["pension scheme alpha", "pension scheme beta", "pension scheme gamma widow"])
check("rare words count more than common ones", rare.scores("pension widow")[2] > rare.scores("pension widow")[0])
check("idf is higher for rarer words", rare.idf("widow") > rare.idf("pension"))
length = bm25.BM25().fit(["pension", "pension " + " ".join(f"filler{i}" for i in range(40))])
check("a short document beats a long one with the same count", length.scores("pension")[0] > length.scores("pension")[1])
tf = bm25.BM25().fit(["pension pension pension pension", "pension x y z"])
check("repeating a word helps, with diminishing returns", tf.scores("pension")[0] > tf.scores("pension")[1] and
      tf.scores("pension")[0] < 4 * tf.scores("pension")[1])
ties = bm25.BM25().fit(["same words here", "same words here", "same words here"])
check("ties keep the input order", [i for i, _ in ties.top("same words")] == [0, 1, 2])
check("k limits the results", len(m.top("pension", k=1)) == 1)

# ---- the keyword baseline on your real data
try:
    import l4_common
    from keyword_baseline import KeywordBaseline
    kb = KeywordBaseline()
    have_data = True
except Exception as e:                                    # no Layer 1 data / Layer 3 folder: skip this part
    print(f"(skipping the keyword-baseline tests: {str(e)[:100]})")
    have_data = False

if have_data:
    valid_ids = {s["scheme_id"] for s in kb.schemes}
    found = sum(s["scheme_id"] in [x["scheme_id"] for x in kb.run(s["scheme_name"], k=3)["schemes"]] for s in kb.schemes)
    check(f"a scheme's name finds that scheme in the top 3 ({found}/{len(kb.schemes)})", found >= 0.9 * len(kb.schemes))
    r = kb.run("scholarship for students", k=5)
    check("result follows the common format", l4_common.validate_result(r, valid_ids) == [], str(l4_common.validate_result(r, valid_ids)))
    check("k results at most, ranks 1..n, scores sorted", len(r["schemes"]) <= 5 and [s["rank"] for s in r["schemes"]] == list(range(1, len(r["schemes"]) + 1))
          and [s["score"] for s in r["schemes"]] == sorted((s["score"] for s in r["schemes"]), reverse=True))
    check("a keyword search provides no documents or steps", all(s["documents"] is None and s["steps"] is None for s in r["schemes"]))
    check("evidence is a sentence from the scheme", all(s["evidence"] and len(s["evidence"][0]["text"]) > 10 for s in r["schemes"]))
    states = sorted({s["state"] for s in kb.schemes if s["state"] != "Central"})
    st = states[0]
    rs = kb.run("scheme scholarship pension farmers", k=30, state=st)
    check(f"the state filter keeps only {st} and Central schemes", all(s["state"] in (st, "Central") for s in rs["schemes"]) and rs["schemes"])
    rn = kb.run("scheme scholarship pension farmers", k=30, state=st, include_central=False)
    check("include_central=False keeps only that state", all(s["state"] == st for s in rn["schemes"]))
    check("a state code works", kb.run("farmers", state="MP")["meta"]["state_filter"] == "Madhya Pradesh")
    check("nothing matching gives an empty, valid result", l4_common.validate_result(kb.run("zzzqqq xxyyzz"), valid_ids) == [] and kb.run("zzzqqq xxyyzz")["schemes"] == [])
    rp = kb.run_profile({"age": 30, "gender": "female", "state": "Rajasthan", "occupation": "student"}, k=5)
    check("run_profile asks the standard question and filters by state", rp["query"].startswith("I am 30") and all(s["state"] in ("Rajasthan", "Central") for s in rp["schemes"]))
    for label, kwargs in [("empty query", dict(query="  ")), ("k = 0", dict(query="x", k=0)), ("unknown state", dict(query="x", state="Atlantis"))]:
        try:
            kb.run(**kwargs)
            check(f"{label} is rejected", False)
        except ValueError:
            check(f"{label} is rejected", True)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
