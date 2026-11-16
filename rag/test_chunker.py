"""
test_chunker.py  -  unit tests for the chunking logic. No index, model or internet needed.

Run:  python test_chunker.py
"""
import sys

import chunker
from embedders import approx_token_count as count

passed = failed = 0


def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
    else:
        failed += 1
        print(f"FAIL  {name}   {detail}")


def scheme(sid="S001", **kw):
    base = {"scheme_id": sid, "scheme_name": "Test Scheme", "level": "State", "state": "Goa",
            "primary_category": "Education & Learning", "categories": "Education & Learning | Women and Child",
            "official_link": "https://example.org/" + sid.lower(), "details": "A scheme for students. It helps them study.",
            "benefits": "Rs 5000 per year.", "eligibility": "The applicant should be a woman. Age above 18 years.",
            "application": "Apply online on the portal.", "documents": "Aadhaar Card Income Certificate", "tags": "scholarship, girls"}
    base.update(kw)
    return base


def sentences(n, words=14, prefix="Sentence"):
    return " ".join(f"{prefix} {i} " + " ".join(f"w{i}x{j}" for j in range(words)) + "." for i in range(n))


words = lambda s: set(s.replace(".", " ").split())

# ---- sentence splitting
check("splits on sentence ends", chunker.split_sentences("One. Two! Three? Four.") == ["One.", "Two!", "Three?", "Four."])
txt = "Pay Rs. 5.5 lakh now. Next step. Dr. Rao will call."
check("splitting never changes the text (joined pieces equal the original)", " ".join(chunker.split_sentences(txt)) == txt)
check("empty text", chunker.split_sentences("   ") == [] and chunker.split_text("", count, 100, 10) == [])
check("collapses whitespace", chunker.split_sentences("A  b\n\nc.") == ["A b c."])

# ---- split_text
short = "This is short. Very short."
check("short text stays one piece", chunker.split_text(short, count, 200, 20) == [short])
long_text = sentences(60)
pieces = chunker.split_text(long_text, count, 120, 30)
check("long text gives several pieces", len(pieces) > 3, str(len(pieces)))
check("every piece fits the limit", all(count(p) <= 120 for p in pieces), str([count(p) for p in pieces][:8]))
check("no text lost", words(long_text) <= words(" ".join(pieces)))
first_sentences = [chunker.split_sentences(p) for p in pieces]
check("overlap: next piece starts with the end of the previous one",
      all(a[-1] in b for a, b in zip(first_sentences, first_sentences[1:])))
no_overlap = chunker.split_text(long_text, count, 120, 0)
check("overlap 0: pieces do not share sentences",
      all(set(chunker.split_sentences(a)).isdisjoint(chunker.split_sentences(b)) for a, b in zip(no_overlap, no_overlap[1:])))
check("deterministic", chunker.split_text(long_text, count, 120, 30) == pieces)

# ---- a huge 'sentence' with no punctuation (like a long document list)
blob = " ".join(f"Document{i}" for i in range(400))
wp = chunker.split_text(blob, count, 100, 20)
check("unpunctuated text is split by words", len(wp) > 3 and all(count(p) <= 100 for p in wp), str([count(p) for p in wp][:6]))
check("word windows overlap", all(set(a.split()[-5:]) & set(b.split()[:25]) for a, b in zip(wp, wp[1:])))
check("word windows lose nothing", words(blob) <= words(" ".join(wp)))
one_word = chunker.split_by_words("x" * 5000, count, 100, 10)
check("a single endless word cannot loop forever", len(one_word) == 1)

# ---- unicode
hindi = "यह योजना किसानों के लिए है। आवेदक की आय ₹2,50,000 से कम होनी चाहिए।"
check("Devanagari and the rupee sign are handled", chunker.split_text(hindi, count, 200, 20) == [hindi])

# ---- chunk_scheme: merged
c = chunker.chunk_scheme(scheme(), count, 420, 60, "merged")
check("short scheme -> one merged chunk", len(c) == 1, str(len(c)))
check("sections in reading order", c[0].sections == ["overview", "benefits", "eligibility", "application", "documents"])
check("labels inside merged chunk", all(l in c[0].text for l in ["Overview:", "Benefits:", "Eligibility:", "How to apply:", "Documents required:"]))
check("chunk id", c[0].chunk_id == "S001-c0")
check("tags are part of the overview", "Tags: scholarship, girls" in c[0].text)
check("embed text starts with the header", c[0].embed_text.startswith("Test Scheme. Category: Education & Learning. State: Goa."))
check("central header", "Central scheme" in chunker.header_for(scheme(level="Central", state="Central")))

# ---- chunk_scheme: section
s = chunker.chunk_scheme(scheme(), count, 420, 60, "section")
check("section strategy -> one chunk per section", len(s) == 5 and [x.sections for x in s] == [[n] for n in chunker.SECTION_NAMES])
check("section chunks have no labels", all(":" not in x.text.split()[0] for x in s) and not s[1].text.startswith("Benefits"))

# ---- long sections, packing, limits
big = scheme(details=sentences(80), eligibility=sentences(10, prefix="Elig"))
cm = chunker.chunk_scheme(big, count, 200, 40, "merged")
check("long scheme gives several chunks", len(cm) > 3)
check("all chunks within the limit", all(x.tokens <= 200 for x in cm), str([x.tokens for x in cm]))
check("indices contiguous, n_chunks consistent", [x.index for x in cm] == list(range(len(cm))) and all(x.metadata["n_chunks"] == len(cm) for x in cm))
check("ids unique", len({x.chunk_id for x in cm}) == len(cm))
check("no text lost across chunks", words(" ".join(f"{big[k]}" for k in ("details", "eligibility"))) <= words(" ".join(x.text for x in cm)))
check("a short section is never split across chunks",
      sum("Documents required:" in x.text for x in cm) == 1 and sum("Benefits:" in x.text for x in cm) == 1)

# ---- empty and missing sections
sparse = chunker.chunk_scheme(scheme(details="", benefits="", application="", documents="", tags=""), count, 420, 60, "merged")
check("empty sections are skipped", len(sparse) == 1 and sparse[0].sections == ["eligibility"])
none = chunker.chunk_scheme(scheme(details="", benefits="", eligibility="", application="", documents="", tags=""), count, 420, 60, "merged")
check("a scheme with no text gives no chunks", none == [])

# ---- metadata
m = c[0].metadata
check("metadata values are chroma-safe (str/int/float/bool, never None)",
      all(isinstance(v, (str, int, float, bool)) and v is not None for v in m.values()), str({k: type(v) for k, v in m.items()}))
check("section flags set", m["sec_overview"] is True and m["sec_documents"] is True)
check("category flags set", m["cat_education_learning"] is True and m["cat_women_and_child"] is True)
check("scheme fields copied", (m["scheme_id"], m["state"], m["level"]) == ("S001", "Goa", "State") and m["official_link"].endswith("s001"))
check("category flag spelling", chunker.category_flag("Agriculture,Rural & Environment") == "cat_agriculture_rural_environment")
check("chunks never mix schemes",
      all(x.scheme_id == "S002" for x in chunker.chunk_scheme(scheme("S002", details=sentences(50)), count, 150, 30)))

# ---- parameter validation
for bad in [dict(max_tokens=50, overlap_tokens=10), dict(max_tokens=200, overlap_tokens=150), dict(max_tokens=200, overlap_tokens=-1)]:
    try:
        chunker.chunk_scheme(scheme(), count, strategy="merged", **bad)
        check(f"rejects {bad}", False)
    except ValueError:
        check(f"rejects {bad}", True)
try:
    chunker.chunk_scheme(scheme(), count, 200, 20, "nonsense"); check("rejects unknown strategy", False)
except ValueError:
    check("rejects unknown strategy", True)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
