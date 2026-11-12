"""
STEP G  -  Find scheme pairs that conflict or overlap.

Two commands:

  python step_g_conflicts.py candidates [--top 30]
      1. Compares every pair of schemes by text similarity (TF-IDF) and keeps the most
         similar pairs that could apply to the same person (same state, or one is Central).
      2. Asks the LLM to judge each pair: does it make sense to say "these two overlap /
         should not be claimed together"?
      3. Writes data/interim/conflicts_candidates.csv.

  python step_g_conflicts.py finalize
      After you open conflicts_candidates.csv and type `yes` in the `keep` column for the
      pairs you agree with, this writes data/processed/conflicts.csv (the final list).

A "conflict" here means: the two schemes give the same kind of benefit to the same kind of
person, so the Planner agent should warn the user (or the scheme text forbids claiming both).
Expect only a handful of real conflicts; the human check is the important part.
"""
import argparse
from itertools import combinations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
from llm_utils import cached_llm_json

CANDIDATES = config.INTERIM_DIR / "conflicts_candidates.csv"

# The exact JSON shape Gemini must return (structured output).
CONFLICT_SCHEMA = {
    "type": "object",
    "properties": {
        "conflict": {"type": "boolean"},
        "reason": {"type": "string", "description": "one sentence"},
    },
    "required": ["conflict", "reason"],
}

SYSTEM_PROMPT = """You compare two Indian government welfare schemes.

Decide whether they CONFLICT, meaning: one person could be eligible for both, they give the same kind of benefit (for example both are scholarships for the same class, both are widow pensions, both give the same agricultural subsidy), and it would be duplicate support that a planner should warn about, or the text itself forbids claiming both.
They do NOT conflict if they give different kinds of benefit, or apply to different people.

Return ONLY a JSON object: {"conflict": true/false, "reason": "<one sentence>"}"""


def geography_compatible(a, b) -> bool:
    """Two schemes can only apply to the same person if the states match or one is Central."""
    return a["state"] == b["state"] or "Central" in (a["state"], b["state"])


def make_candidates(top_n: int):
    df = pd.read_csv(config.SCHEMES_FINAL, dtype=str).fillna("")
    text = (df["scheme_name"] + " " + df["benefits"] + " " + df["eligibility"]).tolist()
    sim = cosine_similarity(TfidfVectorizer(stop_words="english").fit_transform(text))

    pairs = []
    for i, j in combinations(range(len(df)), 2):
        if geography_compatible(df.iloc[i], df.iloc[j]):
            pairs.append((sim[i, j], i, j))
    pairs.sort(reverse=True)
    pairs = pairs[:top_n]
    print(f"Judging the {len(pairs)} most similar pairs ...")

    rows = []
    for n, (score, i, j) in enumerate(pairs, start=1):
        a, b = df.iloc[i], df.iloc[j]
        print(f"[{n}/{len(pairs)}] {a['scheme_id']} vs {b['scheme_id']}  (similarity {score:.2f})")
        user = (f"SCHEME A: {a['scheme_name']}\nBenefits: {a['benefits'][:700]}\n"
                f"Eligibility: {a['eligibility'][:700]}\n\n"
                f"SCHEME B: {b['scheme_name']}\nBenefits: {b['benefits'][:700]}\n"
                f"Eligibility: {b['eligibility'][:700]}\n\nReturn the JSON object now.")
        verdict = cached_llm_json("conflicts", f"{a['scheme_id']}_{b['scheme_id']}",
                                  SYSTEM_PROMPT, user, schema=CONFLICT_SCHEMA)
        verdict = verdict if isinstance(verdict, dict) else {}
        rows.append({"scheme_a": a["scheme_id"], "scheme_b": b["scheme_id"],
                     "name_a": a["scheme_name"], "name_b": b["scheme_name"],
                     "similarity": round(float(score), 3),
                     "llm_conflict": verdict.get("conflict", ""),
                     "reason": verdict.get("reason", ""), "keep": ""})

    pd.DataFrame(rows).to_csv(CANDIDATES, index=False, encoding="utf-8-sig")
    flagged = sum(1 for r in rows if r["llm_conflict"] is True)
    print(f"\nSaved {CANDIDATES}  ({flagged} pairs flagged by the LLM)")
    print("Open it, type 'yes' in the `keep` column for the pairs you agree with, "
          "then run:  python step_g_conflicts.py finalize")


def finalize():
    cand = pd.read_csv(CANDIDATES, dtype=str).fillna("")
    kept = cand[cand["keep"].str.strip().str.lower().isin(["yes", "y", "true", "1"])]
    kept[["scheme_a", "scheme_b", "reason"]].to_csv(config.CONFLICTS_CSV, index=False,
                                                    encoding="utf-8-sig")
    print(f"Saved {len(kept)} conflict pair(s) -> {config.CONFLICTS_CSV}")
    if kept.empty:
        print("No rows had keep = yes. That is fine if you found no real conflicts, "
              "but check that you saved the file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["candidates", "finalize"])
    parser.add_argument("--top", type=int, default=30, help="how many similar pairs to judge")
    args = parser.parse_args()
    if args.command == "candidates":
        make_candidates(args.top)
    else:
        finalize()
