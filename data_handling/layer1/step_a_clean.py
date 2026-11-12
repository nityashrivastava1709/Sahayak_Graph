"""
STEP A  -  Clean the text and add the missing columns.

Input : data/raw/schemes_clean.csv            (your 90 schemes, only ever READ, never edited)
Output: data/interim/schemes_step_a.csv

What this script does
  1. Removes invisible junk characters (the \ufeff you have in 59 rows), HTML tags
     and extra spaces from every text column.
  2. Adds a `state` column ("Central" for central schemes; for state schemes it is
     found by counting state names in the scheme's text).
  3. Splits the messy category strings into a clean list and picks a primary category.
  4. Adds `official_link` built from the slug.
  5. Saves a review file for any scheme whose state could not be found.

Run:  python step_a_clean.py
"""
import hashlib
import html
import re

import pandas as pd

import config

TEXT_COLS = ["scheme_name", "details", "benefits", "eligibility",
             "application", "documents", "tags"]

# The 8 category names in your data. Some contain commas themselves, which is why
# we search for these exact names instead of splitting on commas.
CANONICAL_CATEGORIES = [
    "Agriculture,Rural & Environment",
    "Education & Learning",
    "Social welfare & Empowerment",
    "Women and Child",
    "Science, IT & Communications",
    "Banking,Financial Services and Insurance",
    "Health & Wellness",
    "Skills & Employment",
]


def clean_text(value) -> str:
    """Remove junk characters, HTML and extra whitespace from one text value."""
    s = html.unescape(str(value))
    s = re.sub(r"<[^>]+>", " ", s)                      # HTML tags
    for junk in ("\ufeff", "\u200b", "\u200c", "\u200d", "\u2060"):
        s = s.replace(junk, "")                         # invisible characters
    s = s.replace("\xa0", " ")                          # non-breaking space
    return re.sub(r"\s+", " ", s).strip()               # collapse spaces/newlines


def split_categories(raw: str):
    """'Education & Learning, Women and Child' -> ['Education & Learning', 'Women and Child']"""
    found = [(raw.find(c), c) for c in CANONICAL_CATEGORIES if c in raw]
    found.sort()                                        # keep the original order
    leftover = raw
    for _, c in found:
        leftover = leftover.replace(c, "")
    if re.search(r"[A-Za-z]", leftover):                # something we did not recognise
        print(f"   WARNING: unrecognised category text in: {raw!r}")
    return [c for _, c in found]


def build_state_patterns():
    """One regex per state, covering alternative spellings."""
    names = {s.lower(): s for s in config.INDIAN_STATES_AND_UTS}
    names.update({alias: state for alias, state in config.STATE_ALIASES.items()})
    patterns = []
    for text, state in names.items():
        patterns.append((re.compile(r"\b" + re.escape(text) + r"\b", re.IGNORECASE), state))
    return patterns


STATE_PATTERNS = build_state_patterns()
# a mention in the title or eligibility text is stronger evidence than elsewhere
FIELD_WEIGHTS = {"scheme_name": 5, "eligibility": 3, "details": 1,
                 "application": 1, "benefits": 1, "documents": 1}


def detect_state(row) -> str:
    """Count state-name mentions (weighted) and return the most mentioned state."""
    if row["level"].strip().lower() == "central":
        return "Central"
    if row["scheme_id"] in config.STATE_OVERRIDES:
        return config.STATE_OVERRIDES[row["scheme_id"]]
    score = {}
    for field, weight in FIELD_WEIGHTS.items():
        for pattern, state in STATE_PATTERNS:
            hits = len(pattern.findall(row[field]))
            if hits:
                score[state] = score.get(state, 0) + hits * weight
    return max(score, key=score.get) if score else "UNKNOWN"


REQUIRED_COLUMNS = ["scheme_id", "scheme_name", "slug", "details", "benefits", "eligibility",
                    "application", "documents", "level", "schemeCategory", "tags"]


def record_input_fingerprint():
    """Save/compare a SHA-256 of the input file, to prove it was never modified."""
    digest = hashlib.sha256(config.RAW_SUBSET.read_bytes()).hexdigest()
    if config.RAW_HASH_FILE.exists():
        if config.RAW_HASH_FILE.read_text().strip() != digest:
            print("WARNING: the input CSV has CHANGED since the first run "
                  f"({config.RAW_HASH_FILE.name}). If you replaced it on purpose, delete that "
                  "file; otherwise restore your original.")
    else:
        config.RAW_HASH_FILE.write_text(digest)
    print(f"Input file fingerprint (SHA-256): {digest[:16]}...")


def check_input(df) -> list:
    """Stop early if the input is not what the pipeline expects. Returns the IDs in order."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"Input CSV is missing columns: {missing}\nFound: {list(df.columns)}")
    ids = df["scheme_id"].str.strip().tolist()
    bad = [i for i in ids if not re.fullmatch(r"S\d{3}", i)]
    if bad:
        raise SystemExit(f"These scheme_id values are not in the form S001: {bad[:5]}")
    if len(set(ids)) != len(ids):
        raise SystemExit("scheme_id values are not unique.")
    if len(ids) != 90:
        print(f"NOTE: expected 90 schemes, found {len(ids)}.")
    return ids


def main():
    record_input_fingerprint()
    df = pd.read_csv(config.RAW_SUBSET, dtype=str, encoding="utf-8-sig").fillna("")
    original_ids = check_input(df)      # IDs are only ever copied from the file, never generated
    print(f"Loaded {len(df)} schemes from {config.RAW_SUBSET.name}")

    junk_before = df.apply(lambda r: any("\ufeff" in v for v in r), axis=1).sum()
    for col in TEXT_COLS:
        df[col] = df[col].map(clean_text)
    for col in ("slug", "level", "schemeCategory", "scheme_id"):
        df[col] = df[col].str.strip()
    print(f"Rows that contained junk characters before cleaning: {junk_before}")

    broken = df.apply(lambda r: any(("â€" in v or "Ã" in v) for v in r), axis=1).sum()
    if broken:
        print(f"WARNING: {broken} rows look like they have broken encoding (e.g. 'â€™'). "
              "Open them and check.")

    # state
    df["state"] = df.apply(detect_state, axis=1)

    # categories
    df["category_list"] = df["schemeCategory"].map(split_categories)
    df["categories"] = df["category_list"].map(" | ".join)
    df["primary_category"] = df["category_list"].map(lambda L: L[0] if L else "UNKNOWN")

    # link
    df["official_link"] = "https://www.myscheme.gov.in/schemes/" + df["slug"]

    out_cols = ["scheme_id", "scheme_name", "slug", "official_link", "level", "state",
                "primary_category", "categories", "details", "benefits", "eligibility",
                "application", "documents", "tags"]
    df[out_cols].to_csv(config.STEP_A_OUT, index=False, encoding="utf-8-sig")

    # Proof that no ID was changed, added, dropped or reordered.
    saved_ids = pd.read_csv(config.STEP_A_OUT, dtype=str, encoding="utf-8-sig")["scheme_id"].tolist()
    if saved_ids != original_ids:
        raise SystemExit("ERROR: scheme IDs changed between input and output. Output was not trusted.")
    print(f"ID check passed: {len(saved_ids)} IDs identical to the input and in the same order.")

    # ---- report ----
    print("\nSchemes per state:")
    print(df["state"].value_counts().to_string())
    print("\nSchemes per primary category:")
    print(df["primary_category"].value_counts().to_string())

    unknown = df[df["state"] == "UNKNOWN"]
    if len(unknown):
        review = config.INTERIM_DIR / "state_review.csv"
        unknown[["scheme_id", "scheme_name", "eligibility"]].to_csv(review, index=False)
        print(f"\n{len(unknown)} scheme(s) have no detectable state -> see {review}")
        print("Fix: add them to STATE_OVERRIDES in config.py (e.g. {'S002': 'Puducherry'}) "
              "and run this script again.")
    print(f"\nSaved {config.STEP_A_OUT}")


if __name__ == "__main__":
    main()
