"""
data_io.py  -  reads the Layer 1 output files and prepares them for the graph.

Both the loader (load_graph.py) and the test reference (reference.py) use this, so the graph and
the reference answers always start from exactly the same cleaned data.

What it does
  * checks every file exists and has the columns it should (clear message if not)
  * keeps scheme IDs exactly as they are in schemes_clean.csv
  * keeps only VALID eligibility rules (valid == True) and re-checks each one against the vocabulary
  * keeps 'unmapped' requirements as text on the scheme (never silently dropped)
  * gives identical rules one spelling, so they become ONE shared graph node
  * checks that every document / step / conflict points at a scheme that exists
"""
import re
from dataclasses import dataclass, field

import pandas as pd

import kg_config as config
import matching


class KGDataError(Exception):
    """A Layer 1 file is missing or malformed. The message says which and what to do."""


@dataclass
class Layer1Data:
    schemes: list = field(default_factory=list)      # one dict per scheme (node properties)
    conditions: list = field(default_factory=list)   # valid, normalised, de-duplicated rules
    documents: list = field(default_factory=list)    # scheme_id, document_name, is_conditional
    steps: dict = field(default_factory=dict)        # scheme_id -> ordered list of step texts
    conflicts: list = field(default_factory=list)    # scheme_a, scheme_b, reason (a < b)
    report: dict = field(default_factory=dict)       # counts and anything that was skipped


def _read(path, required, hint, optional=False):
    if not path.exists():
        if optional:
            return pd.DataFrame(columns=required)
        raise KGDataError(f"{path.name} not found in {path.parent}. {hint}")
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KGDataError(f"{path.name} is missing columns {missing}. Found: {list(df.columns)}. {hint}")
    return df


def load_layer1() -> Layer1Data:
    out = Layer1Data()
    rep = out.report

    # ---------------- schemes ----------------
    s_cols = ["scheme_id", "scheme_name", "slug", "official_link", "level", "state", "primary_category",
              "categories", "details", "benefits", "eligibility", "application_mode",
              "application_steps", "department", "tags"]
    schemes = _read(config.SCHEMES_CSV, s_cols, "Run Layer 1 steps A-F first.")
    ids = schemes["scheme_id"].str.strip().tolist()
    bad = [i for i in ids if not re.fullmatch(r"S\d{3}", i)]
    if bad:
        raise KGDataError(f"scheme_id values not in the form S001: {bad[:5]}")
    if len(set(ids)) != len(ids):
        raise KGDataError("scheme_id values in schemes_clean.csv are not unique.")
    known_ids = set(ids)

    # ---------------- conditions ----------------
    c_cols = ["scheme_id", "group_id", "attribute", "operator", "value", "source_sentence", "valid"]
    cond_df = _read(config.CONDITIONS_CSV, c_cols, "Run Layer 1 step C (python step_c_extract_conditions.py --all).")
    unmapped, seen, rejected = {}, set(), []
    n_invalid = n_dupes = 0
    for row in cond_df.itertuples(index=False):
        sid = row.scheme_id.strip()
        if sid not in known_ids:
            raise KGDataError(f"conditions.csv refers to {sid}, which is not in schemes_clean.csv.")
        if row.attribute == "unmapped":
            if row.value.strip():
                unmapped.setdefault(sid, []).append(row.value.strip())
            continue
        if row.valid != "True":
            n_invalid += 1
            continue
        problems = matching.validate_condition_value(row.attribute, row.operator, row.value)
        if problems:                                   # valid in Layer 1 but fails the re-check
            rejected.append({"scheme_id": sid, "rule": f"{row.attribute} {row.operator} {row.value}",
                             "problems": problems})
            continue
        value = matching.canonical_value(row.attribute, row.operator, row.value)
        key = matching.condition_key(row.attribute, row.operator, value)
        group = row.group_id.strip() or "G0"
        if (sid, group, key) in seen:
            n_dupes += 1
            continue
        seen.add((sid, group, key))
        out.conditions.append({"scheme_id": sid, "group_id": group, "attribute": row.attribute,
                               "operator": row.operator, "value": value, "condition_key": key,
                               "source_sentence": row.source_sentence.strip()})

    # ---------------- scheme node properties ----------------
    for r in schemes.itertuples(index=False):
        sid = r.scheme_id.strip()
        out.schemes.append({
            "scheme_id": sid, "name": r.scheme_name, "slug": r.slug, "official_link": r.official_link,
            "level": r.level, "state": r.state, "primary_category": r.primary_category,
            "categories": [c.strip() for c in r.categories.split("|") if c.strip()],
            "details": r.details, "benefits": r.benefits, "eligibility": r.eligibility,
            "application_mode": r.application_mode or "unclear",
            "department": r.department or "Not stated", "tags": r.tags,
            "unmapped_requirements": unmapped.get(sid, []),
        })

    # ---------------- documents ----------------
    d_df = _read(config.DOCUMENTS_CSV, ["scheme_id", "document_name", "is_conditional"],
                 "Run Layer 1 step E (python step_e_documents.py).")
    seen_docs = set()
    for r in d_df.itertuples(index=False):
        sid, name = r.scheme_id.strip(), r.document_name.strip()
        if not name:
            continue
        if sid not in known_ids:
            raise KGDataError(f"documents.csv refers to {sid}, which is not in schemes_clean.csv.")
        if (sid, name) in seen_docs:
            continue
        seen_docs.add((sid, name))
        out.documents.append({"scheme_id": sid, "document_name": name,
                              "is_conditional": r.is_conditional == "True"})

    # ---------------- application steps ----------------
    steps_df = _read(config.APP_STEPS_CSV, ["scheme_id", "step_no", "step_text"],
                     "Run Layer 1 step F (python step_f_application_department.py).")
    tmp = {}
    for r in steps_df.itertuples(index=False):
        if r.scheme_id.strip() not in known_ids:
            raise KGDataError(f"application_steps.csv refers to {r.scheme_id}, not in schemes_clean.csv.")
        tmp.setdefault(r.scheme_id.strip(), []).append((int(float(r.step_no)), r.step_text.strip()))
    out.steps = {sid: [t for _, t in sorted(v)] for sid, v in tmp.items()}
    for s in schemes.itertuples(index=False):                  # fallback to the joined text column
        sid = s.scheme_id.strip()
        if sid not in out.steps:
            out.steps[sid] = [p.strip() for p in s.application_steps.split("||") if p.strip()]

    # ---------------- conflicts (the file may be empty) ----------------
    cf = _read(config.CONFLICTS_CSV, ["scheme_a", "scheme_b", "reason"],
               "Run Layer 1 step G (python step_g_conflicts.py finalize).", optional=True)
    pairs = {}
    for r in cf.itertuples(index=False):
        a, b = r.scheme_a.strip(), r.scheme_b.strip()
        if a == b:
            continue
        if a not in known_ids or b not in known_ids:
            raise KGDataError(f"conflicts.csv refers to an unknown scheme: {a}, {b}")
        pairs[tuple(sorted((a, b)))] = r.reason.strip()
    out.conflicts = [{"scheme_a": a, "scheme_b": b, "reason": reason} for (a, b), reason in sorted(pairs.items())]

    # ---------------- report ----------------
    stated = [s for s in out.schemes if s["department"] != "Not stated"]
    rep.update({
        "schemes": len(out.schemes),
        "condition_rows_loaded": len(out.conditions),
        "distinct_conditions": len({c["condition_key"] for c in out.conditions}),
        "schemes_with_conditions": len({c["scheme_id"] for c in out.conditions}),
        "unmapped_requirements": sum(len(v) for v in unmapped.values()),
        "invalid_condition_rows_skipped": n_invalid,
        "duplicate_condition_rows_dropped": n_dupes,
        "rejected_on_recheck": rejected,
        "document_links": len(out.documents),
        "distinct_documents": len({d["document_name"] for d in out.documents}),
        "schemes_with_department": len(stated),
        "distinct_departments": len({s["department"] for s in stated}),
        "conflict_pairs": len(out.conflicts),
    })
    return out
