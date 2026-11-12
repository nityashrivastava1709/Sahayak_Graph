"""
STEP C  -  Turn eligibility text into structured conditions (LLM).

Input : data/interim/schemes_step_a.csv
Output: data/processed/conditions.csv        (full run)
        data/interim/conditions_pilot.csv    (pilot run)

Each output row is ONE rule, for example:
    S041 | G3 | annual_income | <= | 24000 | "The annual income ... not exceed 24,000"

Columns
  condition_id    unique id (C0001, ...)
  scheme_id       which scheme the rule belongs to
  group_id        rules with the SAME group_id are alternatives (OR);
                  different groups must ALL be met (AND)
  attribute / operator / value   the rule, written in the Step B vocabulary
  source_sentence the sentence the rule came from (for checking)
  source          'llm' (extracted from text) or 'metadata' (state added from the state column)
  valid           True if the rule passes the vocabulary check and can be used by the graph
  problems        why a rule is not valid

How to use
  1. python step_c_extract_conditions.py --pilot     (first 10 schemes, printed for review)
  2. Read the printout. If rules are wrong, improve SYSTEM_PROMPT below and run --pilot again.
  3. python step_c_extract_conditions.py --all       (all schemes)

  To re-test specific schemes after changing the prompt (cheap, prints them for review):
     python step_c_extract_conditions.py --ids S045,S087,S011
"""
import argparse

import pandas as pd

import config
from llm_utils import cached_llm_json
from step_b_vocabulary import (ATTRIBUTES, OPERATORS, UNMAPPED, validate_condition,
                               vocabulary_for_prompt)

SYSTEM_PROMPT = f"""You convert the eligibility text of an Indian government scheme into structured rules.

You must use ONLY these attributes:
{vocabulary_for_prompt()}

Rules you must follow:
1. Extract only what the text explicitly states. Never guess or add rules.
2. One rule per requirement. Use only the attributes, operators and values listed above.
3. Income: "not exceed / up to / below" -> operator "<=" with the number in rupees ("2.5 lakh" = 250000, "Rs 24,000/-" = 24000).
4. Same attribute with alternatives ("SC or ST", "farmer or agricultural labourer") -> ONE rule with operator "in" and values joined by "|", e.g. "SC|ST".
5. Alternatives across DIFFERENT attributes ("a widow OR aged above 60") -> give those rules the same positive integer in "or_group". Rules that are not alternatives must have "or_group": 0.
6. A requirement that does not fit any attribute (for example "has crossed seven months of pregnancy", "must not receive another pension", "studying in a government school") -> attribute "unmapped", operator "text", value = a short description.
7. Ignore trivial citizenship lines such as "should be a citizen of India".
8. A farmer who is a "bonafide farmer engaged in agriculture" -> occupation = farmer.
9. "source_sentence" must be copied from the text (the sentence the rule comes from).

COMPLETENESS (the most common mistake is leaving requirements out):
10. Work through the text sentence by sentence, in order. EVERY sentence or clause that states a requirement must produce at least one rule: a mapped rule if it fits the vocabulary, otherwise an "unmapped" rule. Never skip a requirement and never merge several different requirements into one rule.
11. One sentence can hold several requirements. Split them: "Widows who wish to start a small business can apply" -> gender = female, marital_status = widowed, plus an unmapped rule for "wishes to start a small business".
12. Age range: "minimum age 20 and maximum age 40" -> ONE rule: age between "20|40". Any relaxation or exception to a limit ("2 years relaxation for women candidates") -> a separate unmapped rule. Date-of-birth cut-offs ("born after 1 March 2010") -> unmapped.
13. Education: a qualification the applicant must already hold ("a Master's degree") -> the matching education_level (postgraduate). A programme the applicant is applying for ("applying for a doctorate") -> education_level = phd. Marks, grades, percentages, class rank, entrance exams -> unmapped rules.
14. Negative requirements ("should not own a four-wheeler", "should not be a government pensioner") -> unmapped rules, one per requirement.
15. Exceptions and alternative routes (for example a section headed "Exceptions" or "Other cases") -> one unmapped rule for each distinct case, with a short description.
16. If the text only lists kinds of applicant (organisations, institutions, panchayats, individuals ...), output one unmapped rule for each kind, with value "applicant type: <kind>".

Return ONLY a JSON list, no commentary. Each item has exactly these keys:
"or_group", "attribute", "operator", "value", "source_sentence".

Example 1 input:
The applicant should be a citizen of India. The applicant should be a native of Puducherry by birth or by continuous residence of not less than 5 years. The applicant should be a woman. The applicant should be pregnant. The applicant should have attained the age of 18 years. The applicant should have crossed seven months of pregnancy. The annual income of the applicant should not exceed Rs 24,000 per annum.

Example 1 output:
[
 {{"or_group": 0, "attribute": "state", "operator": "=", "value": "Puducherry", "source_sentence": "The applicant should be a native of Puducherry by birth or by continuous residence of not less than 5 years."}},
 {{"or_group": 0, "attribute": "gender", "operator": "=", "value": "female", "source_sentence": "The applicant should be a woman."}},
 {{"or_group": 0, "attribute": "pregnancy_status", "operator": "=", "value": "yes", "source_sentence": "The applicant should be pregnant."}},
 {{"or_group": 0, "attribute": "age", "operator": ">=", "value": "18", "source_sentence": "The applicant should have attained the age of 18 years."}},
 {{"or_group": 0, "attribute": "unmapped", "operator": "text", "value": "has crossed seven months of pregnancy", "source_sentence": "The applicant should have crossed seven months of pregnancy."}},
 {{"or_group": 0, "attribute": "annual_income", "operator": "<=", "value": "24000", "source_sentence": "The annual income of the applicant should not exceed Rs 24,000 per annum."}}
]

Example 2 input:
Unemployed women who wish to pursue a Master's degree can apply. Applicants must hold a Bachelor's degree with at least 55% marks. The age should be between 21 and 40 years, with 5 years relaxation for SC/ST candidates. Applicants already receiving any scholarship are not eligible.

Example 2 output:
[
 {{"or_group": 0, "attribute": "gender", "operator": "=", "value": "female", "source_sentence": "Unemployed women who wish to pursue a Master's degree can apply."}},
 {{"or_group": 0, "attribute": "occupation", "operator": "=", "value": "unemployed", "source_sentence": "Unemployed women who wish to pursue a Master's degree can apply."}},
 {{"or_group": 0, "attribute": "unmapped", "operator": "text", "value": "wishes to pursue a Master's degree", "source_sentence": "Unemployed women who wish to pursue a Master's degree can apply."}},
 {{"or_group": 0, "attribute": "education_level", "operator": "=", "value": "undergraduate", "source_sentence": "Applicants must hold a Bachelor's degree with at least 55% marks."}},
 {{"or_group": 0, "attribute": "unmapped", "operator": "text", "value": "at least 55% marks in Bachelor's degree", "source_sentence": "Applicants must hold a Bachelor's degree with at least 55% marks."}},
 {{"or_group": 0, "attribute": "age", "operator": "between", "value": "21|40", "source_sentence": "The age should be between 21 and 40 years, with 5 years relaxation for SC/ST candidates."}},
 {{"or_group": 0, "attribute": "unmapped", "operator": "text", "value": "5 years age relaxation for SC/ST candidates", "source_sentence": "The age should be between 21 and 40 years, with 5 years relaxation for SC/ST candidates."}},
 {{"or_group": 0, "attribute": "unmapped", "operator": "text", "value": "must not already receive any scholarship", "source_sentence": "Applicants already receiving any scholarship are not eligible."}}
]
"""

# The exact JSON shape Gemini must return (structured output). Plain types only, so every
# Gemini model accepts it. The vocabulary check in Step B still runs on every rule afterwards.
CONDITIONS_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "or_group": {"type": "integer",
                         "description": "0 for a normal rule; the same positive integer for alternatives"},
            "attribute": {"type": "string", "enum": list(ATTRIBUTES) + [UNMAPPED]},
            "operator": {"type": "string", "enum": OPERATORS + ["text"]},
            "value": {"type": "string",
                      "description": "number, text, or alternatives joined with | (see the rules)"},
            "source_sentence": {"type": "string",
                                "description": "sentence copied from the eligibility text"},
        },
        "required": ["or_group", "attribute", "operator", "value", "source_sentence"],
    },
}


def norm_value(value) -> str:
    """Make every value a clean string: 250000.0 -> '250000', ['SC','ST'] -> 'SC|ST'."""
    if isinstance(value, (list, tuple)):
        return "|".join(norm_value(v) for v in value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def assign_groups(items):
    """Turn the LLM's 'or_group' numbers into group ids G1, G2, ..."""
    mapping, counter, groups = {}, 1, []
    for it in items:
        og = it.get("or_group")
        if og in (None, "", "null", 0):
            groups.append(f"G{counter}")
            counter += 1
        else:
            if og not in mapping:
                mapping[og] = f"G{counter}"
                counter += 1
            groups.append(mapping[og])
    return groups


def extract_scheme(row):
    """Call the LLM for one scheme and return a list of rule dictionaries."""
    user = (f"Scheme name: {row['scheme_name']}\n\n"
            f"Eligibility text:\n{row['eligibility']}\n\nReturn the JSON list now.")
    items = cached_llm_json("conditions", row["scheme_id"], SYSTEM_PROMPT, user,
                            schema=CONDITIONS_SCHEMA)
    if not isinstance(items, list):
        print(f"   {row['scheme_id']}: no usable answer, skipped")
        return []

    items = [i for i in items if isinstance(i, dict)]
    groups = assign_groups(items)
    rows = []
    for it, g in zip(items, groups):
        attr = str(it.get("attribute", "")).strip()
        op = str(it.get("operator", "")).strip()
        val = norm_value(it.get("value", ""))
        problems = validate_condition(attr, op, val)
        rows.append({
            "scheme_id": row["scheme_id"], "group_id": g,
            "attribute": attr, "operator": op, "value": val,
            "source_sentence": str(it.get("source_sentence", "")).strip(),
            "source": "llm",
            "valid": (not problems) and attr != UNMAPPED,
            "problems": "; ".join(problems),
        })
    return rows


def add_state_from_metadata(rows, row):
    """State schemes: if no valid state rule was extracted, add one from the `state` column."""
    has_state = any(r["attribute"] == "state" and r["valid"] for r in rows)
    if row["level"].lower() == "state" and row["state"] in config.INDIAN_STATES_AND_UTS and not has_state:
        n_groups = len({r["group_id"] for r in rows})
        rows.append({
            "scheme_id": row["scheme_id"], "group_id": f"G{n_groups + 1}",
            "attribute": "state", "operator": "=", "value": row["state"],
            "source_sentence": "(added from the scheme's state column)",
            "source": "metadata", "valid": True, "problems": "",
        })
    return rows


def run(df, out_path):
    all_rows = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        print(f"[{i}/{len(df)}] {row['scheme_id']} {row['scheme_name'][:60]}")
        rows = extract_scheme(row)
        rows = add_state_from_metadata(rows, row)
        all_rows.extend(rows)

    out = pd.DataFrame(all_rows)
    out.insert(0, "condition_id", [f"C{n:04d}" for n in range(1, len(out) + 1)])
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out


def print_pilot_review(df, out):
    """Show eligibility text and extracted rules side by side so you can check them by eye."""
    for _, row in df.iterrows():
        print("\n" + "=" * 90)
        print(f"{row['scheme_id']}  {row['scheme_name'][:80]}")
        print("TEXT :", row["eligibility"][:700])
        print("RULES:")
        for _, c in out[out["scheme_id"] == row["scheme_id"]].iterrows():
            if c["attribute"] == UNMAPPED:
                flag = "   (unmapped: kept for reference, not used for matching)"
            else:
                flag = "" if c["valid"] else "   <-- NOT VALID: " + str(c["problems"])
            print(f"   [{c['group_id']}] {c['attribute']} {c['operator']} {c['value']}{flag}")


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true", help="first 10 schemes only")
    mode.add_argument("--all", action="store_true", help="all schemes")
    mode.add_argument("--ids", help="only these schemes, e.g. S045,S087 (printed for review)")
    args = parser.parse_args()

    df = pd.read_csv(config.STEP_A_OUT, dtype=str).fillna("")
    if args.pilot or args.ids:
        if args.ids:
            wanted = [i.strip().upper() for i in args.ids.split(",") if i.strip()]
            unknown = [i for i in wanted if i not in set(df["scheme_id"])]
            if unknown:
                raise SystemExit(f"Unknown scheme id(s): {unknown}")
            df = df[df["scheme_id"].isin(wanted)]
        else:
            df = df.head(10)
        out = run(df, config.INTERIM_DIR / "conditions_pilot.csv")
        print_pilot_review(df, out)
        print("\nCheck every rule above against its text. If you find mistakes, edit SYSTEM_PROMPT and re-run.")
        print("(This test did not change conditions.csv. Run --all when you are happy.)")
    else:
        out = run(df, config.CONDITIONS_CSV)
        print(f"\nSaved {len(out)} rules for {out['scheme_id'].nunique()} schemes -> {config.CONDITIONS_CSV}")
        print(f"Valid rules: {int(out['valid'].sum())}   Unmapped/invalid: {int((~out['valid']).sum())}")


if __name__ == "__main__":
    main()