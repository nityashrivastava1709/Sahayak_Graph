"""
STEP B  -  The fixed attribute vocabulary.

Output: data/processed/attribute_vocabulary.json

This file is the "dictionary" of what an eligibility rule is allowed to talk about.
It is used three times in the project:
  * Step C  - the LLM may only extract rules that use these attributes
  * Layer 2 - the knowledge graph stores rules in this format
  * Agent 1 - the user profile is built from exactly these attributes
So change it now, not later.

Value format used everywhere
  number attributes : 18   or   250000
  between           : "18|60"           (min|max)
  in                : "SC|ST|OBC"       (alternatives, separated by |)
  yes/no attributes : "yes" or "no"

Run:  python step_b_vocabulary.py     (prints the table and saves the JSON)
Other scripts do:  from step_b_vocabulary import validate_condition
"""
import json

import config

OPERATORS = ["=", "<", "<=", ">", ">=", "between", "in"]
YES_NO = ["yes", "no"]

ATTRIBUTES = {
    "age": {
        "type": "number", "unit": "years",
        "operators": ["=", "<", "<=", ">", ">=", "between"],
        "description": "Age of the applicant (or of the child/student, if the scheme is about them).",
    },
    "gender": {
        "type": "category", "values": ["female", "male", "transgender"],
        "operators": ["=", "in"],
        "description": "Gender the scheme is restricted to.",
    },
    "annual_income": {
        "type": "number", "unit": "rupees per year",
        "operators": ["<", "<=", ">", ">=", "between"],
        "description": "Annual income limit in rupees (family or individual). Convert lakh to rupees (2.5 lakh = 250000).",
    },
    "state": {
        "type": "state",
        "operators": ["=", "in"],
        "description": "State or UT the applicant must live in / belong to. Use the full official name.",
    },
    "occupation": {
        "type": "category",
        "values": ["farmer", "fisherman", "student", "artisan_or_weaver", "construction_worker",
                   "unorganised_worker", "government_employee", "self_employed",
                   "unemployed", "other"],
        "operators": ["=", "in"],
        "description": "What the applicant does for a living.",
    },
    "social_category": {
        "type": "category",
        "values": ["SC", "ST", "OBC", "General", "EWS", "Minority", "Backward_Class", "Other_Community"],
        "operators": ["=", "in"],
        "description": "Caste / community category. Use Other_Community for named communities such as Brahmin or Saharia.",
    },
    "disability": {
        "type": "yesno", "values": YES_NO, "operators": ["="],
        "description": "Whether the applicant (or family member) must have a disability.",
    },
    "marital_status": {
        "type": "category",
        "values": ["unmarried", "married", "widowed", "divorced", "abandoned_or_deserted"],
        "operators": ["=", "in"],
        "description": "Marital status required. A widow is 'widowed'.",
    },
    "education_level": {
        "type": "category",
        "values": ["class_1_5", "class_6_8", "class_9_10", "class_11_12",
                   "diploma_or_iti", "undergraduate", "postgraduate", "phd"],
        "operators": ["=", "in"],
        "description": "Level the applicant is currently studying at (or has completed).",
    },
    "residence_type": {
        "type": "category", "values": ["rural", "urban"], "operators": ["=", "in"],
        "description": "Rural or urban area of residence.",
    },
    "owns_land": {
        "type": "yesno", "values": YES_NO, "operators": ["="],
        "description": "Whether the applicant must own agricultural land.",
    },
    "land_holding_acres": {
        "type": "number", "unit": "acres",
        "operators": ["<", "<=", ">", ">=", "between"],
        "description": "Size of land held, in acres (convert hectares: 1 ha = 2.47 acres).",
    },
    "bpl_status": {
        "type": "yesno", "values": YES_NO, "operators": ["="],
        "description": "Whether the family must be Below Poverty Line.",
    },
    "pregnancy_status": {
        "type": "yesno", "values": YES_NO, "operators": ["="],
        "description": "Whether the applicant must be pregnant or a recent mother.",
    },
}

# Rules that do not fit any attribute above are stored with this special name.
UNMAPPED = "unmapped"


def _is_number(text) -> bool:
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False


def validate_condition(attribute, operator, value):
    """
    Check one rule against the vocabulary.
    Returns a list of problems (empty list = the rule is valid).
    """
    problems = []
    if attribute == UNMAPPED:
        return problems                                   # allowed, but not used for matching
    spec = ATTRIBUTES.get(attribute)
    if spec is None:
        return [f"unknown attribute '{attribute}'"]
    if operator not in spec["operators"]:
        problems.append(f"operator '{operator}' not allowed for {attribute} "
                        f"(allowed: {spec['operators']})")
        return problems

    value = str(value).strip()
    parts = [p.strip() for p in value.split("|")] if operator in ("in", "between") else [value]

    if operator == "between":
        if len(parts) != 2 or not all(_is_number(p) for p in parts):
            problems.append("between needs two numbers like '18|60'")
        elif float(parts[0]) > float(parts[1]):
            problems.append("between: minimum is larger than maximum")
        return problems

    for p in parts:
        if spec["type"] == "number":
            if not _is_number(p):
                problems.append(f"'{p}' is not a number")
        elif spec["type"] == "state":
            if p not in config.INDIAN_STATES_AND_UTS:
                problems.append(f"'{p}' is not a recognised state/UT name")
        else:                                              # category or yesno
            if p not in spec["values"]:
                problems.append(f"'{p}' is not an allowed value for {attribute} "
                                f"(allowed: {spec['values']})")
    return problems


def vocabulary_for_prompt() -> str:
    """The vocabulary written as plain text, to paste into the LLM instructions."""
    lines = []
    for name, spec in ATTRIBUTES.items():
        line = f"- {name}: {spec['description']} Operators: {', '.join(spec['operators'])}."
        if "values" in spec:
            line += f" Allowed values: {', '.join(spec['values'])}."
        if spec["type"] == "state":
            line += " Value must be a full state/UT name such as Madhya Pradesh."
        lines.append(line)
    return "\n".join(lines)


def main():
    config.VOCAB_JSON.write_text(
        json.dumps({"operators": OPERATORS, "attributes": ATTRIBUTES, "unmapped": UNMAPPED},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(ATTRIBUTES)} attributes defined:\n")
    for name, spec in ATTRIBUTES.items():
        print(f"{name:20} {spec['type']:9} operators={spec['operators']}")
    print(f"\nSaved {config.VOCAB_JSON}")


if __name__ == "__main__":
    main()
