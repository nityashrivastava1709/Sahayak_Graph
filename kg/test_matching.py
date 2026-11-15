"""
test_matching.py  -  unit tests for the rule-matching logic. No database needed.

Run:  python test_matching.py
Needs data\\processed\\attribute_vocabulary.json (written by Layer 1 step B).
"""
import sys

import matching

passed = failed = 0


def check(name, got, want):
    global passed, failed
    if got == want:
        passed += 1
    else:
        failed += 1
        print(f"FAIL  {name}\n      got  {got!r}\n      want {want!r}")


def ev(profile, attribute, operator, value):
    clean, _ = matching.normalise_profile(profile)
    return matching.evaluate_condition(clean, attribute, operator, value)


# ---- number parsing and states
check("parse '2,50,000'", matching.parse_number("2,50,000"), 250000.0)
check("parse 'Rs 24000'", matching.parse_number("Rs 24000"), 24000.0)
check("parse rupee sign", matching.parse_number("₹180000"), 180000.0)
check("parse bool is not a number", matching.parse_number(True), None)
check("parse junk", matching.parse_number("abc"), None)
check("state code MP", matching.canonical_state("MP"), "Madhya Pradesh")
check("state lower-case", matching.canonical_state("tamil nadu"), "Tamil Nadu")
check("state alias Orissa", matching.canonical_state("Orissa"), "Odisha")
check("state unknown", matching.canonical_state("Atlantis"), None)

# ---- profile cleaning
clean, warns = matching.normalise_profile({"age": "34", "gender": "Woman", "state": "MP", "occupation": "farmer",
                                           "owns_land": True, "income": "2,50,000", "marital_status": "widow"})
check("age -> float", clean["age"], 34.0)
check("gender synonym", clean["gender"], ["female"])
check("state code", clean["state"], "Madhya Pradesh")
check("bool -> yes", clean["owns_land"], "yes")
check("income alias -> annual_income", clean.get("annual_income"), 250000.0)
check("widow -> widowed", clean["marital_status"], ["widowed"])
check("no warnings for the contract-style profile", warns, [])
clean, warns = matching.normalise_profile({"occupation": "teacher", "colour": "red", "age": "n/a", "social_category": "sc"})
check("unknown occupation -> other", clean["occupation"], ["other"])
check("case-insensitive category", clean["social_category"], ["SC"])
check("bad number dropped", "age" not in clean, True)
check("warnings reported", len(warns), 3)
clean, _ = matching.normalise_profile({"age": None, "gender": "", "state": []})
check("empty values are unknown", clean, {})

# ---- evaluating rules: numbers
check("age >= 18 true", ev({"age": 18}, "age", ">=", "18"), True)
check("age >= 18 false", ev({"age": 17}, "age", ">=", "18"), False)
check("income <= limit (boundary)", ev({"annual_income": 250000}, "annual_income", "<=", "250000"), True)
check("income <= limit (over)", ev({"annual_income": 250001}, "annual_income", "<=", "250000"), False)
check("age < 60 boundary", ev({"age": 60}, "age", "<", "60"), False)
check("between lower bound inclusive", ev({"age": 27}, "age", "between", "27|45"), True)
check("between upper bound inclusive", ev({"age": 45}, "age", "between", "27|45"), True)
check("between outside", ev({"age": 46}, "age", "between", "27|45"), False)
# ---- categories, yes/no, states
check("gender =", ev({"gender": "female"}, "gender", "=", "female"), True)
check("gender = wrong", ev({"gender": "male"}, "gender", "=", "female"), False)
check("social_category in", ev({"social_category": "ST"}, "social_category", "in", "SC|ST"), True)
check("social_category in (miss)", ev({"social_category": "OBC"}, "social_category", "in", "SC|ST"), False)
check("list profile value overlaps", ev({"social_category": ["OBC", "EWS"]}, "social_category", "in", "SC|EWS"), True)
check("occupation in", ev({"occupation": "fisherman"}, "occupation", "in", "farmer|fisherman"), True)
check("yes/no match", ev({"disability": "yes"}, "disability", "=", "yes"), True)
check("yes/no mismatch", ev({"bpl_status": False}, "bpl_status", "=", "yes"), False)
check("state match", ev({"state": "up"}, "state", "=", "Uttar Pradesh"), True)
check("state mismatch", ev({"state": "Goa"}, "state", "=", "Uttar Pradesh"), False)
# ---- unknown is not false
check("missing attribute -> None", ev({"age": 30}, "annual_income", "<=", "100000"), None)
check("empty value -> None", ev({"gender": ""}, "gender", "=", "female"), None)

# ---- rule validation and canonical spelling
check("valid rule", matching.validate_condition_value("age", "between", "18|60"), [])
check("bad operator", len(matching.validate_condition_value("gender", ">=", "female")), 1)
check("bad value", len(matching.validate_condition_value("gender", "=", "robot")), 1)
check("bad number", len(matching.validate_condition_value("age", ">=", "abc")), 1)
check("bad state", len(matching.validate_condition_value("state", "=", "Atlantis")), 1)
check("unknown attribute", len(matching.validate_condition_value("height", "=", "5")), 1)
check("between order", len(matching.validate_condition_value("age", "between", "60|18")), 1)
check("canonical number", matching.canonical_value("annual_income", "<=", "250000.0"), "250000")
check("canonical in is sorted", matching.canonical_value("social_category", "in", "ST|SC"), "SC|ST")
check("canonical between", matching.canonical_value("age", "between", "27.0|45"), "27|45")
check("same rule, same key",
      matching.condition_key("social_category", "in", matching.canonical_value("social_category", "in", "ST|SC")),
      matching.condition_key("social_category", "in", matching.canonical_value("social_category", "in", "SC|ST")))
check("label =", matching.condition_label("gender", "=", "female"), "gender=female")
check("label <=", matching.condition_label("annual_income", "<=", "250000"), "annual_income<=250000")
check("label in", matching.condition_label("social_category", "in", "SC|ST"), "social_category in SC|ST")

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
