"""
matching.py  -  decides whether a person satisfies one eligibility rule.

This is plain Python (no database). It is used to:
  * clean a user's profile (normalise_profile)
  * test the profile against every EligibilityCondition (evaluate_condition) so the
    Person -[:MATCHES]-> Condition edges can be written to the graph
  * validate / normalise the rules read from Layer 1 (validate_condition_value, canonical_value)

evaluate_condition returns three answers, not two:
    True   the person satisfies the rule
    False  the person's value is known and does NOT satisfy it
    None   unknown: the profile does not contain that attribute
"Unknown" is what lets the system ask follow-up questions instead of wrongly rejecting someone.
"""
import json
import math
import re
from functools import lru_cache

import kg_config as config

SYMBOLS = {"=": "=", "<": "<", "<=": "<=", ">": ">", ">=": ">="}

GENDER_SYNONYMS = {"woman": "female", "women": "female", "f": "female", "girl": "female",
                   "lady": "female", "man": "male", "m": "male", "boy": "male",
                   "third_gender": "transgender"}
CATEGORY_SYNONYMS = {"gender": GENDER_SYNONYMS,
                     "marital_status": {"widow": "widowed", "widower": "widowed", "divorcee": "divorced"}}
YES_WORDS, NO_WORDS = {"yes", "y", "true", "1"}, {"no", "n", "false", "0"}

# Other names callers use for the vocabulary attributes (the original handoff contract used "income")
PROFILE_KEY_ALIASES = {"income": "annual_income", "family_income": "annual_income",
                       "land_holding": "land_holding_acres", "caste": "social_category",
                       "bpl": "bpl_status"}


@lru_cache(maxsize=1)
def load_vocabulary() -> dict:
    """The attribute vocabulary written by Layer 1 (step B)."""
    if not config.VOCAB_JSON.exists():
        raise FileNotFoundError(
            f"{config.VOCAB_JSON} not found. Run Layer 1 step B (python step_b_vocabulary.py) first.")
    return json.loads(config.VOCAB_JSON.read_text(encoding="utf-8"))["attributes"]


# ------------------------------------------------------------------ small parsers
def parse_number(x):
    """'2,50,000', 'Rs 250000', 250000, 2.5e5 -> float.  Anything else -> None."""
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return None if (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else float(x)
    s = re.sub(r"(₹|rs\.?|inr)", "", str(x).strip().lower()).replace(",", "").strip()
    try:
        f = float(s)
    except ValueError:
        return None
    return None if (math.isnan(f) or math.isinf(f)) else f


def fmt_number(f: float) -> str:
    return str(int(f)) if float(f) == int(f) else str(f)


def norm_token(s) -> str:
    return re.sub(r"[\s\-]+", "_", str(s).strip().lower())


def parse_yesno(x):
    if isinstance(x, bool):
        return "yes" if x else "no"
    t = norm_token(x)
    return "yes" if t in YES_WORDS else "no" if t in NO_WORDS else None


def canonical_state(x):
    """'mp', 'MP', 'madhya pradesh', 'Orissa' -> 'Madhya Pradesh'.  Unrecognised -> None."""
    s = str(x).strip()
    lower = s.lower()
    by_lower = {st.lower(): st for st in config.INDIAN_STATES_AND_UTS}
    if lower in by_lower:
        return by_lower[lower]
    if lower in config.STATE_ALIASES:
        return config.STATE_ALIASES[lower]
    return config.STATE_CODES.get(s.upper())


# ------------------------------------------------------------------ rules from Layer 1
def validate_condition_value(attribute, operator, value) -> list:
    """Problems with one rule (empty list = fine). Mirrors the Layer 1 validator."""
    spec = load_vocabulary().get(attribute)
    if spec is None:
        return [f"unknown attribute '{attribute}'"]
    if operator not in spec["operators"]:
        return [f"operator '{operator}' not allowed for {attribute}"]
    value = str(value).strip()
    parts = [p.strip() for p in value.split("|")] if operator in ("in", "between") else [value]
    if operator == "between":
        nums = [parse_number(p) for p in parts]
        if len(parts) != 2 or None in nums:
            return ["between needs two numbers like '18|60'"]
        return [] if nums[0] <= nums[1] else ["between: minimum is larger than maximum"]
    problems = []
    for p in parts:
        if spec["type"] == "number":
            if parse_number(p) is None:
                problems.append(f"'{p}' is not a number")
        elif spec["type"] == "state":
            if p not in config.INDIAN_STATES_AND_UTS:
                problems.append(f"'{p}' is not a recognised state/UT")
        elif p not in spec["values"]:
            problems.append(f"'{p}' is not an allowed value for {attribute}")
    return problems


def canonical_value(attribute, operator, value) -> str:
    """One spelling per rule, so identical rules from different schemes share one graph node."""
    spec = load_vocabulary()[attribute]
    value = str(value).strip()
    if operator == "between":
        lo, hi = (parse_number(p) for p in value.split("|"))
        return f"{fmt_number(lo)}|{fmt_number(hi)}"
    if operator == "in":
        parts = sorted({p.strip() for p in value.split("|") if p.strip()})
        if spec["type"] == "number":
            parts = [fmt_number(parse_number(p)) for p in parts]
        return "|".join(parts)
    return fmt_number(parse_number(value)) if spec["type"] == "number" else value


def condition_key(attribute, operator, value) -> str:
    return f"{attribute}|{operator}|{value}"


def condition_label(attribute, operator, value) -> str:
    """Human-readable form used in graph paths, e.g.  annual_income<=250000  or  social_category in SC|ST."""
    if operator in SYMBOLS:
        return f"{attribute}{SYMBOLS[operator]}{value}"
    return f"{attribute} {operator} {value}"


# ------------------------------------------------------------------ the person's profile
def normalise_profile(profile: dict):
    """
    Clean a raw profile. Returns (clean_profile, warnings).
      numbers  -> float            yes/no  -> 'yes'/'no'        state -> full official name
      category -> list of allowed vocabulary values (even if only one)
    Empty values are treated as unknown. Unrecognised attributes/values are reported in
    `warnings`, never silently turned into a match.
    """
    vocab = load_vocabulary()
    clean, warnings = {}, []
    items = []
    for key, raw in (profile or {}).items():
        target = PROFILE_KEY_ALIASES.get(key, key) if key not in vocab else key
        if target != key and target in (profile or {}):
            continue                                   # the exact vocabulary name is also present; it wins
        items.append((target, raw))
    for key, raw in items:
        if key not in vocab:
            warnings.append(f"ignored unknown attribute '{key}'")
            continue
        if raw is None or (isinstance(raw, str) and not raw.strip()) or \
                (isinstance(raw, (list, tuple, set)) and len(raw) == 0):
            continue
        spec = vocab[key]
        kind = spec["type"]
        if kind == "number":
            n = parse_number(raw)
            if n is None:
                warnings.append(f"{key}: '{raw}' is not a number, treated as unknown")
            else:
                clean[key] = n
        elif kind == "yesno":
            v = parse_yesno(raw)
            if v is None:
                warnings.append(f"{key}: '{raw}' is not yes/no, treated as unknown")
            else:
                clean[key] = v
        elif kind == "state":
            st = canonical_state(raw)
            if st is None:
                warnings.append(f"{key}: '{raw}' is not a recognised state/UT, treated as unknown")
            else:
                clean[key] = st
        else:  # category
            allowed = {norm_token(v): v for v in spec["values"]}
            synonyms = CATEGORY_SYNONYMS.get(key, {})
            values = []
            for item in (raw if isinstance(raw, (list, tuple, set)) else [raw]):
                token = norm_token(item)
                token = synonyms.get(token, token)
                if token in allowed:
                    values.append(allowed[token])
                elif key == "occupation":
                    values.append("other")
                    warnings.append(f"occupation: '{item}' is not in the vocabulary, recorded as 'other'")
                else:
                    warnings.append(f"{key}: '{item}' is not an allowed value, treated as unknown")
            if values:
                clean[key] = sorted(set(values))
    return clean, warnings


def evaluate_condition(clean: dict, attribute: str, operator: str, value: str):
    """True / False / None (unknown) for one rule against a cleaned profile."""
    if attribute not in clean:
        return None
    kind = load_vocabulary()[attribute]["type"]
    have = clean[attribute]
    if kind == "number":
        if operator == "between":
            lo, hi = (float(p) for p in value.split("|"))
            return lo <= have <= hi
        target = float(value)
        return {"=": have == target, "<": have < target, "<=": have <= target,
                ">": have > target, ">=": have >= target}[operator]
    if kind == "yesno":
        return have == value
    if kind == "state":
        return have in (value.split("|") if operator == "in" else [value])
    alternatives = {norm_token(v) for v in (value.split("|") if operator == "in" else [value])}
    return any(norm_token(h) in alternatives for h in have)
