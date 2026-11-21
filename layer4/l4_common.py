"""
l4_common.py  -  the output format every system returns, plus helpers shared by the baselines.

EVERY system (keyword search, vector RAG, later GraphRAG and the full SahayakGraph) returns a dict of this shape,
so Layer 7 can score them all the same way:

{
  "system":  "keyword" | "vector_rag" | "vector_rag+rerank" | ...,
  "query":   the text that was asked,
  "schemes": [ {                                   # best first; ranks are 1, 2, 3 ...
      "scheme_id": "S012", "name": "...", "rank": 1,
      "score": float | None,                       # the system's own relevance score
      "reason": str | None,                        # why it was recommended (systems that explain)
      "documents": [str] | None,                   # None = this system does not provide it
      "steps": [str] | None,                       # application steps; None = not provided
      "evidence": [ {"chunk_id": str | None, "text": str} ],
      "retrieved": bool,                           # was it in the retrieved context? (False = the LLM named a scheme it was not shown)
      "official_link": str | None, "state": str | None } ],
  "answer":  str | None,                           # plain-language summary, for systems that write one
  "invalid_ids": [str],                            # scheme ids the system named that do not exist at all
  "meta":    { "latency_s": float, ... }
}
"""
import json
import re

RESULT_KEYS = ["system", "query", "schemes", "answer", "invalid_ids", "meta"]
SCHEME_KEYS = ["scheme_id", "name", "rank", "score", "reason", "documents", "steps", "evidence",
               "retrieved", "official_link", "state"]


def scheme_entry(scheme_id, name, rank, score=None, reason=None, documents=None, steps=None, evidence=None,
                 retrieved=True, official_link=None, state=None) -> dict:
    return {"scheme_id": scheme_id, "name": name, "rank": rank,
            "score": None if score is None else round(float(score), 6),
            "reason": reason, "documents": documents, "steps": steps,
            "evidence": evidence or [], "retrieved": bool(retrieved),
            "official_link": official_link, "state": state}


def make_result(system, query, schemes, answer=None, invalid_ids=None, meta=None) -> dict:
    return {"system": system, "query": query, "schemes": schemes, "answer": answer,
            "invalid_ids": list(invalid_ids or []), "meta": dict(meta or {})}


def validate_result(result, valid_ids=None) -> list:
    """Problems with a result (empty list = it follows the format)."""
    problems = []
    if not isinstance(result, dict):
        return ["result is not a dict"]
    for k in RESULT_KEYS:
        if k not in result:
            problems.append(f"missing key '{k}'")
    if problems:
        return problems
    if not isinstance(result["system"], str) or not result["system"]:
        problems.append("system must be a non-empty string")
    if not isinstance(result["query"], str) or not result["query"].strip():
        problems.append("query must be a non-empty string")
    if result["answer"] is not None and not isinstance(result["answer"], str):
        problems.append("answer must be a string or None")
    if not isinstance(result["invalid_ids"], list):
        problems.append("invalid_ids must be a list")
    if not isinstance(result["meta"], dict) or not isinstance(result["meta"].get("latency_s"), (int, float)):
        problems.append("meta must be a dict with a numeric latency_s")
    if not isinstance(result["schemes"], list):
        return problems + ["schemes must be a list"]
    seen = set()
    for i, s in enumerate(result["schemes"], start=1):
        if not isinstance(s, dict):
            problems.append(f"scheme #{i} is not a dict")
            continue
        missing = [k for k in SCHEME_KEYS if k not in s]
        if missing:
            problems.append(f"scheme #{i} is missing {missing}")
            continue
        if s["rank"] != i:
            problems.append(f"scheme #{i} has rank {s['rank']} (ranks must be 1, 2, 3 ...)")
        if s["scheme_id"] in seen:
            problems.append(f"scheme {s['scheme_id']} appears twice")
        seen.add(s["scheme_id"])
        if not re.fullmatch(r"S\d{3}", str(s["scheme_id"])):
            problems.append(f"scheme_id '{s['scheme_id']}' is not in the form S001")
        if valid_ids is not None and s["scheme_id"] not in valid_ids:
            problems.append(f"scheme {s['scheme_id']} does not exist")
        if s["score"] is not None and not isinstance(s["score"], (int, float)):
            problems.append(f"scheme {s['scheme_id']}: score must be a number or None")
        for field in ("documents", "steps"):
            v = s[field]
            if v is not None and not (isinstance(v, list) and all(isinstance(x, str) for x in v)):
                problems.append(f"scheme {s['scheme_id']}: {field} must be a list of strings or None")
        if not isinstance(s["evidence"], list):
            problems.append(f"scheme {s['scheme_id']}: evidence must be a list")
        if not isinstance(s["retrieved"], bool):
            problems.append(f"scheme {s['scheme_id']}: retrieved must be true/false")
    try:
        json.dumps(result)
    except (TypeError, ValueError) as e:
        problems.append(f"result is not JSON-serialisable ({e})")
    return problems


def group_hits_by_scheme(hits, k, per_scheme=2) -> list:
    """Chunk hits (best first) -> up to k schemes in order of their best chunk, each with up to `per_scheme` chunks."""
    grouped = {}
    for h in hits:
        entry = grouped.setdefault(h["scheme_id"], {"scheme_id": h["scheme_id"], "scheme_name": h["scheme_name"],
                                                    "score": h.get("score"), "state": h.get("state"),
                                                    "official_link": h.get("official_link"), "chunks": []})
        if len(entry["chunks"]) < per_scheme:
            entry["chunks"].append(h)
    return list(grouped.values())[:k]


# ---------------------------------------------------------------------------------------
# a person's profile -> the question a baseline is asked
# ---------------------------------------------------------------------------------------
_ALIASES = {"income": "annual_income", "family_income": "annual_income", "land_holding": "land_holding_acres",
            "caste": "social_category", "bpl": "bpl_status"}
_NICE = {"construction_worker": "construction worker", "artisan_or_weaver": "artisan or weaver",
         "unorganised_worker": "unorganised sector worker", "government_employee": "government employee",
         "self_employed": "self-employed person", "class_1_5": "classes 1 to 5", "class_6_8": "classes 6 to 8",
         "class_9_10": "class 9 or 10", "class_11_12": "class 11 or 12", "diploma_or_iti": "a diploma or ITI course",
         "undergraduate": "an undergraduate degree", "postgraduate": "a postgraduate degree", "phd": "a Ph.D.",
         "abandoned_or_deserted": "abandoned or deserted", "Backward_Class": "backward class",
         "Other_Community": "other community"}


def _nice(value) -> str:
    v = str(value)
    return _NICE.get(v, v.replace("_", " "))


def _yes(v) -> bool:
    return v is True or str(v).strip().lower() in ("yes", "y", "true", "1")


def _no(v) -> bool:
    return v is False or str(v).strip().lower() in ("no", "n", "false", "0")


def _num(v):
    try:
        f = float(str(v).replace(",", "").replace("Rs", "").replace("₹", "").strip())
        return int(f) if f == int(f) else f
    except ValueError:
        return None


def _first(v):
    return v[0] if isinstance(v, (list, tuple)) and v else v


def _normalise(profile) -> dict:
    out = {}
    for k, v in (profile or {}).items():
        key = _ALIASES.get(k, k) if k not in out else k
        if v is None or v == "" or v == []:
            continue
        out[key] = v
    return out


def _state_name(value):
    """'MP' -> 'Madhya Pradesh' when Layer 3 is available; otherwise the value as typed."""
    try:
        import l4_config
        l4_config.ensure_layer3()
        import rag_config
        return rag_config.canonical_state(value) or str(value)
    except Exception:
        return str(value)


def profile_state(profile):
    """The profile's state as typed (None if absent); used as a filter by systems that support one."""
    p = _normalise(profile)
    return str(p["state"]) if p.get("state") else None


def profile_to_query(profile) -> str:
    """
    A fixed, natural-language question built from a structured profile. All baselines receive this same text,
    so differences between systems come from the systems, not from how the question was phrased.
    """
    p = _normalise(profile)
    parts = []
    if (age := _num(p.get("age"))) is not None:
        parts.append(f"I am {age} years old")
    gender = _first(p.get("gender"))
    if gender:
        parts.append({"female": "I am a woman", "male": "I am a man"}.get(str(gender).lower(), f"I am {gender}"))
    if p.get("marital_status"):
        parts.append(f"I am {_nice(_first(p['marital_status']))}")
    if p.get("occupation"):
        parts.append(f"I work as a {_nice(_first(p['occupation']))}")
    if p.get("education_level"):
        parts.append(f"I am studying for or have completed {_nice(_first(p['education_level']))}")
    if p.get("state"):
        parts.append(f"I live in {_state_name(p['state'])}")
    if p.get("residence_type"):
        parts.append(f"I live in a {_first(p['residence_type'])} area")
    if (inc := _num(p.get("annual_income"))) is not None:
        parts.append(f"my annual family income is Rs {inc}")
    if p.get("social_category"):
        cat = p["social_category"]
        parts.append("I belong to the " + " and ".join(_nice(c) for c in (cat if isinstance(cat, (list, tuple)) else [cat])) + " category")
    if "disability" in p and _yes(p["disability"]):
        parts.append("I have a disability")
    if "bpl_status" in p:
        if _yes(p["bpl_status"]):
            parts.append("my family is below the poverty line")
        elif _no(p["bpl_status"]):
            parts.append("my family is not below the poverty line")
    if "owns_land" in p:
        if _yes(p["owns_land"]):
            parts.append("I own agricultural land")
        elif _no(p["owns_land"]):
            parts.append("I do not own agricultural land")
    if (acres := _num(p.get("land_holding_acres"))) is not None:
        parts.append(f"I hold about {acres} acres of land")
    if "pregnancy_status" in p and _yes(p["pregnancy_status"]):
        parts.append("I am pregnant")
    text = ", ".join(parts)
    text = (text[0].upper() + text[1:] + ". ") if text else ""
    return text + "Which government welfare schemes can I apply for, and what documents do I need?"
