"""
queries.py  -  the reusable graph queries (Layer 2's public interface).

Every function takes `db` (from graph_db.connect()) as its first argument and returns plain
Python lists / dicts that can be saved as JSON. Agents in later layers only use these functions;
they never write Cypher themselves.

  PERSON
    create_person(db, profile)              store a user profile as a Person node
    match_person_to_conditions(db, ...)     write  (Person)-[:MATCHES]->(Condition)  edges
    register_person(db, profile)            the two steps above in one call
    delete_person(db, person_id)            remove a Person (and its edges)
  DISCOVERY (multi-hop traversal  Person -> Condition -> Scheme)
    eligible_schemes(db, person_id)         schemes where EVERY eligibility group is satisfied
    possible_schemes(db, person_id, clean)  schemes that could qualify if missing details were supplied
    kg_paths(db, person_id, scheme_id)      the evidence: which rules the person matched, per scheme
  SCHEME FACTS
    scheme_summaries(db, scheme_ids)        name, level, state, category, benefits, link ...
    scheme_details(db, scheme_id)           everything stored on one scheme
    documents_for_schemes(db, scheme_ids)   required documents (and whether they are conditional)
    scheme_department(db, scheme_id)        issuing department (None if the data does not state one)
    scheme_application(db, scheme_id)       application mode + ordered steps
  PLANNING
    scheme_conflicts(db, scheme_ids)        pairs inside a set that conflict
    conflicts_of(db, scheme_id)             every scheme that conflicts with this one
    shared_documents(db, scheme_ids)        documents needed by two or more of the schemes
  ADMIN
    graph_stats(db)                         node and relationship counts

Eligibility rule: a scheme qualifies when EVERY group of its conditions has at least one matched
condition (conditions inside one group are alternatives, i.e. OR; groups are combined with AND).
"""
import uuid

import kg_config as config
import matching
from schema import NODE_LABELS, REL_TYPES


# =========================================================================================
# PERSON
# =========================================================================================
def create_person(db, profile: dict, person_id: str = None):
    """
    Store a (cleaned) profile as a Person node.
    Returns (person_id, clean_profile, warnings). `profile` may use raw values; it is normalised
    first (numbers, yes/no, state names, category spellings).
    """
    clean, warnings = matching.normalise_profile(profile)
    person_id = person_id or f"P-{uuid.uuid4().hex[:12]}"
    _write_person(db, person_id, clean)
    return person_id, clean, warnings


def _write_person(db, person_id, clean):
    """Create/update the Person node with one property per known attribute."""
    vocab = matching.load_vocabulary()
    # property names come from the fixed vocabulary (normalise_profile guarantees it), never from raw user input
    assert all(a in vocab for a in clean), "profile contains an attribute outside the vocabulary"
    sets = ", ".join(f"p.{a} = ${a}" for a in clean)
    db.run(f"MERGE (p:Person {{person_id: $pid}}){(' SET ' + sets) if sets else ''}", pid=person_id, **clean)


def match_person_to_conditions(db, person_id: str, clean_profile: dict) -> dict:
    """
    Test the profile against every EligibilityCondition in the graph and write the
    (Person)-[:MATCHES]->(EligibilityCondition) edges for the ones that are satisfied.
    Old MATCHES edges of this person are replaced, so it is safe to call again after the profile changes.
    """
    conditions = db.run("""
        MATCH (c:EligibilityCondition)
        RETURN c.condition_key AS key, c.attribute AS attribute, c.operator AS operator, c.value AS value""")
    matched, not_matched, unknown = [], 0, 0
    for c in conditions:
        verdict = matching.evaluate_condition(clean_profile, c["attribute"], c["operator"], c["value"])
        if verdict is True:
            matched.append(c["key"])
        elif verdict is False:
            not_matched += 1
        else:
            unknown += 1
    db.run("MATCH (p:Person {person_id: $pid})-[m:MATCHES]->() DELETE m", pid=person_id)
    for i in range(0, len(matched), config.BATCH_SIZE):
        db.run("""
            MATCH (p:Person {person_id: $pid})
            UNWIND $keys AS key
            MATCH (c:EligibilityCondition {condition_key: key})
            MERGE (p)-[:MATCHES]->(c)""", pid=person_id, keys=matched[i:i + config.BATCH_SIZE])
    return {"matched": len(matched), "not_matched": not_matched, "unknown": unknown}


def register_person(db, profile: dict, person_id: str = None) -> dict:
    """create_person + match_person_to_conditions. The usual entry point for a new user."""
    pid, clean, warnings = create_person(db, profile, person_id)
    counts = match_person_to_conditions(db, pid, clean)
    return {"person_id": pid, "clean_profile": clean, "warnings": warnings, "condition_counts": counts}


def delete_person(db, person_id: str):
    db.run("MATCH (p:Person {person_id: $pid}) DETACH DELETE p", pid=person_id)


# =========================================================================================
# DISCOVERY
# =========================================================================================
def eligible_schemes(db, person_id: str) -> list:
    """
    Schemes the person fully qualifies for: every eligibility group has at least one matched condition.
    Pure graph traversal:  Person -[:MATCHES]-> Condition -[:QUALIFIES_FOR]-> Scheme.
    """
    rows = db.run("""
        MATCH (p:Person {person_id: $pid})
        MATCH (c:EligibilityCondition)-[q:QUALIFIES_FOR]->(s:Scheme)
        OPTIONAL MATCH (p)-[m:MATCHES]->(c)
        WITH s, q.group_id AS grp, count(m) AS hits
        WITH s, grp, CASE WHEN hits > 0 THEN 1 ELSE 0 END AS group_ok
        WITH s, count(grp) AS n_groups, sum(group_ok) AS n_ok
        WHERE n_groups = n_ok
        RETURN s.scheme_id AS scheme_id, s.name AS name, s.level AS level, s.state AS state,
               s.primary_category AS primary_category, n_groups AS groups
        ORDER BY scheme_id""", pid=person_id)
    return rows


def possible_schemes(db, person_id: str, clean_profile: dict) -> list:
    """
    Schemes the person might qualify for if they supplied the missing details.
    A scheme is 'possible' when no eligibility group is contradicted by the profile and at least one
    group could still be satisfied by an attribute the profile does not contain yet.
    Each result lists `missing_attributes` (what to ask the user) and `graph_score`
    (share of groups already satisfied).
    """
    rows = db.run("""
        MATCH (p:Person {person_id: $pid})
        MATCH (c:EligibilityCondition)-[q:QUALIFIES_FOR]->(s:Scheme)
        OPTIONAL MATCH (p)-[m:MATCHES]->(c)
        WITH s, q.group_id AS grp, count(m) AS hits, collect(DISTINCT c.attribute) AS attrs
        RETURN s.scheme_id AS scheme_id, grp, hits, attrs""", pid=person_id)
    known = set(clean_profile)
    schemes = {}
    for r in rows:
        info = schemes.setdefault(r["scheme_id"], {"ok": 0, "open": 0, "violated": 0, "missing": set()})
        if r["hits"] > 0:
            info["ok"] += 1
            continue
        unknown_attrs = [a for a in r["attrs"] if a not in known]
        if unknown_attrs:
            info["open"] += 1
            info["missing"].update(unknown_attrs)
        else:
            info["violated"] += 1
    out = []
    for sid, info in sorted(schemes.items()):
        if info["violated"] == 0 and info["open"] > 0:
            total = info["ok"] + info["open"]
            out.append({"scheme_id": sid, "missing_attributes": sorted(info["missing"]),
                        "groups_satisfied": info["ok"], "groups_total": total,
                        "graph_score": round(info["ok"] / total, 3)})
    return out


def kg_paths(db, person_id: str, scheme_id: str) -> list:
    """
    The evidence for one scheme: every rule of the scheme that this person matched.
    Each item has the handoff-contract path ["Person", "MATCHES:<rule>", "QUALIFIES_FOR", "<scheme_id>"].
    """
    rows = db.run("""
        MATCH (p:Person {person_id: $pid})-[:MATCHES]->(c:EligibilityCondition)-[q:QUALIFIES_FOR]->(s:Scheme {scheme_id: $sid})
        RETURN c.attribute AS attribute, c.operator AS operator, c.value AS value,
               q.group_id AS group_id, q.source_sentence AS source_sentence
        ORDER BY group_id, attribute""", pid=person_id, sid=scheme_id)
    paths = []
    for r in rows:
        label = matching.condition_label(r["attribute"], r["operator"], r["value"])
        paths.append({"group_id": r["group_id"], "rule": label, "source_sentence": r["source_sentence"],
                      "path": ["Person", f"MATCHES:{label}", "QUALIFIES_FOR", scheme_id]})
    return paths


# =========================================================================================
# SCHEME FACTS
# =========================================================================================
def scheme_summaries(db, scheme_ids: list) -> list:
    """Short facts for a list of schemes, returned in the same order as `scheme_ids`."""
    if not scheme_ids:
        return []
    rows = db.run("""
        MATCH (s:Scheme) WHERE s.scheme_id IN $ids
        RETURN s.scheme_id AS scheme_id, s.name AS name, s.level AS level, s.state AS state,
               s.primary_category AS primary_category, s.benefits AS benefits,
               s.official_link AS official_link, s.unmapped_requirements AS unmapped_requirements""",
                  ids=list(scheme_ids))
    by_id = {r["scheme_id"]: r for r in rows}
    return [by_id[i] for i in scheme_ids if i in by_id]


def scheme_details(db, scheme_id: str):
    """Everything stored on one scheme (or None if the ID does not exist)."""
    rows = db.run("""
        MATCH (s:Scheme {scheme_id: $sid})
        RETURN s.scheme_id AS scheme_id, s.name AS name, s.slug AS slug, s.official_link AS official_link,
               s.level AS level, s.state AS state, s.primary_category AS primary_category,
               s.categories AS categories, s.details AS details, s.benefits AS benefits,
               s.eligibility AS eligibility, s.application_mode AS application_mode,
               s.department AS department, s.tags AS tags,
               s.unmapped_requirements AS unmapped_requirements""", sid=scheme_id)
    return rows[0] if rows else None


def documents_for_schemes(db, scheme_ids: list) -> dict:
    """{scheme_id: [{'document': name, 'is_conditional': bool}, ...]} for the given schemes."""
    result = {sid: [] for sid in scheme_ids}
    if not scheme_ids:
        return result
    rows = db.run("""
        MATCH (s:Scheme)-[r:REQUIRES]->(d:Document) WHERE s.scheme_id IN $ids
        RETURN s.scheme_id AS scheme_id, d.name AS document, r.is_conditional AS is_conditional
        ORDER BY scheme_id, document""", ids=list(scheme_ids))
    for r in rows:
        result[r["scheme_id"]].append({"document": r["document"], "is_conditional": bool(r["is_conditional"])})
    return result


def scheme_department(db, scheme_id: str):
    """Department name, or None when the source data does not state one (nothing is guessed)."""
    rows = db.run("""
        MATCH (s:Scheme {scheme_id: $sid})
        OPTIONAL MATCH (s)-[:ADMINISTERED_BY]->(d:Department)
        RETURN d.name AS department""", sid=scheme_id)
    return rows[0]["department"] if rows else None


def scheme_application(db, scheme_id: str):
    """{'mode': 'online'|'offline'|'both'|'unclear', 'steps': [ordered step texts]} or None."""
    rows = db.run("""
        MATCH (s:Scheme {scheme_id: $sid})-[:FOLLOWS]->(a:ApplicationProcess)
        RETURN a.mode AS mode, a.steps AS steps""", sid=scheme_id)
    if not rows:
        return None
    return {"mode": rows[0]["mode"], "steps": list(rows[0]["steps"] or [])}


# =========================================================================================
# PLANNING
# =========================================================================================
def scheme_conflicts(db, scheme_ids: list) -> list:
    """Conflicting pairs where BOTH schemes are in `scheme_ids`. Each pair appears once (a < b)."""
    if len(scheme_ids) < 2:
        return []
    return db.run("""
        MATCH (a:Scheme)-[r:CONFLICTS_WITH]-(b:Scheme)
        WHERE a.scheme_id IN $ids AND b.scheme_id IN $ids AND a.scheme_id < b.scheme_id
        RETURN a.scheme_id AS scheme_a, b.scheme_id AS scheme_b, r.reason AS reason
        ORDER BY scheme_a, scheme_b""", ids=list(scheme_ids))


def conflicts_of(db, scheme_id: str) -> list:
    """Every scheme that conflicts with `scheme_id`, with the reason."""
    return db.run("""
        MATCH (a:Scheme {scheme_id: $sid})-[r:CONFLICTS_WITH]-(b:Scheme)
        RETURN b.scheme_id AS scheme_id, b.name AS name, r.reason AS reason
        ORDER BY scheme_id""", sid=scheme_id)


def shared_documents(db, scheme_ids: list) -> list:
    """Documents needed by two or more of the schemes (collect each once, use it many times)."""
    if len(scheme_ids) < 2:
        return []
    rows = db.run("""
        MATCH (s:Scheme)-[:REQUIRES]->(d:Document) WHERE s.scheme_id IN $ids
        WITH d, collect(s.scheme_id) AS schemes
        WHERE size(schemes) > 1
        RETURN d.name AS document, schemes, size(schemes) AS n
        ORDER BY n DESC, document""", ids=list(scheme_ids))
    return [{"document": r["document"], "schemes": sorted(r["schemes"])} for r in rows]


# =========================================================================================
# ADMIN
# =========================================================================================
def graph_stats(db) -> dict:
    nodes = {label: db.run(f"MATCH (n:{label}) RETURN count(n) AS n")[0]["n"] for label in NODE_LABELS}
    rels = {rel: db.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS n")[0]["n"] for rel in REL_TYPES}
    return {"nodes": nodes, "relationships": rels}
