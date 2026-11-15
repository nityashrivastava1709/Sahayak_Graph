"""
discover.py  -  the graph half of Agent 2 (Discovery), in one function.

    from discover import graph_discover
    result = graph_discover(db, {"age": 34, "gender": "female", "state": "MP", ...})

It stores the profile, matches it against the graph, finds the schemes the person qualifies for
(and the ones that need more details), and attaches documents, department, application steps,
the graph evidence and conflicts. The result follows the handoff contract (section 6 of the plan);
`supporting_text`, `source_chunk_ids` and the final `match_score` are left empty / None because
they come from the vector store in Layer 3 and the Discovery agent in Layer 5.
"""
import queries


def _tidy(value):
    """12.0 -> 12 and ['female'] -> 'female', so the profile matches the handoff contract."""
    if isinstance(value, list):
        return value[0] if len(value) == 1 else value
    return int(value) if isinstance(value, float) and value == int(value) else value


def graph_discover(db, profile: dict, keep_person: bool = False) -> dict:
    reg = queries.register_person(db, profile)
    pid, clean = reg["person_id"], reg["clean_profile"]
    try:
        eligible = queries.eligible_schemes(db, pid)
        possible = queries.possible_schemes(db, pid, clean)
        eligible_ids = [s["scheme_id"] for s in eligible]
        all_ids = eligible_ids + [s["scheme_id"] for s in possible]

        summaries = {s["scheme_id"]: s for s in queries.scheme_summaries(db, all_ids)}
        documents = queries.documents_for_schemes(db, all_ids)

        matched = []
        for status, items in (("eligible", eligible), ("possible", possible)):
            for item in items:
                sid = item["scheme_id"]
                paths = queries.kg_paths(db, pid, sid)
                summary = summaries[sid]
                matched.append({
                    "scheme_id": sid,
                    "name": summary["name"],
                    "status": status,                              # eligible | possible (needs more details)
                    "match_score": None,                           # set by the Discovery agent (Layer 5)
                    "graph_score": 1.0 if status == "eligible" else item["graph_score"],
                    "level": summary["level"], "state": summary["state"],
                    "primary_category": summary["primary_category"],
                    "benefits": summary["benefits"], "official_link": summary["official_link"],
                    "kg_path": paths[0]["path"] if paths else ["Person", "QUALIFIES_FOR", sid],
                    "kg_paths": [p["path"] for p in paths],
                    "matched_rules": [{"group": p["group_id"], "rule": p["rule"],
                                       "source_sentence": p["source_sentence"]} for p in paths],
                    "supporting_text": [], "source_chunk_ids": [],  # filled by the RAG layer
                    "required_documents": [d["document"] for d in documents[sid]],
                    "conditional_documents": [d["document"] for d in documents[sid] if d["is_conditional"]],
                    "department": queries.scheme_department(db, sid),   # None = not stated in the data
                    "application": queries.scheme_application(db, sid),
                    "conflicts_with": [c["scheme_id"] for c in queries.conflicts_of(db, sid)],
                    "unverified_requirements": summary["unmapped_requirements"] or [],
                    "missing_attributes": item.get("missing_attributes", []),
                })

        return {
            "profile": {k: _tidy(v) for k, v in clean.items()},
            "profile_warnings": reg["warnings"],
            "matched_schemes": matched,
            "missing_attributes": sorted({a for s in possible for a in s["missing_attributes"]}),
            "conflicts_among_eligible": queries.scheme_conflicts(db, eligible_ids),
            "shared_documents": queries.shared_documents(db, eligible_ids),
        }
    finally:
        if not keep_person:
            queries.delete_person(db, pid)
