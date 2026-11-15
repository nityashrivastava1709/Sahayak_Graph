"""
reference.py  -  an independent, database-free answer key used by the tests.

It applies the eligibility rule directly to the cleaned Layer 1 data in plain Python:
    scheme qualifies   <=>  every group has at least one satisfied condition
    scheme is possible <=>  no group is contradicted, and at least one group could still be
                            satisfied by an attribute the profile does not contain
test_layer2.py compares the Neo4j answers with these.
"""
import matching


def reference_results(data, clean_profile: dict):
    """Returns (eligible_ids: set, possible: {scheme_id: set(missing attributes)})."""
    groups = {}
    for c in data.conditions:
        groups.setdefault(c["scheme_id"], {}).setdefault(c["group_id"], []).append(c)
    eligible, possible = set(), {}
    for sid, group_map in groups.items():
        open_groups, violated, missing = 0, 0, set()
        for conds in group_map.values():
            verdicts = [matching.evaluate_condition(clean_profile, c["attribute"], c["operator"], c["value"])
                        for c in conds]
            if any(v is True for v in verdicts):
                continue
            if any(v is None for v in verdicts):
                open_groups += 1
                missing.update(c["attribute"] for c, v in zip(conds, verdicts) if v is None)
            else:
                violated += 1
        if violated == 0 and open_groups == 0:
            eligible.add(sid)
        elif violated == 0:
            possible[sid] = missing
    return eligible, possible


def reference_groups(data) -> dict:
    """{scheme_id: set(group ids)} - used to check that the graph evidence covers every group."""
    out = {}
    for c in data.conditions:
        out.setdefault(c["scheme_id"], set()).add(c["group_id"])
    return out
