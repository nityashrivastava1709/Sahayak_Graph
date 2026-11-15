"""
validate_graph.py  -  integrity checks on the graph that is in Neo4j.

Run:  python validate_graph.py

PASS = requirement met
WARN = a known limit of the source data (listed, nothing invented to hide it)
FAIL = something is wrong with the graph; rebuild it or fix the Layer 1 data

Checks: counts match the Layer 1 files, no orphan nodes, every scheme has a document and an
application process, and every scheme with usable rules has at least one condition.
"""
import sys

import data_io
import kg_config as config
import queries
from graph_db import KGConnectionError, connect

SCHEME_LACKS = {   # relationship pattern -> schemes that have none
    "QUALIFIES_FOR": "(s)<-[r:QUALIFIES_FOR]-()",
    "REQUIRES": "(s)-[r:REQUIRES]->()",
    "FOLLOWS": "(s)-[r:FOLLOWS]->()",
    "ADMINISTERED_BY": "(s)-[r:ADMINISTERED_BY]->()",
}


def schemes_without(db, rel_name):
    pattern = SCHEME_LACKS[rel_name]
    rows = db.run(f"""
        MATCH (s:Scheme)
        OPTIONAL MATCH {pattern}
        WITH s, count(r) AS n
        WHERE n = 0
        RETURN s.scheme_id AS scheme_id
        ORDER BY scheme_id""")
    return [r["scheme_id"] for r in rows]


def orphans(db, labels):
    rows = db.run("""
        MATCH (n)
        WHERE any(l IN labels(n) WHERE l IN $labels)
        OPTIONAL MATCH (n)-[r]-()
        WITH n, count(r) AS degree
        WHERE degree = 0
        RETURN labels(n)[0] AS label, count(n) AS orphans""", labels=labels)
    return {r["label"]: r["orphans"] for r in rows}


def run_checks(db, data) -> list:
    """Return a list of (status, text). Used by this script and by the tests."""
    out = []
    add = lambda ok, text, warn=False: out.append(("PASS" if ok else ("WARN" if warn else "FAIL"), text))
    stats = queries.graph_stats(db)
    nodes, rels = stats["nodes"], stats["relationships"]
    rep = data.report

    expected_nodes = {"Scheme": rep["schemes"], "EligibilityCondition": rep["distinct_conditions"],
                      "Document": rep["distinct_documents"], "ApplicationProcess": rep["schemes"],
                      "Department": rep["distinct_departments"]}
    for label, want in expected_nodes.items():
        add(nodes[label] == want, f"{label} nodes: {nodes[label]} in the graph, {want} expected from the files")
    expected_rels = {"QUALIFIES_FOR": rep["condition_rows_loaded"], "REQUIRES": rep["document_links"],
                     "FOLLOWS": rep["schemes"], "ADMINISTERED_BY": rep["schemes_with_department"],
                     "CONFLICTS_WITH": rep["conflict_pairs"]}
    for rel, want in expected_rels.items():
        add(rels[rel] == want, f"{rel} relationships: {rels[rel]} in the graph, {want} expected")

    orph = orphans(db, ["EligibilityCondition", "Document", "Department", "ApplicationProcess"])
    add(not orph, f"no orphan nodes (EligibilityCondition, Document, Department, ApplicationProcess){'' if not orph else ': ' + str(orph)}")

    no_doc = schemes_without(db, "REQUIRES")
    add(not no_doc, f"every scheme requires at least one document{'' if not no_doc else ': ' + str(no_doc)}")
    no_proc = schemes_without(db, "FOLLOWS")
    add(not no_proc, f"every scheme has an application process{'' if not no_proc else ': ' + str(no_proc)}")
    no_cond = schemes_without(db, "QUALIFIES_FOR")
    add(not no_cond, f"every scheme has at least one eligibility condition ({len(no_cond)} have none because "
                     f"their text names nothing in the vocabulary: {no_cond}; they cannot be matched automatically)",
        warn=True)
    no_dept = schemes_without(db, "ADMINISTERED_BY")
    add(not no_dept, f"every scheme has a department ({len(no_dept)} 'Not stated': {no_dept})", warn=True)

    # every state-level scheme should be tied to its state by a rule
    state_rows = db.run("""
        MATCH (s:Scheme) WHERE toLower(s.level) = 'state'
        OPTIONAL MATCH (c:EligibilityCondition {attribute: 'state'})-[:QUALIFIES_FOR]->(s)
        WITH s, count(c) AS n WHERE n = 0
        RETURN s.scheme_id AS scheme_id ORDER BY scheme_id""")
    missing_state = [r["scheme_id"] for r in state_rows]
    add(not missing_state, f"every state-level scheme has a state condition"
                           f"{'' if not missing_state else ': ' + str(missing_state)}")
    return out


def main():
    try:
        data = data_io.load_layer1()
    except data_io.KGDataError as e:
        raise SystemExit(f"DATA PROBLEM: {e}")
    try:
        db = connect()
    except KGConnectionError as e:
        raise SystemExit(f"CONNECTION PROBLEM: {e}")
    with db:
        checks = run_checks(db, data)
    for status, text in checks:
        print(f"{status}  {text}")
    statuses = [s for s, _ in checks]
    print(f"\n{statuses.count('PASS')} passed, {statuses.count('WARN')} warnings, {statuses.count('FAIL')} failed")
    if statuses.count("FAIL"):
        print("Rebuild with  python load_graph.py  and run this again.")
        return 1
    print("The graph is consistent with the Layer 1 files. Next: python test_layer2.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
