"""
load_graph.py  -  build the knowledge graph in Neo4j from the Layer 1 files.

Run:   python load_graph.py            rebuild the graph (recommended; safe to repeat)
       python load_graph.py --keep     do not delete anything first; only MERGE (add/update)

By default the six project labels (Scheme, EligibilityCondition, Document, ApplicationProcess,
Department, Person) are deleted and rebuilt, so the graph always matches the CSV files exactly.
Nodes with other labels in your Neo4j database are never touched.

Every write uses MERGE, so running it twice never creates duplicates.
Reads : data/processed/schemes_clean.csv, conditions.csv, documents.csv,
        application_steps.csv, conflicts.csv, attribute_vocabulary.json
Writes: data/processed/kg_load_report.json
"""
import argparse
import json
import sys

import data_io
import kg_config as config
import queries
from graph_db import KGConnectionError, connect
from schema import NODE_LABELS, create_schema

Q_SCHEMES = """
UNWIND $rows AS row
MERGE (s:Scheme {scheme_id: row.scheme_id})
SET s.name = row.name, s.slug = row.slug, s.official_link = row.official_link,
    s.level = row.level, s.state = row.state, s.primary_category = row.primary_category,
    s.categories = row.categories, s.details = row.details, s.benefits = row.benefits,
    s.eligibility = row.eligibility, s.application_mode = row.application_mode,
    s.department = row.department, s.tags = row.tags,
    s.unmapped_requirements = row.unmapped_requirements
"""

Q_CONDITIONS = """
UNWIND $rows AS row
MERGE (c:EligibilityCondition {condition_key: row.condition_key})
SET c.attribute = row.attribute, c.operator = row.operator, c.value = row.value
WITH c, row
MATCH (s:Scheme {scheme_id: row.scheme_id})
MERGE (c)-[q:QUALIFIES_FOR {group_id: row.group_id}]->(s)
SET q.source_sentence = row.source_sentence
"""

Q_DOCUMENTS = """
UNWIND $rows AS row
MATCH (s:Scheme {scheme_id: row.scheme_id})
MERGE (d:Document {name: row.document_name})
MERGE (s)-[r:REQUIRES]->(d)
SET r.is_conditional = row.is_conditional
"""

Q_PROCESSES = """
UNWIND $rows AS row
MATCH (s:Scheme {scheme_id: row.scheme_id})
MERGE (a:ApplicationProcess {process_id: row.scheme_id})
SET a.mode = row.mode, a.steps = row.steps, a.step_count = row.step_count
MERGE (s)-[:FOLLOWS]->(a)
"""

Q_DEPARTMENTS = """
UNWIND $rows AS row
MATCH (s:Scheme {scheme_id: row.scheme_id})
MERGE (d:Department {name: row.department})
MERGE (s)-[:ADMINISTERED_BY]->(d)
"""

Q_CONFLICTS = """
UNWIND $rows AS row
MATCH (a:Scheme {scheme_id: row.scheme_a})
MATCH (b:Scheme {scheme_id: row.scheme_b})
MERGE (a)-[r:CONFLICTS_WITH]->(b)
SET r.reason = row.reason
"""


def batches(rows, size):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def write_in_batches(db, query, rows):
    for chunk in batches(rows, config.BATCH_SIZE):
        db.run(query, rows=chunk)


def clear_graph(db) -> dict:
    """Delete the project's nodes (and their relationships). Other labels are left alone."""
    removed = {}
    for label in NODE_LABELS:
        n = db.run(f"MATCH (n:{label}) RETURN count(n) AS n")[0]["n"]
        if n:
            db.run(f"MATCH (n:{label}) DETACH DELETE n")
        removed[label] = n
    return removed


def build_graph(db, data=None, reset=True) -> dict:
    """Load everything. Returns a report dict (also useful from tests)."""
    data = data or data_io.load_layer1()
    removed = clear_graph(db) if reset else {}
    create_schema(db)

    write_in_batches(db, Q_SCHEMES, data.schemes)
    write_in_batches(db, Q_CONDITIONS, data.conditions)
    write_in_batches(db, Q_DOCUMENTS, data.documents)
    write_in_batches(db, Q_PROCESSES, [
        {"scheme_id": s["scheme_id"], "mode": s["application_mode"],
         "steps": data.steps.get(s["scheme_id"], []), "step_count": len(data.steps.get(s["scheme_id"], []))}
        for s in data.schemes])
    write_in_batches(db, Q_DEPARTMENTS, [
        {"scheme_id": s["scheme_id"], "department": s["department"]}
        for s in data.schemes if s["department"] != "Not stated"])
    if data.conflicts:
        write_in_batches(db, Q_CONFLICTS, data.conflicts)

    report = dict(data.report)
    report["deleted_before_load"] = removed
    report["graph_after_load"] = queries.graph_stats(db)
    return report


def main():
    parser = argparse.ArgumentParser(description="Build the SahayakGraph knowledge graph in Neo4j.")
    parser.add_argument("--keep", action="store_true",
                        help="do not delete the existing project graph first")
    args = parser.parse_args()

    try:
        data = data_io.load_layer1()
    except data_io.KGDataError as e:
        raise SystemExit(f"DATA PROBLEM: {e}")
    rep = data.report
    print(f"Read Layer 1 files: {rep['schemes']} schemes, {rep['condition_rows_loaded']} valid rules "
          f"({rep['distinct_conditions']} distinct), {rep['document_links']} document links, "
          f"{rep['conflict_pairs']} conflict pairs.")
    print(f"  not loaded as rules: {rep['unmapped_requirements']} unmapped requirements "
          f"(kept as text on the schemes), {rep['invalid_condition_rows_skipped']} invalid rows, "
          f"{rep['duplicate_condition_rows_dropped']} duplicates.")
    if rep["rejected_on_recheck"]:
        print(f"  WARNING: {len(rep['rejected_on_recheck'])} rule(s) failed the vocabulary re-check and "
              f"were skipped: {rep['rejected_on_recheck'][:3]}")

    try:
        db = connect()
    except KGConnectionError as e:
        raise SystemExit(f"CONNECTION PROBLEM: {e}")
    with db:
        print(f"Connected to Neo4j at {config.NEO4J_URI} (database '{config.NEO4J_DATABASE}').")
        if not args.keep:
            print("Rebuilding: existing SahayakGraph nodes will be replaced.")
        report = build_graph(db, data, reset=not args.keep)

    config.KG_LOAD_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    g = report["graph_after_load"]
    print("\nGraph built:")
    for label, n in g["nodes"].items():
        print(f"  {label:22} {n}")
    for rel, n in g["relationships"].items():
        print(f"  {rel:22} {n}")
    print(f"\nSaved {config.KG_LOAD_REPORT}")
    print("Next: python validate_graph.py")


if __name__ == "__main__":
    sys.exit(main())
