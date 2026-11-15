"""
schema.py  -  the shape of the knowledge graph, written down once.

NODES                      KEY PROPERTY                  WHAT IT IS
  Scheme                   scheme_id  (S001 ...)         a welfare scheme (+ name, state, benefits ...)
  EligibilityCondition     condition_key                 one rule, SHARED by every scheme that uses it,
                                                         e.g. "annual_income|<=|250000"
  Document                 name                          one standard document, shared across schemes
  ApplicationProcess       process_id (= scheme_id)      application mode + ordered steps
  Department               name                          issuing department (only when stated)
  Person                   person_id                     a user profile (created at query time)

RELATIONSHIPS
  (Person)-[:MATCHES]->(EligibilityCondition)            the person satisfies this rule
  (EligibilityCondition)-[:QUALIFIES_FOR {group_id}]->(Scheme)
                                                         the rule is part of the scheme's eligibility.
                                                         Rules with the SAME group_id are alternatives
                                                         (OR); a scheme needs ONE match in EVERY group.
  (Scheme)-[:REQUIRES {is_conditional}]->(Document)
  (Scheme)-[:FOLLOWS]->(ApplicationProcess)
  (Scheme)-[:ADMINISTERED_BY]->(Department)
  (Scheme)-[:CONFLICTS_WITH {reason}]->(Scheme)          stored once per pair; read it undirected

Eligibility rule of thumb:  scheme qualifies  <=>  every group has at least one matched condition.
Requirements the vocabulary cannot express are kept on the Scheme node as `unmapped_requirements`
(they are never silently dropped, and never used for matching).
"""
NODE_LABELS = ["Scheme", "EligibilityCondition", "Document", "ApplicationProcess", "Department", "Person"]
REL_TYPES = ["MATCHES", "QUALIFIES_FOR", "REQUIRES", "FOLLOWS", "ADMINISTERED_BY", "CONFLICTS_WITH"]

# Labels that are rebuilt from the Layer 1 files (Person nodes are created per query)
KG_LABELS = ["Scheme", "EligibilityCondition", "Document", "ApplicationProcess", "Department"]

# (constraint name, label, property) - uniqueness also creates the index MERGE needs to be fast
CONSTRAINTS = [
    ("kg_scheme_id", "Scheme", "scheme_id"),
    ("kg_condition_key", "EligibilityCondition", "condition_key"),
    ("kg_document_name", "Document", "name"),
    ("kg_department_name", "Department", "name"),
    ("kg_process_id", "ApplicationProcess", "process_id"),
    ("kg_person_id", "Person", "person_id"),
]


def create_schema(db):
    """Create the uniqueness constraints (safe to run again)."""
    for name, label, prop in CONSTRAINTS:
        db.run(f"CREATE CONSTRAINT {name} IF NOT EXISTS FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE")
