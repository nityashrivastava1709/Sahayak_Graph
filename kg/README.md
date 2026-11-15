# SahayakGraph - Layer 2 (Knowledge Graph)

Turns the Layer 1 data into a Neo4j knowledge graph and provides the query functions the agents
will use. Layer 2 reads Layer 1's files and never changes them.

## The graph

| Node | One per | Key |
|---|---|---|
| `Scheme` | scheme | `scheme_id` (S001...) |
| `EligibilityCondition` | distinct rule (shared by every scheme that uses it) | `condition_key`, e.g. `annual_income\|<=\|250000` |
| `Document` | standard document name | `name` |
| `ApplicationProcess` | scheme | `process_id` (= scheme_id): mode + ordered steps |
| `Department` | department (only when the data states one) | `name` |
| `Person` | user profile (created per query, deleted afterwards) | `person_id` |

| Relationship | Meaning |
|---|---|
| `(Person)-[:MATCHES]->(EligibilityCondition)` | the person satisfies this rule |
| `(EligibilityCondition)-[:QUALIFIES_FOR {group_id}]->(Scheme)` | the rule belongs to the scheme |
| `(Scheme)-[:REQUIRES {is_conditional}]->(Document)` | document needed |
| `(Scheme)-[:FOLLOWS]->(ApplicationProcess)` | how to apply |
| `(Scheme)-[:ADMINISTERED_BY]->(Department)` | issuing department |
| `(Scheme)-[:CONFLICTS_WITH {reason}]->(Scheme)` | overlapping schemes (read undirected) |

**Eligibility rule:** a scheme qualifies when EVERY group of its rules has at least one matched rule.
Rules in the same group are alternatives (OR), groups are combined with AND.
**Unknown is not "no":** if the profile lacks an attribute, the scheme is reported as *possible* and the
missing attribute is listed, so the system can ask a follow-up question instead of wrongly rejecting.
**Unmapped requirements** (Layer 1 could not express them) stay on the scheme as
`unmapped_requirements` and are shown as "check also"; they are never used for matching or dropped.

## Files

| File | Purpose |
|---|---|
| `kg_config.py` | paths, `.env` settings, state names |
| `graph_db.py` | the only file that talks to the Neo4j driver; friendly connection errors |
| `schema.py` | the graph design in one place + uniqueness constraints |
| `matching.py` | decides whether a profile satisfies a rule (true / false / unknown); cleans profiles |
| `data_io.py` | reads and validates the Layer 1 files |
| `load_graph.py` | builds the graph (MERGE, safe to repeat) |
| `validate_graph.py` | integrity checks (counts, orphans, documents, processes) |
| `queries.py` | the reusable query functions (see its header for the list) |
| `discover.py` | `graph_discover(db, profile)`: the graph half of the Discovery agent |
| `demo_profiles.py`, `run_demo.py` | hand-made profiles and a command-line demo |
| `reference.py` | independent plain-Python answer key used by the tests |
| `test_connection.py`, `test_matching.py`, `test_layer2.py` | the three tests |

## One-time setup

1. **Have a Neo4j database running.** Easiest: **Neo4j Aura Free** (no install): create an instance at
   console.neo4j.io and download the credentials file. Or use **Neo4j Desktop** (start the database until it shows Active).
2. **Add the connection lines to the root `.env`** (`D:\Sahayak_Graph\.env`; open it with `notepad` and save as UTF-8,
   do not append with PowerShell). See `.env.example`:
   `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`.
3. Install the driver.

```powershell
cd D:\Sahayak_Graph
.\.venv\Scripts\Activate.ps1
cd layer2
pip install -r requirements.txt
```

## Run order

```powershell
python test_matching.py       # unit tests, no database needed
python test_connection.py     # is Neo4j reachable?
python load_graph.py          # build the graph (rebuilds; safe to repeat)
python validate_graph.py      # integrity checks
python test_layer2.py         # graph vs independent answer key, 7 hand-made + 150 random profiles
python run_demo.py            # see real results for the hand-made profiles
```

Try your own profile: save `{"age": 34, "gender": "female", "state": "MP", "annual_income": 180000}` as
`my.json` and run `python run_demo.py --profile my.json --json result.json`.

## Using it from later layers

```python
from graph_db import connect
from discover import graph_discover
with connect() as db:
    result = graph_discover(db, profile)       # contract-shaped dict, JSON-serialisable
```
`supporting_text`, `source_chunk_ids` and `match_score` are left empty / None: they come from Layer 3 and Layer 5.

## If something goes wrong

- **Cannot reach Neo4j:** the database is not started, or the URI is wrong (try `bolt://127.0.0.1:7687`; Aura uses `neo4j+s://`).
- **Authentication failed:** check `NEO4J_USER` / `NEO4J_PASSWORD` in `.env`.
- **Database not found:** set `NEO4J_DATABASE` to your database's name.
- **Validator shows FAIL after changing Layer 1:** run `python load_graph.py` again.
- **WARN lines** are limits of the source data (schemes with no mappable rule, departments "Not stated"), not errors.
