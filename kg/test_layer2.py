"""
test_layer2.py  -  proves the knowledge graph answers correctly.

Run (after  python load_graph.py):   python test_layer2.py

It compares what Neo4j returns with an independent plain-Python answer key (reference.py)
computed from the same Layer 1 files:
  * the 7 hand-made profiles in demo_profiles.py
  * 150 random profiles (seeded, so repeatable), including values exactly on rule boundaries
    and profiles with many unknown attributes
and checks documents, graph evidence, conflicts and shared documents. Nothing is hard-coded to
your data, so it keeps working when Layer 1 data changes.
"""
import json
import random
import sys

import data_io
import demo_profiles
import graph_db
import kg_config as config
import matching
import queries
import reference
import validate_graph
from discover import graph_discover

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail and not ok else ""))


def make_random_profiles(data, n, seed):
    rng = random.Random(seed)
    vocab = matching.load_vocabulary()
    numbers = {a: [] for a, s in vocab.items() if s["type"] == "number"}
    for c in data.conditions:
        if c["attribute"] in numbers:
            numbers[c["attribute"]] += [matching.parse_number(p) for p in c["value"].split("|")]
    profiles = []
    for _ in range(n):
        p = {}
        for attr, spec in vocab.items():
            if rng.random() > 0.7:                       # leave ~30% of attributes unknown
                continue
            kind = spec["type"]
            if kind == "number":
                pool = [v for v in numbers[attr] if v is not None]
                base = rng.choice(pool) if pool else rng.choice([5, 18, 30, 60, 80])
                p[attr] = max(0, base + rng.choice([-1, 0, 0, 1, rng.randint(-20, 20)]))   # boundaries matter
            elif kind == "yesno":
                p[attr] = rng.choice(["yes", "no"])
            elif kind == "state":
                p[attr] = rng.choice(config.INDIAN_STATES_AND_UTS)
            else:
                values = spec["values"]
                p[attr] = rng.sample(values, 2) if (attr == "social_category" and rng.random() < 0.2) \
                    else rng.choice(values)
        profiles.append(p)
    return profiles


def compare(db, data, profile, label):
    """Run one profile through Neo4j and through the reference; return a list of problems."""
    problems = []
    reg = queries.register_person(db, profile)
    pid, clean = reg["person_id"], reg["clean_profile"]
    try:
        want_eligible, want_possible = reference.reference_results(data, clean)
        got_eligible = {s["scheme_id"] for s in queries.eligible_schemes(db, pid)}
        got_possible = {s["scheme_id"]: set(s["missing_attributes"])
                        for s in queries.possible_schemes(db, pid, clean)}
        if got_eligible != want_eligible:
            problems.append(f"eligible differs: only in Neo4j {sorted(got_eligible - want_eligible)}, "
                            f"only in reference {sorted(want_eligible - got_eligible)}")
        if got_possible != want_possible:
            problems.append(f"possible differs: Neo4j {sorted(got_possible)} vs reference {sorted(want_possible)}")
        groups = reference.reference_groups(data)
        for sid in got_eligible & want_eligible:                    # the evidence must cover every group
            covered = {p["group_id"] for p in queries.kg_paths(db, pid, sid)}
            if covered != groups[sid]:
                problems.append(f"{sid}: graph evidence covers groups {sorted(covered)}, scheme has {sorted(groups[sid])}")
                break
    finally:
        queries.delete_person(db, pid)
    return problems


def main():
    try:
        data = data_io.load_layer1()
    except data_io.KGDataError as e:
        raise SystemExit(f"DATA PROBLEM: {e}")
    try:
        db = graph_db.connect()
    except graph_db.KGConnectionError as e:
        raise SystemExit(f"CONNECTION PROBLEM: {e}")

    with db:
        stats = queries.graph_stats(db)
        loaded = stats["nodes"]["Scheme"] == data.report["schemes"]
        check(f"graph is loaded ({stats['nodes']['Scheme']} schemes in Neo4j, {data.report['schemes']} in the files)", loaded)
        if not loaded:
            raise SystemExit("Run  python load_graph.py  first, then run this test again.")

        checks = validate_graph.run_checks(db, data)
        failed_checks = [t for s, t in checks if s == "FAIL"]
        check("validate_graph.py has no FAIL", not failed_checks, "; ".join(failed_checks))

        # ---------------- hand-made profiles ----------------
        print()
        for name, profile in demo_profiles.PROFILES.items():
            problems = compare(db, data, profile, name)
            check(f"hand-made profile '{name}': Neo4j == reference", not problems, "; ".join(problems))
            result = graph_discover(db, profile)
            eligible = [s for s in result["matched_schemes"] if s["status"] == "eligible"]
            check(f"   '{name}': every eligible scheme has documents, evidence and an application process",
                  all(s["required_documents"] and s["kg_paths"] and s["application"] for s in eligible),
                  f"{len(eligible)} eligible schemes")
            print(f"        -> qualifies for {len(eligible)}, could qualify for "
                  f"{len(result['matched_schemes']) - len(eligible)} more")
            json.dumps(result)                                       # must be JSON-serialisable
        # the example profile from the handoff contract (uses 'income' and a state code)
        result = graph_discover(db, {"age": 34, "income": 180000, "state": "MP", "occupation": "farmer",
                                     "gender": "female"})
        check("contract example profile (income, state code 'MP') is understood",
              not result["profile_warnings"] and result["profile"].get("annual_income") == 180000
              and result["profile"].get("state") == "Madhya Pradesh", str(result["profile_warnings"]))

        # ---------------- random profiles ----------------
        print()
        n = 150
        bad = 0
        for i, profile in enumerate(make_random_profiles(data, n, config.SEED)):
            problems = compare(db, data, profile, f"random {i}")
            if problems:
                bad += 1
                if bad <= 3:
                    print(f"   random profile {i}: {profile}\n      {problems}")
        check(f"{n} random profiles: Neo4j == reference for eligible, possible and graph evidence", bad == 0,
              f"{bad} profile(s) differ")

        # ---------------- facts, conflicts, shared documents ----------------
        print()
        ids = [s["scheme_id"] for s in data.schemes]
        docs = queries.documents_for_schemes(db, ids)
        want_docs = {}
        for d in data.documents:
            want_docs.setdefault(d["scheme_id"], set()).add(d["document_name"])
        check("documents_for_schemes matches documents.csv for every scheme",
              all({x["document"] for x in docs[sid]} == want_docs.get(sid, set()) for sid in ids))
        steps_ok = all((queries.scheme_application(db, sid) or {}).get("steps") == data.steps.get(sid, [])
                       for sid in ids[:30])
        check("application steps keep their order (first 30 schemes)", steps_ok)
        dept_ok = all((queries.scheme_department(db, s["scheme_id"]) or "Not stated") == s["department"]
                      for s in data.schemes)
        check("departments match (None when the data says 'Not stated')", dept_ok)

        pairs_ok = True
        for c in data.conflicts:
            found = queries.scheme_conflicts(db, [c["scheme_a"], c["scheme_b"]])
            other = [x["scheme_id"] for x in queries.conflicts_of(db, c["scheme_b"])]
            pairs_ok &= len(found) == 1 and c["scheme_a"] in other
        check(f"all {len(data.conflicts)} conflict pair(s) are found, once each, from both sides", pairs_ok)

        sample = ids[:12]
        shared = queries.shared_documents(db, sample)
        counts = {}
        for sid in sample:
            for name in want_docs.get(sid, set()):
                counts[name] = counts.get(name, 0) + 1
        want_shared = {n for n, k in counts.items() if k > 1}
        check("shared_documents finds exactly the documents needed by 2+ schemes",
              {s["document"] for s in shared} == want_shared)

        check("a person node leaves nothing behind",
              db.run("MATCH (p:Person) RETURN count(p) AS n")[0]["n"] == 0)

    print(f"\n{sum(results)}/{len(results)} checks passed")
    if all(results):
        print("Layer 2 is complete: the graph returns the correct schemes and their documents.")
        return 0
    print("Some checks failed. If you just changed Layer 1 data, rebuild with  python load_graph.py.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
