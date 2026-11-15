"""
test_connection.py  -  run this BEFORE loading the graph.

It checks, in order:
  1. Settings   : .env found, Neo4j URI / user / database, password present (its value is never shown)
  2. Network    : DNS -> port reachable -> encrypted handshake (tells you WHERE it breaks)
  3. Connection : Neo4j answers and accepts the login
  4. Read/write : a temporary test node is created, read back and deleted

Run:  python test_connection.py
"""
import sys
import uuid

import kg_config as config
import graph_db
import net_check

results = []


def report(name, status, detail=""):
    print(f"[{status}] {name}" + (f"  -  {detail}" if detail else ""))
    results.append(status)


print("=" * 70)
print("1. SETTINGS")
print("=" * 70)
print(f"Project root : {config.ROOT}")
print(f".env file    : {config.ENV_FILE}  ({'found' if config.ENV_FILE.exists() else 'NOT FOUND'})")
print(f"NEO4J_URI    : {config.NEO4J_URI}")
print(f"NEO4J_USER   : {config.NEO4J_USER}")
print(f"NEO4J_DATABASE: {config.NEO4J_DATABASE}")
if config.ENV_PROBLEM:
    report(".env readable", "FAIL", config.ENV_PROBLEM)
    sys.exit(1)
if not config.NEO4J_PASSWORD:
    report("NEO4J_PASSWORD loaded", "FAIL", f"add NEO4J_PASSWORD=... to {config.ENV_FILE}")
    sys.exit(1)
report("NEO4J_PASSWORD loaded", "PASS", f"{len(config.NEO4J_PASSWORD)} characters (value not shown)")

print("\n" + "=" * 70)
print("2. NETWORK (where does the connection break, if it does?)")
print("=" * 70)
stages = net_check.diagnose(config.NEO4J_URI)
for r in stages:
    report(f"{r['stage']}", "PASS" if r["ok"] else "FAIL", r["detail"])
    if not r["ok"]:
        print(f"\n   WHAT TO DO: {r['advice']}")
if not all(r["ok"] for r in stages):
    print("\nFix the failing stage above, then run this test again.")
    sys.exit(1)

print("\n" + "=" * 70)
print("3. CONNECTION")
print("=" * 70)
try:
    db = graph_db.connect()
except graph_db.KGConnectionError as e:
    report("connection", "FAIL", str(e))
    sys.exit(1)
report("connection", "PASS", "Neo4j answered and accepted the login")

with db:
    try:
        info = db.run("CALL dbms.components() YIELD name, versions, edition "
                      "RETURN name, versions[0] AS version, edition")
        print(f"Server       : {info[0]['name']} {info[0]['version']} ({info[0]['edition']})")
    except Exception:
        print("Server       : (version not available with this account; not a problem)")

    print("\n" + "=" * 70)
    print("4. READ / WRITE")
    print("=" * 70)
    tag = uuid.uuid4().hex[:10]
    try:
        db.run("CREATE (n:ConnectionTest {tag: $tag, note: 'created by test_connection.py'})", tag=tag)
        rows = db.run("MATCH (n:ConnectionTest {tag: $tag}) RETURN n.note AS note", tag=tag)
        ok = len(rows) == 1
        db.run("MATCH (n:ConnectionTest {tag: $tag}) DETACH DELETE n", tag=tag)
        left = db.run("MATCH (n:ConnectionTest {tag: $tag}) RETURN count(n) AS n", tag=tag)[0]["n"]
        report("create, read and delete a test node", "PASS" if ok and left == 0 else "FAIL",
               "round trip worked" if ok and left == 0 else "the test node was not read back / not removed")
    except Exception as e:
        report("create, read and delete a test node", "FAIL", f"{type(e).__name__}: {str(e)[:300]}")

print("\n" + "=" * 70)
fails = results.count("FAIL")
print(f"{results.count('PASS')} passed, {fails} failed")
if fails:
    sys.exit(1)
print("Neo4j is ready. Next: python load_graph.py")