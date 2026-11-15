"""
run_demo.py  -  try the graph with hand-made or your own profiles.

    python run_demo.py                          all hand-made profiles in demo_profiles.py
    python run_demo.py --name farmer_mp         one of them
    python run_demo.py --profile my.json        your own profile (a JSON file of attribute: value)
    python run_demo.py --name farmer_mp --json out.json     also save the full result

For each profile it prints the schemes the person qualifies for, the schemes that need more
details, and for every scheme: documents, department, how to apply, the graph evidence and conflicts.
"""
import argparse
import json
import sys

import demo_profiles
from discover import graph_discover
from graph_db import KGConnectionError, connect


def show(name, profile, result, max_schemes=8):
    print("=" * 92)
    print(f"PROFILE: {name}")
    print("  " + ", ".join(f"{k}={v}" for k, v in result["profile"].items()))
    for w in result["profile_warnings"]:
        print("  WARNING:", w)
    eligible = [s for s in result["matched_schemes"] if s["status"] == "eligible"]
    possible = [s for s in result["matched_schemes"] if s["status"] == "possible"]

    print(f"\n  QUALIFIES FOR {len(eligible)} scheme(s):")
    for s in eligible[:max_schemes]:
        print(f"   {s['scheme_id']}  {s['name'][:70]}   [{s['state']}]")
        print(f"        documents : {', '.join(s['required_documents']) or '(none listed)'}")
        print(f"        department: {s['department'] or 'not stated in the data'}   |   apply: "
              f"{(s['application'] or {}).get('mode', '?')}")
        if s["matched_rules"]:
            print(f"        because   : {'; '.join(r['rule'] for r in s['matched_rules'])}")
        if s["unverified_requirements"]:
            print(f"        check also: {len(s['unverified_requirements'])} requirement(s) the graph cannot verify, "
                  f"e.g. {s['unverified_requirements'][0][:80]!r}")
        if s["conflicts_with"]:
            print(f"        conflicts : {', '.join(s['conflicts_with'])}")
    if len(eligible) > max_schemes:
        print(f"   ... and {len(eligible) - max_schemes} more")

    print(f"\n  COULD QUALIFY for {len(possible)} more if these details are provided: "
          f"{', '.join(result['missing_attributes']) or '-'}")
    for s in possible[:3]:
        print(f"   {s['scheme_id']}  {s['name'][:60]}  needs: {', '.join(s['missing_attributes'])}")
    if result["conflicts_among_eligible"]:
        print("\n  CONFLICTS among the eligible schemes:")
        for c in result["conflicts_among_eligible"]:
            print(f"   {c['scheme_a']} <-> {c['scheme_b']}: {c['reason']}")
    if result["shared_documents"]:
        top = result["shared_documents"][:4]
        print("\n  DOCUMENTS used by several eligible schemes: "
              + "; ".join(f"{d['document']} ({len(d['schemes'])})" for d in top))
    print()


def main():
    parser = argparse.ArgumentParser(description="Query the SahayakGraph knowledge graph with a profile.")
    parser.add_argument("--name", help="one of the profiles in demo_profiles.py")
    parser.add_argument("--profile", help="path to a JSON file with your own profile")
    parser.add_argument("--json", help="save the full result(s) to this JSON file")
    args = parser.parse_args()

    if args.profile:
        with open(args.profile, encoding="utf-8-sig") as f:
            profiles = {args.profile: json.load(f)}
    elif args.name:
        if args.name not in demo_profiles.PROFILES:
            raise SystemExit(f"Unknown name. Choose from: {', '.join(demo_profiles.PROFILES)}")
        profiles = {args.name: demo_profiles.PROFILES[args.name]}
    else:
        profiles = demo_profiles.PROFILES

    try:
        db = connect()
    except KGConnectionError as e:
        raise SystemExit(f"CONNECTION PROBLEM: {e}")
    saved = {}
    with db:
        for name, profile in profiles.items():
            result = graph_discover(db, profile)
            saved[name] = result
            show(name, profile, result)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(saved, f, indent=2, ensure_ascii=False)
        print(f"Saved {args.json}")


if __name__ == "__main__":
    sys.exit(main())
