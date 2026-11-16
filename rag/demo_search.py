"""
demo_search.py  -  try the search from the command line.

    python demo_search.py "pension for widows"
    python demo_search.py "scholarship for girls" --state Rajasthan --k 5
    python demo_search.py "documents needed" --section documents --category "Agriculture,Rural & Environment"
    python demo_search.py "fellowship for women scientists" --schemes          one line per scheme
    python demo_search.py "water conservation award" --context                the text block an LLM would receive

Filters: --state (name or code like MP; Central schemes are included unless --no-central),
         --level Central|State, --category, --section overview|benefits|eligibility|application|documents
"""
import argparse
import sys

from chunker import SECTION_NAMES
from retriever import Retriever, format_context
from vector_store import IndexMissingError


def main():
    p = argparse.ArgumentParser(description="Search the SahayakGraph vector index.")
    p.add_argument("query")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--state")
    p.add_argument("--no-central", action="store_true", help="with --state: leave out Central schemes")
    p.add_argument("--level", choices=["Central", "State"])
    p.add_argument("--category", action="append", help="repeat for several categories")
    p.add_argument("--section", action="append", choices=SECTION_NAMES, help="repeat for several sections")
    p.add_argument("--schemes", action="store_true", help="group by scheme")
    p.add_argument("--context", action="store_true", help="print the LLM context block")
    a = p.parse_args()

    try:
        r = Retriever()
    except IndexMissingError as e:
        raise SystemExit(str(e))
    filters = dict(state=a.state, include_central=not a.no_central, level=a.level,
                   categories=a.category, sections=a.section)
    try:
        if a.schemes:
            results = r.search_schemes(a.query, k=a.k, **filters)
            for s in results:
                print(f"{s['score']:.3f}  {s['scheme_id']}  {s['scheme_name'][:60]}  [{s['state']}]  {s['primary_category']}")
                print(f"        {s['official_link']}")
            return
        hits = r.search(a.query, k=a.k, **filters)
    except ValueError as e:
        raise SystemExit(f"Problem with the search: {e}")
    if a.context:
        print(format_context(hits, max_chars=3000))
        return
    for h in hits:
        print(f"{h['score']:.3f}  {h['chunk_id']:9} {h['scheme_name'][:50]}  [{h['state']}]  ({', '.join(h['sections'])})")
        print(f"        {h['text'][:170].replace(chr(10), ' ')}...")
    if not hits:
        print("No results for these filters.")


if __name__ == "__main__":
    sys.exit(main())
