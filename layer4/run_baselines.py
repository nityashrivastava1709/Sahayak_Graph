"""
run_baselines.py  -  ask both baselines the same question and compare their answers.

    python run_baselines.py "I am a widow in Uttar Pradesh with two children, what help can I get?" --state UP
    python run_baselines.py --profile my_profile.json            a structured profile (see Layer 2's demo_profiles.py)
    python run_baselines.py "..." --no-llm                       skip Gemini: the vector baseline shows its retrieval order only
    python run_baselines.py "..." --rerank                       also run vector RAG + the cross-encoder reranker
"""
import argparse
import json
import sys

import l4_common
import l4_config as config
import llm_client
from keyword_baseline import KeywordBaseline


def show(result):
    print("=" * 96)
    m = result["meta"]
    extra = []
    if m.get("llm_failed"):
        extra.append("LLM FAILED: retrieval order shown")
    if not m.get("llm_used", True) and result["system"] != "keyword":
        extra.append("retrieval only")
    if m.get("rerank"):
        extra.append(f"rerank used: {m['rerank']['used']}")
    print(f"{result['system'].upper()}   ({m['latency_s']:.2f}s{'; ' + '; '.join(extra) if extra else ''})")
    for s in result["schemes"]:
        docs = "" if s["documents"] is None else f"  docs: {len(s['documents'])}"
        grounded = "" if s["retrieved"] else "  [NOT in the retrieved context]"
        score = "  -  " if s["score"] is None else f"{s['score']:.3f}"
        print(f"  {s['rank']:>2}. {s['scheme_id']}  {score}  {s['name'][:62]}{docs}{grounded}")
        if s["reason"]:
            print(f"        why: {s['reason'][:130]}")
    if result["invalid_ids"]:
        print(f"  Named schemes that do not exist: {result['invalid_ids']}")
    if result["answer"]:
        print(f"\n  Answer: {result['answer'][:400]}")


def main():
    p = argparse.ArgumentParser(description="Run both baselines on one question.")
    p.add_argument("query", nargs="?")
    p.add_argument("--profile", help="JSON file with a structured profile (turned into a question)")
    p.add_argument("--state")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--rerank", action="store_true")
    a = p.parse_args()
    if a.profile:
        with open(a.profile, encoding="utf-8-sig") as f:
            profile = json.load(f)
        query, state = l4_common.profile_to_query(profile), a.state or l4_common.profile_state(profile)
    elif a.query:
        query, state = a.query, a.state
    else:
        raise SystemExit("Give a question or --profile file.json")
    print(f"QUESTION: {query}\n")
    try:
        show(KeywordBaseline().run(query, k=a.k, state=state))
        from vector_rag_baseline import VectorRAGBaseline
        use_llm = not a.no_llm
        if use_llm and not config.GEMINI_API_KEY:
            print("\n(no GEMINI_API_KEY found: the vector baseline will show its retrieval order only)")
            use_llm = False
        show(VectorRAGBaseline().run(query, k=a.k, state=state, use_llm=use_llm))
        if a.rerank:
            from reranker import CrossEncoderReranker, RerankerUnavailable
            try:
                show(VectorRAGBaseline(reranker=CrossEncoderReranker.load_active()).run(query, k=a.k, state=state, use_llm=use_llm))
            except RerankerUnavailable as e:
                print(f"\nReranker not available: {e}")
    except ValueError as e:
        raise SystemExit(f"Problem with the question: {e}")
    except llm_client.LLMFatalError as e:
        raise SystemExit(f"GEMINI PROBLEM: {e}")


if __name__ == "__main__":
    sys.exit(main())
