"""
benchmark_latency.py  -  how long does reranking take, and does the fallback work?

Run:  python benchmark_latency.py

Times the active reranker on RERANK_TOP_N candidates (default 20, the number Discovery will rerank) and on 50, reports
mean / median / 95th percentile, and checks the 95th percentile against MAX_LATENCY_SECONDS (default 3 s).
Then proves that a broken or missing model never breaks the pipeline: safe_rerank falls back to the original order.
If it is too slow: rerank fewer candidates (RERANK_TOP_N), set RERANKER_MAX_LENGTH=384, use a GPU, or skip reranking.
"""
import json
import random
import statistics
import sys
import time

import l4_config as config
import rerank_data as rd
from reranker import CrossEncoderReranker, RerankerUnavailable, active_info, safe_rerank


def percentile(values, q):
    s = sorted(values)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]


def candidate_pool(items):
    return [c for it in items for c in it["candidates"]]


def main():
    items = rd.read_jsonl(config.EVAL_TEST) or rd.read_jsonl(config.EVAL_VAL)
    if not items:
        raise SystemExit("No evaluation questions found. Run  python make_training_data.py  first.")
    info = active_info()
    try:
        reranker = CrossEncoderReranker.load_active()
    except RerankerUnavailable as e:
        raise SystemExit(f"RERANKER PROBLEM: {e}")
    print(f"Model: {reranker.model_path} ({info['active']}) | device: {reranker.device} | max length: {reranker.max_length}")
    rng = random.Random(config.SEED)
    pool = candidate_pool(items)
    report = {"model": reranker.model_path, "device": reranker.device, "limit_s": config.MAX_LATENCY_SECONDS, "runs": {}}
    ok = True
    for n in sorted({config.RERANK_TOP_N, 50}):
        for _ in range(2):                                                    # warm-up (first calls are slower)
            reranker.rerank(items[0]["query"], rng.sample(pool, min(n, len(pool))))
        times = []
        for item in rng.sample(items, min(15, len(items))):
            cands = rng.sample(pool, min(n, len(pool)))
            started = time.perf_counter()
            reranker.rerank(item["query"], cands)
            times.append(time.perf_counter() - started)
        stats = {"candidates": min(n, len(pool)), "mean_s": round(statistics.mean(times), 3), "median_s": round(statistics.median(times), 3),
                 "p95_s": round(percentile(times, 0.95), 3), "max_s": round(max(times), 3),
                 "ms_per_candidate": round(1000 * statistics.mean(times) / min(n, len(pool)), 1)}
        report["runs"][str(n)] = stats
        within = stats["p95_s"] <= config.MAX_LATENCY_SECONDS
        if n == config.RERANK_TOP_N:
            ok = within
        print(f"{stats['candidates']:>3} candidates: mean {stats['mean_s']:.2f}s | median {stats['median_s']:.2f}s | p95 {stats['p95_s']:.2f}s | "
              f"max {stats['max_s']:.2f}s | {stats['ms_per_candidate']} ms per candidate  "
              f"{'[within the ' + str(config.MAX_LATENCY_SECONDS) + 's budget]' if within else '[OVER the ' + str(config.MAX_LATENCY_SECONDS) + 's budget]'}")

    # ---- the fallback must never break the pipeline ----
    class Broken:
        model_path = "broken"

        def rerank(self, *a, **k):
            raise RuntimeError("simulated model failure")

    sample = items[0]["candidates"][:5]
    fallback_ok = True
    for label, rr in (("no reranker", None), ("broken reranker", Broken())):
        out, finfo = safe_rerank(rr, items[0]["query"], sample)
        good = (not finfo["used"]) and [c["chunk_id"] for c in out] == [c["chunk_id"] for c in sample] and all(c["rerank_score"] is None for c in out)
        fallback_ok &= good
        print(f"fallback with {label}: {'returned the original order (' + str(finfo['reason'])[:60] + ')' if good else 'FAILED'}")
    report["within_budget"], report["fallback_ok"] = ok, fallback_ok
    config.LATENCY_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {config.LATENCY_REPORT.name}")
    if not ok:
        print(f"Reranking {config.RERANK_TOP_N} candidates is slower than {config.MAX_LATENCY_SECONDS}s. Options: lower RERANK_TOP_N, "
              "set RERANKER_MAX_LENGTH=384, use a GPU, or skip reranking (the agents can run without it).")
    return 0 if (ok and fallback_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
