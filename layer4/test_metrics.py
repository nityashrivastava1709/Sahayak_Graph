"""
test_metrics.py  -  tests for the ranking metrics. No model, data or internet needed.

Run:  python test_metrics.py
"""
import math
import sys

import rerank_metrics as m

passed = failed = 0


def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
    else:
        failed += 1
        print(f"FAIL  {name}   {detail}")


close = lambda a, b: abs(a - b) < 1e-9

# ---- single ranked lists
check("reciprocal rank: first relevant at position 3", close(m.reciprocal_rank([0, 0, 1, 0, 1]), 1 / 3))
check("reciprocal rank: relevant first", close(m.reciprocal_rank([1, 0, 0]), 1.0))
check("reciprocal rank: none relevant", m.reciprocal_rank([0, 0, 0]) == 0.0 and m.reciprocal_rank([]) == 0.0)
check("hit@k", m.hit_at_k([0, 0, 1], 2) == 0.0 and m.hit_at_k([0, 0, 1], 3) == 1.0 and m.hit_at_k([0, 0, 0], 5) == 0.0)
check("recall@k counts all relevant items", close(m.recall_at_k([0, 1, 0, 1], 2), 0.5) and close(m.recall_at_k([1, 1], 1), 0.5))
check("recall@k with nothing relevant", m.recall_at_k([0, 0], 5) == 0.0)
check("k larger than the list", m.hit_at_k([1], 10) == 1.0 and m.ndcg_at_k([1], 10) == 1.0)

dcg = 1 / math.log2(4) + 1 / math.log2(6)
ideal = 1 + 1 / math.log2(3)
check("nDCG worked out by hand", close(m.ndcg_at_k([0, 0, 1, 0, 1], 5), dcg / ideal))
check("nDCG is 1.0 for a perfect order and 0 for none", close(m.ndcg_at_k([1, 1, 0, 0], 4), 1.0) and m.ndcg_at_k([0, 0], 5) == 0.0)
check("nDCG rewards relevant items higher up", m.ndcg_at_k([1, 0, 0, 0], 4) > m.ndcg_at_k([0, 0, 0, 1], 4))
check("nDCG only looks at the top k", m.ndcg_at_k([0, 0, 0, 1], 3) == 0.0)

# ---- query_metrics / aggregate
q = m.query_metrics([0, 1, 0])
check("query_metrics has every metric", set(q) == set(m.METRIC_NAMES) and close(q["mrr"], 0.5) and q["hit@1"] == 0.0 and q["hit@3"] == 1.0)
agg = m.aggregate([m.query_metrics([1, 0]), m.query_metrics([0, 0])])
check("aggregate is the mean over questions", close(agg["mrr"], 0.5) and agg["queries"] == 2 and close(agg["hit@1"], 0.5))
check("aggregate of nothing is all zeros", m.aggregate([])["mrr"] == 0.0 and m.aggregate([])["queries"] == 0)

# ---- ordering and evaluation
check("order_by_scores: best first", m.order_by_scores([0.1, 0.9, 0.5]) == [1, 2, 0])
check("order_by_scores: ties keep the original order", m.order_by_scores([0.5, 0.5, 0.5, 0.9]) == [3, 0, 1, 2])
cands = [{"scheme_id": "S001"}, {"scheme_id": "S002"}, {"scheme_id": "S001"}]
check("relevance_for_order follows the given order", m.relevance_for_order(cands, [1, 2, 0], "S001") == [0, 1, 1])
items = [{"relevant_scheme_id": "S001", "candidates": cands}, {"relevant_scheme_id": "S002", "candidates": cands}]
agg1, per = m.evaluate_items(items, lambda it: [0, 1, 2])
check("evaluate_items: first-stage order", close(per[0]["mrr"], 1.0) and close(per[1]["mrr"], 0.5) and close(agg1["mrr"], 0.75))
agg2, _ = m.evaluate_items(items, lambda it: [1, 0, 2])
check("a different order gives different metrics", close(agg2["mrr"], 0.75) and m.evaluate_items(items, lambda it: [2, 0, 1])[0]["hit@1"] == 0.5)

# ---- bootstrap
same = m.paired_bootstrap([1, 0, 1, 0.5], [1, 0, 1, 0.5])
check("bootstrap: identical systems -> difference 0, interval around 0", same["mean_diff"] == 0 and same["ci95"] == (0.0, 0.0))
better = m.paired_bootstrap([1.0] * 30, [0.2] * 30)
check("bootstrap: a clearly better system has an interval above 0", better["ci95"][0] > 0 and better["p_a_better"] == 1.0)
noisy = m.paired_bootstrap([1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 1, 0])
check("bootstrap: a noisy tie includes 0", noisy["ci95"][0] <= 0 <= noisy["ci95"][1])
check("bootstrap is repeatable (same seed)", m.paired_bootstrap([1, 0, 1, 0.3], [0, 1, 0, 0.9]) == m.paired_bootstrap([1, 0, 1, 0.3], [0, 1, 0, 0.9]))
check("bootstrap: the interval brackets the mean", noisy["ci95"][0] <= noisy["mean_diff"] <= noisy["ci95"][1])
check("bootstrap: empty input", m.paired_bootstrap([], [])["queries"] == 0)
try:
    m.paired_bootstrap([1], [1, 2])
    check("bootstrap rejects lists of different lengths", False)
except ValueError:
    check("bootstrap rejects lists of different lengths", True)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
