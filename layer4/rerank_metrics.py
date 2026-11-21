"""
rerank_metrics.py  -  ranking-quality measures. Plain Python, no PyTorch.

A ranked list is described by `relevance`: one 0/1 per candidate in ranked order (1 = the candidate belongs to the
scheme the question was written about). Example: [0, 0, 1, 0, 1]  -> the first relevant item is at position 3.

  reciprocal_rank  1 / position of the first relevant item (0 if none)        its mean over questions = MRR
  hit_at_k         1 if any relevant item is in the top k                     its mean = Hit@k (the plan's "Recall@k")
  recall_at_k      share of ALL relevant items that are in the top k
  ndcg_at_k        rewards relevant items near the top; 1.0 = perfect order, 0 = none in the top k
"""
import math
import random


def reciprocal_rank(relevance) -> float:
    for i, r in enumerate(relevance, start=1):
        if r:
            return 1.0 / i
    return 0.0


def hit_at_k(relevance, k) -> float:
    return 1.0 if any(relevance[:k]) else 0.0


def recall_at_k(relevance, k) -> float:
    total = sum(1 for r in relevance if r)
    return (sum(1 for r in relevance[:k] if r) / total) if total else 0.0


def ndcg_at_k(relevance, k) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(relevance[:k], start=1) if r)
    n_rel = sum(1 for r in relevance if r)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(n_rel, k) + 1))
    return dcg / ideal if ideal else 0.0


METRIC_NAMES = ["mrr", "hit@1", "hit@3", "hit@5", "recall@5", "ndcg@5", "ndcg@10"]


def query_metrics(relevance) -> dict:
    return {"mrr": reciprocal_rank(relevance), "hit@1": hit_at_k(relevance, 1), "hit@3": hit_at_k(relevance, 3),
            "hit@5": hit_at_k(relevance, 5), "recall@5": recall_at_k(relevance, 5),
            "ndcg@5": ndcg_at_k(relevance, 5), "ndcg@10": ndcg_at_k(relevance, 10)}


def aggregate(per_query: list) -> dict:
    """Mean of every metric over the questions."""
    if not per_query:
        return {name: 0.0 for name in METRIC_NAMES} | {"queries": 0}
    out = {name: sum(q[name] for q in per_query) / len(per_query) for name in METRIC_NAMES}
    out["queries"] = len(per_query)
    return out


def relevance_for_order(candidates, order, relevant_scheme_id) -> list:
    """0/1 relevance of `candidates` listed in `order` (indices); a candidate is relevant if its scheme is the right one."""
    return [1 if candidates[i]["scheme_id"] == relevant_scheme_id else 0 for i in order]


def paired_bootstrap(a: list, b: list, n: int = 2000, seed: int = 42) -> dict:
    """
    How sure are we that system A beats system B? `a` and `b` are the per-question scores (for example the
    reciprocal ranks) of the two systems on the SAME questions. Resamples the questions n times and returns
    the mean difference (A - B), its 95% interval, and the share of resamples where A > B.
    With few questions the interval is wide: if it includes 0, the difference may be luck.
    """
    if len(a) != len(b):
        raise ValueError("both lists must have one score per question")
    if not a:
        return {"mean_diff": 0.0, "ci95": (0.0, 0.0), "p_a_better": 0.0, "queries": 0}
    rng = random.Random(seed)
    diffs = [x - y for x, y in zip(a, b)]
    means = []
    for _ in range(n):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        means.append(sum(sample) / len(sample))
    means.sort()
    return {"mean_diff": sum(diffs) / len(diffs),
            "ci95": (means[int(0.025 * n)], means[int(0.975 * n) - 1]),
            "p_a_better": sum(1 for m in means if m > 0) / n, "queries": len(a)}


def order_by_scores(scores) -> list:
    """Candidate indices best-first; equal scores keep the original order."""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def evaluate_items(items, order_fn):
    """
    items     evaluation items: {'candidates': [{'scheme_id', 'text', ...}], 'relevant_scheme_id': 'S012', ...}
    order_fn  item -> candidate indices, best first (for example the first-stage order, or a model's order)
    Returns (mean metrics, list of per-question metrics).
    """
    per_query = []
    for item in items:
        order = order_fn(item)
        per_query.append(query_metrics(relevance_for_order(item["candidates"], order, item["relevant_scheme_id"])))
    return aggregate(per_query), per_query
