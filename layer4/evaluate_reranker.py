"""
evaluate_reranker.py  -  is the reranker actually better? Decides which model the agents should use.

Run:   python evaluate_reranker.py                   (after train_reranker.py)
       python evaluate_reranker.py --force pretrained    override the decision

It compares, on the validation and the UNSEEN TEST schemes:
    first_stage   the order the Layer 3 vector search already gives (no reranking)
    pretrained    the pretrained cross-encoder, before any fine-tuning   (your safety net)
    finetuned     the model trained by train_reranker.py                 (if it exists)
Metrics: MRR, Hit@1/3/5 (the plan's "Recall@k"), nDCG@5/10. A paired bootstrap shows how sure the differences are.

THE DECISION (written to data/reranker/active_reranker.json, read by Layer 5):
  * the fine-tuned model becomes active only if its MRR is at least the pretrained model's on validation AND higher on test
  * "rerank_recommended" is true only if the chosen model beats the first-stage order alone on the test questions
    (with a good first-stage search, reranking may NOT help; then the agents should skip it)
"""
import argparse
import json
import sys
from datetime import datetime

import l4_config as config
import rerank_data as rd
import rerank_metrics as rm
from reranker import CrossEncoderReranker, RerankerUnavailable

COLUMNS = ["mrr", "hit@1", "hit@3", "hit@5", "ndcg@5", "ndcg@10"]


def decide_active(first_stage: dict, pretrained: dict, finetuned, force=None) -> dict:
    """
    Each argument is {'val': metrics, 'test': metrics}; `finetuned` is None when no fine-tuned model exists.
    Returns {'active': 'pretrained'|'finetuned', 'rerank_recommended': bool, 'reason': [str]}.
    """
    reasons = []
    if force == "finetuned" and finetuned is None:
        raise ValueError("--force finetuned, but no fine-tuned model exists. Run train_reranker.py first.")
    if force:
        active = force
        reasons.append(f"forced with --force {force}")
    elif finetuned is None:
        active = "pretrained"
        reasons.append("no fine-tuned model exists yet; using the pretrained one")
    else:
        val_ok = finetuned["val"]["mrr"] >= pretrained["val"]["mrr"]
        test_ok = finetuned["test"]["mrr"] > pretrained["test"]["mrr"]
        if val_ok and test_ok:
            active = "finetuned"
            reasons.append(f"fine-tuned beats pretrained: val MRR {finetuned['val']['mrr']:.3f} vs {pretrained['val']['mrr']:.3f}, "
                           f"test MRR {finetuned['test']['mrr']:.3f} vs {pretrained['test']['mrr']:.3f}")
        else:
            active = "pretrained"
            reasons.append(f"fine-tuned is not better on both validation (MRR {finetuned['val']['mrr']:.3f} vs {pretrained['val']['mrr']:.3f}) "
                           f"and test (MRR {finetuned['test']['mrr']:.3f} vs {pretrained['test']['mrr']:.3f}); keeping the pretrained model. "
                           "Check the training questions (review_questions.csv) before training again.")
    chosen = finetuned if active == "finetuned" else pretrained
    recommended = chosen["test"]["mrr"] > first_stage["test"]["mrr"]
    reasons.append(
        f"{active} test MRR {chosen['test']['mrr']:.3f} vs first-stage alone {first_stage['test']['mrr']:.3f}: "
        + ("reranking helps, so use it" if recommended else "reranking does NOT improve on the first-stage order; the agents should skip it"))
    return {"active": active, "rerank_recommended": bool(recommended), "reason": reasons}


def model_order(reranker):
    def order(item):
        return rm.order_by_scores(reranker.score(item["query"], [c["text"] for c in item["candidates"]]))
    return order


def print_table(title, results, n):
    print(f"\n{title}  ({n} questions)")
    print(f"{'system':14}" + "".join(f"{c:>9}" for c in COLUMNS))
    for name, m in results.items():
        print(f"{name:14}" + "".join(f"{m[c]:>9.3f}" for c in COLUMNS))


def main():
    p = argparse.ArgumentParser(description="Compare first-stage, pretrained and fine-tuned ranking.")
    p.add_argument("--force", choices=["pretrained", "finetuned"], help="override the decision")
    args = p.parse_args()

    items = {"val": rd.read_jsonl(config.EVAL_VAL), "test": rd.read_jsonl(config.EVAL_TEST)}
    if not items["val"] or not items["test"]:
        raise SystemExit("No evaluation data. Run  python make_training_data.py  first.")
    try:
        rankers = {"pretrained": CrossEncoderReranker(config.RERANKER_BASE_MODEL)}
        if (config.MODEL_DIR / "config.json").exists():
            rankers["finetuned"] = CrossEncoderReranker(str(config.MODEL_DIR))
    except RerankerUnavailable as e:
        raise SystemExit(f"RERANKER PROBLEM: {e}")

    results, per_query = {"val": {}, "test": {}}, {"val": {}, "test": {}}
    for split in ("val", "test"):
        orders = {"first_stage": lambda it: list(range(len(it["candidates"])))}
        orders.update({name: model_order(r) for name, r in rankers.items()})
        for name, order_fn in orders.items():
            agg, pq = rm.evaluate_items(items[split], order_fn)
            results[split][name] = agg
            per_query[split][name] = [q["mrr"] for q in pq]
        ceiling = sum(any(c["scheme_id"] == it["relevant_scheme_id"] for c in it["candidates"]) for it in items[split]) / len(items[split])
        print_table(f"{split.upper()} schemes" + (" (also used to pick the best epoch)" if split == "val" else " (never seen in training)"),
                    results[split], len(items[split]))
        print(f"   first-stage ceiling: a right-scheme chunk is among the top {config.RERANK_TOP_N} for {ceiling:.1%} of the questions "
              "(no reranker can do better than this)")

    test_pq = per_query["test"]
    boots = {}
    for a, b in (("pretrained", "first_stage"), ("finetuned", "pretrained"), ("finetuned", "first_stage")):
        if a in test_pq and b in test_pq:
            r = rm.paired_bootstrap(test_pq[a], test_pq[b])
            boots[f"{a}_vs_{b}"] = {"mean_mrr_diff": round(r["mean_diff"], 4), "ci95": [round(x, 4) for x in r["ci95"]],
                                    "p_better": round(r["p_a_better"], 3)}
            sure = "the interval excludes 0" if (r["ci95"][0] > 0 or r["ci95"][1] < 0) else "the interval includes 0: could be luck"
            print(f"\nTEST MRR {a} - {b}: {r['mean_diff']:+.4f}  (95% interval {r['ci95'][0]:+.3f} to {r['ci95'][1]:+.3f}; {sure})")

    bundle = lambda name: {"val": results["val"][name], "test": results["test"][name]}
    try:
        decision = decide_active(bundle("first_stage"), bundle("pretrained"), bundle("finetuned") if "finetuned" in rankers else None, args.force)
    except ValueError as e:
        raise SystemExit(str(e))
    active_path = str(config.MODEL_DIR) if decision["active"] == "finetuned" else config.RERANKER_BASE_MODEL
    config.ACTIVE_FILE.write_text(json.dumps({**decision, "model_path": active_path, "decided_at": datetime.now().isoformat(timespec="seconds"),
                                              "test_mrr": {k: round(v["mrr"], 4) for k, v in results["test"].items()}}, indent=2), encoding="utf-8")
    config.RERANKER_REPORT.write_text(json.dumps({"results": results, "bootstrap_test_mrr": boots, "decision": decision,
                                                  "n_questions": {k: len(v) for k, v in items.items()},
                                                  "base_model": config.RERANKER_BASE_MODEL}, indent=2), encoding="utf-8")
    print("\nDECISION")
    for line in decision["reason"]:
        print("  -", line)
    print(f"  active model: {decision['active']}   |   rerank_recommended: {decision['rerank_recommended']}")
    print(f"\nSaved {config.ACTIVE_FILE.name} and {config.RERANKER_REPORT.name}. Next: python benchmark_latency.py")


if __name__ == "__main__":
    sys.exit(main())
