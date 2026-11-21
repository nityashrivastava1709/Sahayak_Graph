"""
make_training_data.py  -  build the reranker's training, validation and test data.

Run:   python make_training_data.py              (needs the Layer 3 index and a Gemini key)
       python make_training_data.py --limit-chunks 20      small trial run first (recommended)

What it does
  1. reads every chunk from the Layer 3 vector index
  2. splits the SCHEMES into train / validation / test (kept the same on every re-run, so the test schemes stay unseen)
  3. asks Gemini for realistic user questions about each chunk (cached, so a re-run costs nothing)
  4. drops questions that give the scheme's name away, or that are unusable
  5. builds, per question, a training group (1 positive + negatives from other schemes of the SAME split)
     and, for validation/test, the list of first-stage candidates over the whole index
  6. audits everything for leakage and writes the files in data/reranker/

LOOK AT THE QUESTIONS: the model can only be as good as this data. Open data/reranker/review_questions.csv and read
about 30 rows: are the questions realistic, and does the chunk really answer them?
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter

import l4_config as config

config.ensure_layer3()
from retriever import Retriever  # noqa: E402  (Layer 3)

import llm_client  # noqa: E402
import rerank_data as rd  # noqa: E402

QUESTION_SYSTEM = """You write realistic questions that ordinary people in India would type or say when looking for government help. You are given one passage from an official scheme description. Write the requested number of different questions that this passage can answer.

Rules:
1. Every question must be answerable from the passage alone.
2. NEVER use the scheme's name, its acronym, or any official title. Describe the person's situation or need instead.
3. Mix these styles: "situation" = the person describes their circumstances and asks for help (for example: my husband died and I have two children, is there any pension for me?); "short" = a brief search-style phrase of 3 to 8 words; "formal" = a polite, complete question.
4. Use simple everyday English the way real users write, including occasional informal wording. Do not copy long phrases from the passage.
5. Do not invent facts that the passage does not support.
6. Return the style of each question in the "style" field."""

QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"style": {"type": "string", "enum": list(rd.STYLES)}, "text": {"type": "string"}},
                "required": ["style", "text"],
            },
        }
    },
    "required": ["questions"],
}


def load_chunks(retriever) -> list:
    data = retriever.collection.get(include=["documents", "metadatas"])
    chunks = [{"chunk_id": i, "scheme_id": m["scheme_id"], "scheme_name": m["scheme_name"], "text": d,
               "chunk_index": m["chunk_index"], "state": m["state"], "primary_category": m["primary_category"]}
              for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])]
    return sorted(chunks, key=lambda c: (c["scheme_id"], c["chunk_index"]))


def load_or_make_splits(scheme_ids, new_splits=False) -> dict:
    """Reuse the saved split when the schemes are unchanged, so the test schemes stay unseen across re-runs."""
    wanted = set(scheme_ids)
    if config.SPLITS_JSON.exists() and not new_splits:
        saved = json.loads(config.SPLITS_JSON.read_text(encoding="utf-8"))
        if set(saved["train"]) | set(saved["val"]) | set(saved["test"]) == wanted:
            print(f"Reusing the saved split ({config.SPLITS_JSON.name}).")
            return saved
        print("WARNING: the set of schemes changed since the last run, so a NEW split is made. "
              "Models trained earlier have seen different train/test schemes.")
    splits = rd.split_schemes(wanted, config.VAL_FRACTION, config.TEST_FRACTION, config.SEED)
    config.SPLITS_JSON.parent.mkdir(parents=True, exist_ok=True)
    config.SPLITS_JSON.write_text(json.dumps({**splits, "seed": config.SEED}, indent=1), encoding="utf-8")
    return splits


def generate_questions(chunks, splits, n_per_chunk):
    records, failures, dropped_total, few = [], [], Counter(), []
    for i, chunk in enumerate(chunks, start=1):
        user = (f"Scheme category: {chunk['primary_category']}\nState: {chunk['state']}\n\nPASSAGE:\n{chunk['text']}\n\n"
                f"Write {n_per_chunk} questions.")
        parsed = llm_client.cached_llm_json("train_questions", chunk["chunk_id"], QUESTION_SYSTEM, user, QUESTION_SCHEMA)
        if i % 20 == 0 or i == len(chunks):
            print(f"   questions: {i}/{len(chunks)} chunks")
        if parsed is None:
            failures.append(chunk["chunk_id"])
            continue
        kept, dropped = rd.clean_questions(parsed["questions"], chunk["scheme_name"], max_n=n_per_chunk + 1)
        dropped_total.update(dropped)
        if len(kept) < 2:
            few.append(chunk["chunk_id"])
        for j, q in enumerate(kept):
            records.append({"qid": f"{chunk['chunk_id']}-q{j}", "question": q["text"], "style": q["style"],
                            "chunk_id": chunk["chunk_id"], "scheme_id": chunk["scheme_id"],
                            "split": rd.split_of(chunk["scheme_id"], splits)})
    return records, failures, dropped_total, few


def main():
    parser = argparse.ArgumentParser(description="Build the reranker training data.")
    parser.add_argument("--limit-chunks", type=int, help="only the first N chunks (a trial run)")
    parser.add_argument("--new-splits", action="store_true", help="make a fresh train/val/test split")
    args = parser.parse_args()

    try:
        retriever = Retriever()
    except Exception as e:
        raise SystemExit(f"LAYER 3 PROBLEM: {e}")
    chunks = load_chunks(retriever)
    scheme_ids = sorted({c["scheme_id"] for c in chunks})
    splits = load_or_make_splits(scheme_ids, args.new_splits)
    print(f"{len(chunks)} chunks from {len(scheme_ids)} schemes | schemes: train {len(splits['train'])}, "
          f"validation {len(splits['val'])}, test {len(splits['test'])}")
    if args.limit_chunks:
        chunks_for_questions = chunks[:args.limit_chunks]
        print(f"TRIAL RUN: questions for the first {len(chunks_for_questions)} chunks only.")
    else:
        chunks_for_questions = chunks

    print(f"Asking Gemini ({config.GEMINI_MODEL}) for {config.QUESTIONS_PER_CHUNK} questions per chunk ...")
    try:
        questions, failures, dropped, few = generate_questions(chunks_for_questions, splits, config.QUESTIONS_PER_CHUNK)
    except llm_client.LLMFatalError as e:
        raise SystemExit(f"GEMINI PROBLEM: {e}")
    if not questions:
        raise SystemExit("No usable questions were produced. Check data/interim/l4_llm_failures.log.")
    if failures:
        print(f"WARNING: Gemini gave no usable answer for {len(failures)} chunk(s) (see {config.LLM_FAILURE_LOG.name}); "
              "run again to retry just those.")
    rd.write_jsonl(config.QUESTIONS_JSONL, questions)

    # ---- training and validation groups (negatives only from the same split) ----
    by_split = {name: {c["chunk_id"]: c for c in chunks if c["scheme_id"] in set(splits[name])} for name in ("train", "val", "test")}
    groups = {"train": [], "val": []}
    for name in ("train", "val"):
        pool, ids = by_split[name], sorted(splits[name])

        def search_pool(text, k, ids=ids):
            return [{"chunk_id": h["chunk_id"], "scheme_id": h["scheme_id"], "text": h["text"]}
                    for h in retriever.search(text, k=min(k, 200), scheme_ids=ids)]

        for q in (x for x in questions if x["split"] == name):
            rng = random.Random(f"{config.SEED}-{q['qid']}")
            group = rd.build_group(q, pool, search_pool, rng, config.NEGATIVES_PER_QUERY, config.HARD_NEGATIVES)
            if group:
                groups[name].append(group)
    problems = rd.audit_groups(groups["train"] + groups["val"], splits)
    if problems:
        raise SystemExit("DATA LEAKAGE CHECK FAILED:\n  " + "\n  ".join(problems[:8]))
    rd.write_jsonl(config.GROUPS_TRAIN, groups["train"])
    rd.write_jsonl(config.GROUPS_VAL, groups["val"])

    # ---- evaluation items: first-stage candidates over the whole index ----
    def search_all(text, k):
        return retriever.search(text, k=min(k, 200))

    eval_items = {"val": [], "test": []}
    for name in ("val", "test"):
        for q in (x for x in questions if x["split"] == name):
            eval_items[name].append(rd.build_eval_item(q, search_all, config.RERANK_TOP_N))
    rd.write_jsonl(config.EVAL_VAL, eval_items["val"])
    rd.write_jsonl(config.EVAL_TEST, eval_items["test"])

    # ---- report and a sample to read ----
    ceiling = {name: round(sum(any(c["scheme_id"] == it["relevant_scheme_id"] for c in it["candidates"]) for it in items)
                           / max(1, len(items)), 3) for name, items in eval_items.items()}
    counts = Counter(q["split"] for q in questions)
    styles = Counter(q["style"] for q in questions)
    report = {"chunks": len(chunks), "chunks_with_questions": len(chunks_for_questions), "questions": len(questions),
              "questions_per_split": dict(counts), "styles": dict(styles), "groups_train": len(groups["train"]),
              "groups_val": len(groups["val"]), "eval_val": len(eval_items["val"]), "eval_test": len(eval_items["test"]),
              "dropped_questions": dict(dropped), "chunks_with_failures": failures, "chunks_with_fewer_than_2_questions": few,
              "first_stage_ceiling_hit_at_n": ceiling, "n_candidates": config.RERANK_TOP_N,
              "negatives_per_query": config.NEGATIVES_PER_QUERY, "hard_negatives": config.HARD_NEGATIVES,
              "gemini_model": config.GEMINI_MODEL, "trial_run": bool(args.limit_chunks)}
    config.DATA_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    sample = random.Random(config.SEED).sample(questions, min(40, len(questions)))
    chunk_text = {c["chunk_id"]: c["text"] for c in chunks}
    with open(config.REVIEW_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["qid", "style", "question", "chunk_start", "realistic_and_answerable (yes/no)"])
        for q in sample:
            w.writerow([q["qid"], q["style"], q["question"], chunk_text[q["chunk_id"]][:300].replace("\n", " "), ""])

    print(f"\nQuestions: {len(questions)}  (train {counts['train']}, validation {counts['val']}, test {counts['test']})")
    print(f"Dropped: {dict(dropped)}   Styles: {dict(styles)}")
    print(f"Training groups: {len(groups['train'])}   Validation groups: {len(groups['val'])}   "
          f"Eval questions: val {len(eval_items['val'])}, test {len(eval_items['test'])}")
    print(f"First-stage ceiling (a right-scheme chunk is among the top {config.RERANK_TOP_N}): val {ceiling['val']}, test {ceiling['test']}")
    print("\nSample questions:")
    for q in sample[:10]:
        print(f"   [{q['style']:9}] {q['question'][:110]}")
    print(f"\nSaved to {config.RERANKER_DIR}\nRead {config.REVIEW_CSV.name} (about 30 rows) before training: "
          "are the questions realistic, and does each chunk answer its question?")


if __name__ == "__main__":
    sys.exit(main())
