# SahayakGraph - Layer 4 (Baselines + PyTorch reranker)

Two things the rest of the project needs:

* **Baselines** (Part A): two simple systems to compare the full SahayakGraph against in Layer 7, both returning results in
  the *same* output format: **keyword search (BM25)** and **vector RAG + one LLM call**.
* **Reranker** (Part B, "4B"): a **PyTorch cross-encoder** that re-orders the top ~20 search results by reading the question
  and each chunk *together*. Includes training-data generation, a training loop you can read end to end, an honest
  evaluation, and a safe fallback. Layer 5's Discovery agent calls it.

```
Part A   question --> keyword search (BM25) ----------------------------> common result format
         question --> Layer 3 vector search --> ONE Gemini call --------> common result format
Part B   Layer 3 chunks --> Gemini writes user questions --> train/val/test split BY SCHEME
         --> PyTorch fine-tuning --> evaluate vs pretrained vs no reranking --> promote only if better --> safe_rerank()
```

## The common result format (every system returns this)

`{system, query, schemes: [{scheme_id, name, rank, score, reason, documents, steps, evidence, retrieved, official_link, state}],
answer, invalid_ids, meta: {latency_s, ...}}`. `documents`/`steps` are `None` when a system does not provide them (keyword search).
A scheme the LLM named that **does not exist** goes to `invalid_ids`; a real scheme it was **never shown** is kept with
`retrieved = false`. Nothing is silently removed, so Layer 7 can count hallucinations. Full description: top of `l4_common.py`.

## Files

| File | Purpose |
|---|---|
| `l4_config.py` | paths, `.env` settings, finds your Layer 3 folder (`ensure_layer3()`) |
| `llm_client.py` | the only Gemini caller: schema-validated JSON, retries, cut-off detection, cache |
| `l4_common.py` | the common result format, `validate_result`, `profile_to_query` |
| `bm25.py` | the BM25 ranking formula (no dependencies) |
| `keyword_baseline.py` | **Baseline 1**: BM25 over scheme text |
| `vector_rag_baseline.py` | **Baseline 2**: Layer 3 search + one LLM call (optionally with the reranker) |
| `run_baselines.py` | ask both baselines one question and compare |
| `rerank_metrics.py` | MRR, Hit@k, Recall@k, nDCG@k, paired bootstrap |
| `rerank_data.py` | split by scheme, question cleaning and name-leak filter, positive/negative groups, leakage audit |
| `make_training_data.py` | Gemini writes questions per chunk; builds train / validation / test files |
| `train_reranker.py` | **the PyTorch training loop** (listwise or BCE loss, AdamW, warm-up, early stopping) |
| `reranker.py` | `CrossEncoderReranker`, `safe_rerank` (never breaks the pipeline), `active_info` |
| `evaluate_reranker.py` | first-stage vs pretrained vs fine-tuned on unseen test schemes; decides which model is active |
| `benchmark_latency.py` | timing of reranking 20 / 50 candidates + proof that the fallback works |
| `test_metrics.py`, `test_bm25.py`, `test_llm_client.py`, `test_rerank_data.py`, `test_baselines.py`, `test_reranker.py` | tests |

## Setup (once)

```powershell
cd D:\Sahayak_Graph
.\.venv\Scripts\Activate.ps1
cd layer4
pip install -r requirements.txt
```
Layer 3's index must exist (`python build_index.py` in the Layer 3 folder). `GEMINI_API_KEY` must be in the root `.env`.
The first reranker run downloads the pretrained cross-encoder once (about 90 MB).

## Part A: the baselines

```powershell
python test_metrics.py
python test_bm25.py
python test_llm_client.py
python test_baselines.py          # both baselines return the same format (no Gemini key needed: the LLM is faked)
python run_baselines.py "I am a widow in Uttar Pradesh with two children, what help can I get?" --state UP
python run_baselines.py --profile my_profile.json --k 5     # a structured profile; Layer 2's demo_profiles.py shows the format
```

## Part B: the reranker

```powershell
python test_rerank_data.py        # splitting, leakage protection, promotion decision
python test_reranker.py           # model loads, loss maths, the model really learns, save/reload, fallbacks

python make_training_data.py --limit-chunks 20      # TRIAL: ~20 Gemini calls. Read the sample questions it prints.
python make_training_data.py                         # all chunks (~220 Gemini calls, cached)
# open data\reranker\review_questions.csv and read ~30 rows: realistic? answerable from the chunk?

python train_reranker.py --limit-groups 32 --limit-eval 20 --epochs 1      # TRIAL (about a minute)
python train_reranker.py                                                    # real run: roughly 15-60 minutes on a laptop CPU
python evaluate_reranker.py
python benchmark_latency.py
```

**How to read the results.** `evaluate_reranker.py` prints MRR / Hit@k / nDCG for three systems on the validation schemes and on the
**test schemes the model never saw**: the first-stage search alone, the pretrained cross-encoder, and your fine-tuned one, plus a bootstrap interval
(if it includes 0, the difference may be luck; with ~90 schemes the test set is small). The decision rule:
the fine-tuned model becomes active only if it is at least as good as the pretrained one on validation **and** better on test;
`rerank_recommended` is true only if the chosen model beats the first-stage order alone. **A good first-stage search can make reranking
pointless**; if that is what you see, that is a valid result: Layer 5 will skip the reranker.

**The training data is the weak point.** The model can only be as good as the questions Gemini writes. Questions that name the scheme are
dropped automatically, negatives never come from the question's own scheme, and test schemes never appear in training (audited every run),
but whether the questions sound like real users is something only you can judge: read `review_questions.csv`.

## Using it from Layer 5 and Layer 7

```python
from reranker import CrossEncoderReranker, safe_rerank, active_info
info = active_info()                                   # {"active": "finetuned"|"pretrained", "rerank_recommended": True/False/None, ...}
if info["rerank_recommended"] is not False:
    rr = CrossEncoderReranker.load_active()
    ranked, status = safe_rerank(rr, query, hits, top_k=10)   # never raises because of the model; status["used"], status["reason"]

from keyword_baseline import KeywordBaseline
from vector_rag_baseline import VectorRAGBaseline
systems = {"keyword": KeywordBaseline(), "vector_rag": VectorRAGBaseline(),
           "vector_rag+rerank": VectorRAGBaseline(reranker=rr)}
result = systems["vector_rag"].run_profile(profile, k=10)     # same call, same format, for every system
```
`safe_rerank` input: a list of dicts with a `text` key (Layer 3 hits work as they are). Output: the same dicts, best first, plus
`rerank_score`, `original_rank`, `rank`.

## If something goes wrong

| Message / symptom | Fix |
|---|---|
| `Could not find Layer 3` | Put the Layer 3 folder inside the project, or set `LAYER3_DIR` in `.env` |
| `PyTorch or transformers is not installed` | `pip install -r requirements.txt` |
| `Could not load the reranker` | The first run needs internet to download the model once; or set `RERANKER_BASE_MODEL` to a local folder |
| `No training data found` | Run `make_training_data.py` first |
| Many Gemini 429/503 errors | Add `GEMINI_MIN_SECONDS_BETWEEN_CALLS=6` to `.env`; re-run (answers are cached; failures are retried) |
| `The loss became NaN` | `python train_reranker.py --lr 1e-5` |
| Training too slow | `--limit-groups`, a GPU (`RERANKER_DEVICE=cuda`), or `RERANKER_MAX_LENGTH=384` |
| `rerank_recommended: False` | Reranking does not beat your first-stage search here. Not a bug; Layer 5 will skip it |
| Fine-tuned model not promoted | Check `review_questions.csv`; more or better questions help more than more epochs |
