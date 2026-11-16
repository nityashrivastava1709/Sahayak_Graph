# SahayakGraph - Layer 3 (RAG: chunks, embeddings, vector search)

Turns the scheme text from Layer 1 into a searchable vector index (ChromaDB) and gives the agents a
retriever. Layer 3 reads Layer 1's `schemes_clean.csv` and never changes it.

```
schemes_clean.csv -> chunker -> embedder -> ChromaDB (data/vectorstore/) -> Retriever -> agents (Layer 5)
```

## Design decisions

| Decision | Choice | Why |
|---|---|---|
| Chunking | `merged` (default): consecutive sections of a scheme are packed into chunks of up to **420 tokens**; long sections are split at sentence boundaries with **60 tokens of overlap** | Your sections are short (median 50-110 words) but one overview has 1,000+ words. Packing gives the plan's 300-500 token chunks; `CHUNK_STRATEGY=section` gives one chunk per section instead |
| Context header | Every chunk is embedded as *scheme name + category + state + text* | A chunk about "income limit" still knows which scheme it belongs to |
| Metadata on every chunk | scheme_id, scheme_name, sections, state, level, primary_category, categories, official_link, tokens, chunk_index/n_chunks | Filtering by state/category/section, and every answer can cite its source |
| Chunk ids | `S012-c0`, `S012-c1` ... (stable) | The handoff contract's `source_chunk_ids` |
| Embedding model | `BAAI/bge-small-en-v1.5` (384 dimensions, 512-token input, runs on CPU) | Strong retrieval for its size; fits the chunk size. Downloaded once (~130 MB). Change with `EMBEDDING_MODEL` |
| Distance | cosine (set explicitly; Chroma's default is not cosine) | Scores are 0-1, higher = more similar |
| Index remembers its embedder | model, prefixes, dimension, chunk settings, data fingerprint | Queries always use the model that built the index; a stale index is detected |

## Files

| File | Purpose |
|---|---|
| `rag_config.py` | paths, `.env` settings, state names |
| `embedders.py` | the embedding model (`sentence-transformers`) and a keyword fallback (`hashing`, no download, for tests/offline) |
| `chunker.py` | cuts schemes into chunks (sentence packing, overlap, header, metadata) |
| `corpus.py` | reads `schemes_clean.csv`, builds chunks, fingerprints the data |
| `vector_store.py` | the only file that talks to ChromaDB; builds filters |
| `build_index.py` | chunk -> embed -> store (rebuilt from scratch each run) |
| `retriever.py` | **the interface for later layers**: `search`, `search_schemes`, `chunks_for_schemes`, `get_scheme_chunks`, `format_context` |
| `validate_index.py` | integrity checks (matches current data, no text lost, metadata complete, vectors sane) |
| `evaluate_retrieval.py` | recall@k / MRR from queries made out of each scheme's own text |
| `demo_search.py` | search from the command line |
| `test_chunker.py`, `test_embedding.py`, `test_layer3.py` | the three tests |

## Setup (once)

```powershell
cd D:\Sahayak_Graph
.\.venv\Scripts\Activate.ps1
cd layer3
pip install -r requirements.txt
```
This installs PyTorch (via sentence-transformers); you need it again for the Layer 4 reranker. No `.env` settings are required.

## Run order

```powershell
python test_chunker.py       # chunking logic, no model needed
python test_embedding.py     # loads the model (first run downloads ~130 MB, needs internet)
python build_index.py        # chunk + embed + store
python validate_index.py     # integrity checks
python test_layer3.py        # search behaviour: quality, filters, scores, grouping, bad input
python evaluate_retrieval.py # recall / MRR numbers
python demo_search.py "pension for widows" --state UP --k 5
```

More demo searches:
```powershell
python demo_search.py "scholarship for girls" --state Rajasthan --schemes
python demo_search.py "documents needed" --section documents --category "Agriculture,Rural & Environment"
python demo_search.py "fellowship for women scientists" --context        # the text block an LLM would receive
```

## Using it from later layers

```python
from retriever import Retriever, format_context
r = Retriever()
hits = r.search("pension for widows", k=5, state="UP")            # best chunks
schemes = r.search_schemes("pension for widows", k=5)             # best schemes, with their best chunks
support = r.chunks_for_schemes("income limit", ["S012", "S043"])  # supporting text for schemes the GRAPH found
prompt_text = format_context(hits)                                # "[S012-c1] Scheme name (sections) text ..."
```
Hit fields: `chunk_id, scheme_id, scheme_name, sections, text, score, distance, state, level, primary_category,
official_link, chunk_index, n_chunks, tokens`. Filters: `state` (name or code; Central schemes included unless
`include_central=False`), `level`, `categories`, `sections`, `scheme_ids`, `min_score`.

## If something goes wrong

| Message / symptom | Fix |
|---|---|
| `Could not load the embedding model` | The first run needs internet to download the model once. Check your connection / proxy, then run again. Offline? Set `EMBEDDING_BACKEND=hashing` in `.env` (keywords only, no meaning) |
| `chunk(s) would be cut off by the model's N-token input limit` | Lower `CHUNK_MAX_TOKENS` in `.env` as the message suggests, then `python build_index.py` |
| `index matches the current schemes_clean.csv` FAILS | Layer 1 data changed since the index was built: run `python build_index.py` |
| `The vector index ... was not found` | Run `python build_index.py` first |
| `The index was built with ... dimensions` | The model was changed after building: run `python build_index.py` |
| WARN on a natural-language spot check | Informational: this embedding model ranked a known scheme below the top 5 for that question. Try another `EMBEDDING_MODEL` |
| Slow first search | The model loads into memory (a few seconds). Later searches are fast |

Changing the model, chunk size or strategy always needs `python build_index.py` again (about a minute on CPU).
