"""
build_index.py  -  chunk the schemes, embed the chunks and store them in ChromaDB.

Run:   python build_index.py
Reads : data/processed/schemes_clean.csv           (Layer 1; never changed)
Writes: data/vectorstore/                          (the vector index; rebuilt from scratch every run)
        data/processed/rag_index_report.json       (what was built, with which model and settings)

Optional overrides (normally set in .env):  --backend hashing   --model <name>   --strategy section
The first run with the default model downloads it once (about 130 MB).
"""
import argparse
import json
import sys
import time
from datetime import datetime

import corpus
import embedders
import rag_config as config
import vector_store
from chunker import summarize


def main():
    parser = argparse.ArgumentParser(description="Build the SahayakGraph vector index.")
    parser.add_argument("--backend", help="sentence-transformers or hashing (default from .env)")
    parser.add_argument("--model", help="embedding model name or local folder (default from .env)")
    parser.add_argument("--strategy", choices=["merged", "section"], help="chunking strategy (default from .env)")
    args = parser.parse_args()

    started = time.time()
    try:
        schemes = corpus.load_schemes()
    except corpus.CorpusError as e:
        raise SystemExit(f"DATA PROBLEM: {e}")
    print(f"Read {len(schemes)} schemes from {config.SCHEMES_CSV.name}")

    try:
        embedder = embedders.get_embedder(args.backend, args.model)
    except embedders.EmbeddingSetupError as e:
        raise SystemExit(f"EMBEDDING PROBLEM: {e}")
    info = embedder.describe()
    print(f"Embedding model: {info['backend']} / {info['model']}  ({info['dimension']} dimensions, "
          f"input limit {info['max_tokens']} tokens)")

    strategy = args.strategy or config.CHUNK_STRATEGY
    chunks = corpus.build_chunks(schemes, embedder.count_tokens, strategy=strategy)
    stats = summarize(chunks)
    print(f"Chunking ({strategy}, max {config.CHUNK_MAX_TOKENS} tokens, overlap {config.CHUNK_OVERLAP_TOKENS}): "
          f"{stats['chunks']} chunks, tokens min/median/max = {stats['tokens_min']}/{stats['tokens_median']}/{stats['tokens_max']}")

    prefix_tokens = embedder.count_tokens(embedder.doc_prefix) if embedder.doc_prefix else 0
    too_long = [c for c in chunks if c.tokens + c.header_tokens + prefix_tokens + 2 > embedder.max_tokens]
    if too_long:
        worst = max(too_long, key=lambda c: c.tokens + c.header_tokens)
        raise SystemExit(
            f"{len(too_long)} chunk(s) would be cut off by the model's {embedder.max_tokens}-token input limit "
            f"(worst: {worst.chunk_id}, {worst.tokens + worst.header_tokens} tokens with its header). "
            f"Lower CHUNK_MAX_TOKENS in .env (for example to {max(100, embedder.max_tokens - 120)}) and run again.")

    print("Embedding the chunks ...")
    vectors = []
    for i in range(0, len(chunks), 64):
        vectors.extend(embedder.encode_documents([c.embed_text for c in chunks[i:i + 64]]))
        print(f"   {min(i + 64, len(chunks))}/{len(chunks)}")

    meta = dict(info)
    meta.update({
        "embedding_backend": info["backend"], "embedding_model": info["model"],
        "data_hash": corpus.data_hash(schemes), "chunk_strategy": strategy,
        "chunk_max_tokens": config.CHUNK_MAX_TOKENS, "chunk_overlap_tokens": config.CHUNK_OVERLAP_TOKENS,
        "n_chunks": len(chunks), "n_schemes": len(schemes),
        "built_at": datetime.now().isoformat(timespec="seconds"),
    })
    collection_meta = {k: v for k, v in meta.items() if k not in ("backend", "model")}   # flat, no duplicates

    client = vector_store.open_client()
    collection = vector_store.create_collection(client, extra_metadata=collection_meta)
    vector_store.upsert(collection, [c.chunk_id for c in chunks], vectors, [c.text for c in chunks],
                        [c.metadata for c in chunks])
    stored = collection.count()
    if stored != len(chunks):
        raise SystemExit(f"ERROR: stored {stored} chunks but built {len(chunks)}. The index was not trusted.")

    report = {**collection_meta, **stats, "seconds": round(time.time() - started, 1),
              "vectorstore": str(config.VECTORSTORE_DIR), "collection": config.COLLECTION_NAME}
    config.INDEX_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nStored {stored} chunks for {len(schemes)} schemes in {config.VECTORSTORE_DIR}")
    print(f"Saved {config.INDEX_REPORT}   ({report['seconds']}s)")
    print("Next: python validate_index.py")


if __name__ == "__main__":
    sys.exit(main())
