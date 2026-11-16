"""
vector_store.py  -  the only file that talks to ChromaDB.

  * the store lives on disk in data/vectorstore/ (nothing to start or stop)
  * the collection uses COSINE distance (Chroma's default would be squared-L2, which gives different scores)
  * we always pass our own embeddings; Chroma never downloads or runs a model
  * build_where() turns simple filters into Chroma's filter syntax
"""
import chromadb
from chromadb.config import Settings

import chunker
import rag_config as config


class IndexMissingError(Exception):
    """The vector index has not been built yet."""


class IndexMismatchError(Exception):
    """The index was built with a different embedder than the one being used to query it."""


def open_client(path=None):
    path = path or config.VECTORSTORE_DIR
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))


def create_collection(client, name=None, extra_metadata=None):
    """Create an empty cosine collection, replacing any existing one with the same name."""
    name = name or config.COLLECTION_NAME
    try:
        client.delete_collection(name)
    except Exception:
        pass                                                    # it did not exist
    metadata = {"hnsw:space": "cosine"}
    metadata.update(extra_metadata or {})
    return client.create_collection(name=name, metadata=metadata, embedding_function=None)


def get_collection(client, name=None):
    name = name or config.COLLECTION_NAME
    try:
        return client.get_collection(name=name, embedding_function=None)
    except Exception as e:
        raise IndexMissingError(
            f"The vector index '{name}' was not found in {config.VECTORSTORE_DIR}. "
            "Build it first:  python build_index.py") from e


def upsert(collection, ids, embeddings, documents, metadatas, batch_size: int = 128):
    for i in range(0, len(ids), batch_size):
        collection.upsert(ids=ids[i:i + batch_size], embeddings=[list(map(float, v)) for v in embeddings[i:i + batch_size]],
                          documents=documents[i:i + batch_size], metadatas=metadatas[i:i + batch_size])


def _all_of(conditions):
    conditions = [c for c in conditions if c]
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def _any_of(conditions):
    return conditions[0] if len(conditions) == 1 else {"$or": conditions}


def build_where(state=None, include_central=True, level=None, categories=None, sections=None, scheme_ids=None):
    """
    Combine filters (all must hold). Returns None for 'no filter'.
      state            only chunks of that state's schemes (plus Central schemes when include_central)
      level            'Central' or 'State'
      categories       any of these categories (a scheme can have several)
      sections         chunks that contain any of these sections (overview, benefits, eligibility, ...)
      scheme_ids       only these schemes
    """
    conditions = []
    if state:
        names = [state] + (["Central"] if include_central and state != "Central" else [])
        conditions.append({"state": {"$in": names}} if len(names) > 1 else {"state": names[0]})
    if level:
        conditions.append({"level": level})
    if categories:
        conditions.append(_any_of([{chunker.category_flag(c): True} for c in categories]))
    if sections:
        conditions.append(_any_of([{f"sec_{s}": True} for s in sections]))
    if scheme_ids:
        conditions.append({"scheme_id": {"$in": list(scheme_ids)}})
    return _all_of(conditions)
