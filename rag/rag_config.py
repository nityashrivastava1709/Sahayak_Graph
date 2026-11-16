"""
rag_config.py  -  shared settings for every Layer 3 (RAG) script.

Nothing is run here. Other files do `import rag_config as config` to get:
  * folder paths (Layer 1's schemes_clean.csv is READ; the vector store is written to data/vectorstore/)
  * embedding-model and chunking settings, read from the root .env (all optional)
  * state names used to normalise a state filter
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Windows consoles can choke on characters such as the rupee sign. Force UTF-8 output.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _find_root() -> Path:
    """The project root is the folder that holds data\\processed. Set PROJECT_ROOT to override."""
    if os.getenv("PROJECT_ROOT"):
        return Path(os.environ["PROJECT_ROOT"]).resolve()
    here = Path(__file__).resolve().parent
    candidates = (here, *here.parents)
    for c in candidates:
        if (c / "data" / "processed").exists():
            return c
    for c in candidates:
        if (c / ".env").exists():
            return c
    return here.parent


ROOT = _find_root()
ENV_FILE = ROOT / ".env"
ENV_PROBLEM = ""
if ENV_FILE.exists():
    if ENV_FILE.read_bytes()[:2] in (b"\xff\xfe", b"\xfe\xff"):
        ENV_PROBLEM = f"{ENV_FILE} is saved as UTF-16 (PowerShell '>' does this). Re-save it as UTF-8."
        print("WARNING:", ENV_PROBLEM)
    else:
        load_dotenv(ENV_FILE, encoding="utf-8-sig")

# quieter, Windows-friendly defaults for the libraries used here
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

# ---------- folders and files ----------
PROCESSED_DIR = ROOT / "data" / "processed"
SCHEMES_CSV = PROCESSED_DIR / "schemes_clean.csv"                 # Layer 1 output (read only)
VECTORSTORE_DIR = ROOT / "data" / "vectorstore"                   # ChromaDB files
INDEX_REPORT = PROCESSED_DIR / "rag_index_report.json"
RETRIEVAL_REPORT = PROCESSED_DIR / "rag_retrieval_report.json"
COLLECTION_NAME = os.getenv("RAG_COLLECTION", "sahayak_schemes")


def _env_int(name, default):
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


# ---------- embedding model (override any of these in the root .env) ----------
# "sentence-transformers" = a real semantic model that runs on your computer (default).
# "hashing" = a simple keyword-based fallback that needs no download. Fine for tests and
#             offline work, but it does not understand meaning; use the real model for the project.
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "sentence-transformers").strip().lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5").strip()
# Leave unset to use the model's recommended prefixes (bge: query instruction; e5: 'query: ' / 'passage: ').
EMBEDDING_QUERY_PREFIX = os.environ.get("EMBEDDING_QUERY_PREFIX")     # None = automatic
EMBEDDING_DOC_PREFIX = os.environ.get("EMBEDDING_DOC_PREFIX")
EMBEDDING_BATCH_SIZE = _env_int("EMBEDDING_BATCH_SIZE", 16)

# ---------- chunking ----------
# merged  : consecutive sections of a scheme are packed into chunks of up to CHUNK_MAX_TOKENS
#           (the plan's 300-500 token chunks). Long sections are split with overlap.
# section : one chunk per section (finer citations, shorter chunks).
CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "merged").strip().lower()
CHUNK_MAX_TOKENS = _env_int("CHUNK_MAX_TOKENS", 420)      # must fit the model (bge-small: 512) with the header
CHUNK_OVERLAP_TOKENS = _env_int("CHUNK_OVERLAP_TOKENS", 60)

SEED = 42

# ---------- states (used to normalise a state filter) ----------
INDIAN_STATES_AND_UTS = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa",
    "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala",
    "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland",
    "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Andaman and Nicobar Islands", "Chandigarh",
    "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Jammu and Kashmir",
    "Ladakh", "Lakshadweep", "Puducherry",
]
STATE_ALIASES = {"pondicherry": "Puducherry", "orissa": "Odisha", "nct of delhi": "Delhi",
                 "new delhi": "Delhi", "jammu & kashmir": "Jammu and Kashmir",
                 "uttaranchal": "Uttarakhand"}
STATE_CODES = {
    "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh", "AS": "Assam", "BR": "Bihar",
    "CG": "Chhattisgarh", "CT": "Chhattisgarh", "GA": "Goa", "GJ": "Gujarat",
    "HR": "Haryana", "HP": "Himachal Pradesh", "JH": "Jharkhand", "KA": "Karnataka",
    "KL": "Kerala", "MP": "Madhya Pradesh", "MH": "Maharashtra", "MN": "Manipur",
    "ML": "Meghalaya", "MZ": "Mizoram", "NL": "Nagaland", "OD": "Odisha", "OR": "Odisha",
    "PB": "Punjab", "RJ": "Rajasthan", "SK": "Sikkim", "TN": "Tamil Nadu",
    "TS": "Telangana", "TG": "Telangana", "TR": "Tripura", "UP": "Uttar Pradesh",
    "UK": "Uttarakhand", "WB": "West Bengal", "AN": "Andaman and Nicobar Islands",
    "CH": "Chandigarh", "DN": "Dadra and Nagar Haveli and Daman and Diu",
    "DD": "Dadra and Nagar Haveli and Daman and Diu", "DL": "Delhi",
    "JK": "Jammu and Kashmir", "LA": "Ladakh", "LD": "Lakshadweep", "PY": "Puducherry",
}


def canonical_state(value):
    """'mp', 'Orissa', 'madhya pradesh' -> full official name; unrecognised -> None."""
    s = str(value).strip()
    lower = s.lower()
    by_lower = {x.lower(): x for x in INDIAN_STATES_AND_UTS}
    return by_lower.get(lower) or STATE_ALIASES.get(lower) or STATE_CODES.get(s.upper())
