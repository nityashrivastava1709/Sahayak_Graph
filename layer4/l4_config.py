"""
l4_config.py  -  shared settings for every Layer 4 script (baselines + reranker).

Nothing is run here. Other files do `import l4_config as config` to get:
  * folder paths (Layer 1's schemes_clean.csv and Layer 3's vector index are READ; training data and
    models are written to data/reranker/)
  * Gemini settings (for the vector-RAG baseline and the training-question generator)
  * reranker and training settings, all optional, read from the root .env
  * ensure_layer3(): finds your Layer 3 folder and makes its modules importable

SECRETS: the API key is read from the environment / .env only. It is never printed or written anywhere.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

for _stream in (sys.stdout, sys.stderr):                      # Windows consoles: force UTF-8 output
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

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


def _int(name, default):
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


def _float(name, default):
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else default


# ---------- folders and files ----------
PROCESSED_DIR = ROOT / "data" / "processed"
RERANKER_DIR = ROOT / "data" / "reranker"
LLM_CACHE_DIR = ROOT / "data" / "interim" / "l4_llm_cache"
LLM_FAILURE_LOG = ROOT / "data" / "interim" / "l4_llm_failures.log"

SPLITS_JSON = RERANKER_DIR / "splits.json"                  # which schemes are train / val / test
QUESTIONS_JSONL = RERANKER_DIR / "questions.jsonl"          # LLM-written user questions
GROUPS_TRAIN = RERANKER_DIR / "groups_train.jsonl"          # (question, 1 positive, negatives) for training
GROUPS_VAL = RERANKER_DIR / "groups_val.jsonl"              # same, for validation loss
EVAL_VAL = RERANKER_DIR / "eval_val.jsonl"                  # first-stage candidates per question (validation)
EVAL_TEST = RERANKER_DIR / "eval_test.jsonl"                # same, held-out test schemes
DATA_REPORT = RERANKER_DIR / "data_report.json"
REVIEW_CSV = RERANKER_DIR / "review_questions.csv"          # a sample to check by eye
MODEL_DIR = RERANKER_DIR / "finetuned"                      # the fine-tuned model
HISTORY_CSV = RERANKER_DIR / "training_history.csv"
ACTIVE_FILE = RERANKER_DIR / "active_reranker.json"         # which model the agents should use
RERANKER_REPORT = RERANKER_DIR / "reranker_report.json"
LATENCY_REPORT = RERANKER_DIR / "latency_report.json"

# ---------- Gemini (vector-RAG baseline, question generation) ----------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
GEMINI_MAX_OUTPUT_TOKENS = _int("GEMINI_MAX_OUTPUT_TOKENS", 8192)
GEMINI_MAX_OUTPUT_TOKENS_CAP = _int("GEMINI_MAX_OUTPUT_TOKENS_CAP", 32768)
GEMINI_MAX_RETRIES = _int("GEMINI_MAX_RETRIES", 5)
GEMINI_TIMEOUT_SECONDS = _float("GEMINI_TIMEOUT_SECONDS", 120.0)
GEMINI_MIN_SECONDS_BETWEEN_CALLS = _float("GEMINI_MIN_SECONDS_BETWEEN_CALLS", 0.0)
_temp = os.getenv("GEMINI_TEMPERATURE", "").strip()
GEMINI_TEMPERATURE = float(_temp) if _temp else None
GEMINI_THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "").strip()

# ---------- reranker ----------
# A pretrained cross-encoder that scores (question, passage) pairs. First use downloads it (~90 MB).
RERANKER_BASE_MODEL = os.getenv("RERANKER_BASE_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2").strip()
RERANKER_MAX_LENGTH = _int("RERANKER_MAX_LENGTH", 512)
RERANKER_BATCH_SIZE = _int("RERANKER_BATCH_SIZE", 16)
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "auto").strip().lower()      # auto | cpu | cuda
RERANK_TOP_N = _int("RERANK_TOP_N", 20)                    # how many first-stage candidates get reranked
MAX_LATENCY_SECONDS = _float("MAX_LATENCY_SECONDS", 3.0)   # budget for reranking RERANK_TOP_N candidates

# ---------- training-data generation ----------
QUESTIONS_PER_CHUNK = _int("QUESTIONS_PER_CHUNK", 4)
NEGATIVES_PER_QUERY = _int("NEGATIVES_PER_QUERY", 7)
HARD_NEGATIVES = _int("HARD_NEGATIVES", 4)
VAL_FRACTION = _float("VAL_FRACTION", 0.15)
TEST_FRACTION = _float("TEST_FRACTION", 0.15)

# ---------- training ----------
TRAIN_EPOCHS = _int("TRAIN_EPOCHS", 3)
TRAIN_LR = _float("TRAIN_LR", 2e-5)
TRAIN_BATCH_GROUPS = _int("TRAIN_BATCH_GROUPS", 4)         # questions per step (each with 1 + negatives passages)
TRAIN_WARMUP = _float("TRAIN_WARMUP", 0.1)
TRAIN_WEIGHT_DECAY = _float("TRAIN_WEIGHT_DECAY", 0.01)
TRAIN_LOSS = os.getenv("TRAIN_LOSS", "listwise").strip().lower()           # listwise | bce
TRAIN_PATIENCE = _int("TRAIN_PATIENCE", 2)
SEED = _int("SEED", 42)


# ---------- Layer 3 ----------
class Layer3NotFound(Exception):
    """Layer 3 (the RAG folder) could not be located."""


def find_layer3():
    """Locate the folder holding Layer 3's retriever.py and rag_config.py (any folder name works)."""
    override = os.getenv("LAYER3_DIR", "").strip()
    here = Path(__file__).resolve().parent
    if override:
        path = Path(override)
        if (path / "retriever.py").exists() and (path / "rag_config.py").exists():
            return path
        raise Layer3NotFound(f"LAYER3_DIR={override} does not contain retriever.py and rag_config.py")
    for pattern in ("retriever.py", "*/retriever.py", "*/*/retriever.py"):
        for found in sorted(ROOT.glob(pattern)):
            folder = found.parent
            if folder != here and (folder / "rag_config.py").exists():
                return folder
    raise Layer3NotFound(
        f"Could not find Layer 3 (a folder with retriever.py and rag_config.py) under {ROOT}. "
        "Put the layer3 folder inside the project, or set LAYER3_DIR in .env to its full path.")


def ensure_layer3():
    """Make Layer 3's modules importable (retriever, corpus, rag_config, ...). Returns the folder."""
    folder = find_layer3()
    if str(folder) not in sys.path:
        sys.path.append(str(folder))
    return folder
