"""
config.py  -  shared settings for every Layer 1 script (Gemini version).

Nothing is run here. Every other file does `import config` to get:
  * folder paths (they are created automatically)
  * Gemini settings (model name, retries, token limits), read from the root .env
  * the list of Indian states / UTs (used for cleaning and validation)

SECRETS: the API key is read from the environment / .env file ONLY. It is never
written in code, never printed, and never saved in a cache or log file.
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


# ---------- find the project root ----------
def _find_root() -> Path:
    """
    The project root is the folder that holds .env and data/.
    It is found by walking upward from this file, so the scripts work whether they
    sit in D:\\Sahayak_Graph or in D:\\Sahayak_Graph\\layer1.
    Set PROJECT_ROOT in the environment to override.
    """
    if os.getenv("PROJECT_ROOT"):
        return Path(os.environ["PROJECT_ROOT"]).resolve()
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / ".env").exists() or (candidate / "data" / "raw").exists():
            return candidate
    return here.parent


ROOT = _find_root()
ENV_FILE = ROOT / ".env"

# A .env created with PowerShell's  >  operator is saved as UTF-16 and cannot be read.
ENV_PROBLEM = ""
if ENV_FILE.exists():
    if ENV_FILE.read_bytes()[:2] in (b"\xff\xfe", b"\xfe\xff"):
        ENV_PROBLEM = (f"{ENV_FILE} is saved as UTF-16 (PowerShell '>' does this). "
                       "Re-save it as UTF-8 (see README) or the API key cannot be read.")
        print("WARNING:", ENV_PROBLEM)
    else:
        load_dotenv(ENV_FILE, encoding="utf-8-sig")   # utf-8-sig also tolerates a Notepad BOM

# ---------- folders ----------
RAW_DIR = ROOT / "data" / "raw"                     # untouched originals
INTERIM_DIR = ROOT / "data" / "interim"             # work-in-progress files
PROCESSED_DIR = ROOT / "data" / "processed"         # final Layer 1 outputs
CACHE_DIR = INTERIM_DIR / "llm_cache"               # saved LLM answers (so you never pay twice)
for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# The input file. It is only ever READ; no script writes to data/raw/.
RAW_SUBSET = RAW_DIR / "schemes_clean.csv"
RAW_HASH_FILE = INTERIM_DIR / "raw_input.sha256"    # fingerprint to prove the input never changed

# ---------- file names produced by each step ----------
STEP_A_OUT = INTERIM_DIR / "schemes_step_a.csv"             # cleaned text, state, categories
CONDITIONS_CSV = PROCESSED_DIR / "conditions.csv"            # step C
DOCUMENTS_CSV = PROCESSED_DIR / "documents.csv"              # step E
SCHEMES_FINAL = PROCESSED_DIR / "schemes_clean.csv"          # step F (final scheme table)
APP_STEPS_CSV = PROCESSED_DIR / "application_steps.csv"      # step F
CONFLICTS_CSV = PROCESSED_DIR / "conflicts.csv"              # step G
VOCAB_JSON = PROCESSED_DIR / "attribute_vocabulary.json"     # step B
LLM_FAILURE_LOG = INTERIM_DIR / "llm_failures.log"           # every rejected/failed LLM answer

# Safety net: the input file must never be the same file as any output.
assert RAW_SUBSET.resolve() != SCHEMES_FINAL.resolve(), "input and output must be different files"


# ---------- Gemini settings (override any of these in the root .env file) ----------
def _env_float(name, default=None):
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else default


def _env_int(name, default=None):
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


# The key. Quotes/spaces around it in the .env file are tolerated.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")

# Model name. Google retires models regularly, so this is configurable.
# Check https://ai.google.dev/gemini-api/docs/deprecations for what is current.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()

# Output-token budget per call. Gemini "thinking" tokens count against this limit,
# so keep it generous. If a reply is cut off the call is retried with double the
# budget, up to the CAP.
GEMINI_MAX_OUTPUT_TOKENS = _env_int("GEMINI_MAX_OUTPUT_TOKENS", 8192)
GEMINI_MAX_OUTPUT_TOKENS_CAP = _env_int("GEMINI_MAX_OUTPUT_TOKENS_CAP", 32768)

GEMINI_MAX_RETRIES = _env_int("GEMINI_MAX_RETRIES", 5)                  # attempts per call
GEMINI_TIMEOUT_SECONDS = _env_float("GEMINI_TIMEOUT_SECONDS", 120.0)    # per attempt
# Free-tier keys have low requests-per-minute limits. If you see many 429 errors,
# set this to 5 or 6 in .env.
GEMINI_MIN_SECONDS_BETWEEN_CALLS = _env_float("GEMINI_MIN_SECONDS_BETWEEN_CALLS", 0.0)

# Optional. Unset = use the model's own defaults (recommended to start with).
GEMINI_TEMPERATURE = _env_float("GEMINI_TEMPERATURE", None)             # e.g. 0
GEMINI_THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "").strip()  # minimal|low|medium|high
GEMINI_THINKING_BUDGET = _env_int("GEMINI_THINKING_BUDGET", None)       # older 2.5-style models

SEED = 42  # makes random sampling repeatable

# ---------- states and union territories ----------
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
# other spellings that appear in government text -> the one name we use
STATE_ALIASES = {
    "pondicherry": "Puducherry", "orissa": "Odisha", "nct of delhi": "Delhi",
    "new delhi": "Delhi", "jammu & kashmir": "Jammu and Kashmir",
    "uttaranchal": "Uttarakhand",
}

# If step A cannot work out a state, add it here by hand. These three were checked against
# the official myScheme pages (myscheme.gov.in/schemes/bsyrj, /kvy, /nscg).
STATE_OVERRIDES = {
    "S012": "Rajasthan",        # Beej Swalamban Yojna
    "S043": "Uttar Pradesh",    # Kaushal Vikas Yojana
    "S058": "Chhattisgarh",     # Nawabihan Scheme
}

# OPTIONAL. If the scheme text does not name a department, the pipeline says "Not stated" and
# never guesses. If you look a department up on the scheme's official page (the link is in
# schemes_clean.csv), you can add it here, e.g. {"S013": "Department of Social Justice"}.
DEPARTMENT_OVERRIDES = {}