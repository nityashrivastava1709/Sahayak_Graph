"""
kg_config.py  -  shared settings for every Layer 2 (knowledge graph) script.

Nothing is run here. Other files do `import kg_config as config` to get:
  * folder paths (Layer 1 outputs are READ from data/processed/)
  * Neo4j connection settings, read from the root .env file
  * the list of Indian states / UTs (used to normalise a user's state)

SECRETS: the Neo4j password is read from the environment / .env file only. It is never
written in code, printed, or saved.
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
    """
    The project root is the folder that holds data\\processed (and normally .env).
    Found by walking upward from this file, so the scripts work from D:\\Sahayak_Graph\\layer2.
    Set PROJECT_ROOT in the environment to override.
    """
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
        ENV_PROBLEM = (f"{ENV_FILE} is saved as UTF-16 (PowerShell '>' does this). "
                       "Re-save it as UTF-8 or the Neo4j password cannot be read.")
        print("WARNING:", ENV_PROBLEM)
    else:
        load_dotenv(ENV_FILE, encoding="utf-8-sig")

# ---------- folders ----------
PROCESSED_DIR = ROOT / "data" / "processed"        # Layer 1 outputs live here (read only)
INTERIM_DIR = ROOT / "data" / "interim"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Layer 1 files this layer reads ----------
SCHEMES_CSV = PROCESSED_DIR / "schemes_clean.csv"
CONDITIONS_CSV = PROCESSED_DIR / "conditions.csv"
DOCUMENTS_CSV = PROCESSED_DIR / "documents.csv"
APP_STEPS_CSV = PROCESSED_DIR / "application_steps.csv"
CONFLICTS_CSV = PROCESSED_DIR / "conflicts.csv"
VOCAB_JSON = PROCESSED_DIR / "attribute_vocabulary.json"

# ---------- file this layer writes ----------
KG_LOAD_REPORT = PROCESSED_DIR / "kg_load_report.json"

# ---------- Neo4j settings (set these in the root .env) ----------
# Local Neo4j Desktop / server : bolt://localhost:7687
# Neo4j Aura (cloud)           : neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687").strip()
NEO4J_USER = (os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME") or "neo4j").strip()
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "").strip().strip('"').strip("'")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j").strip()

BATCH_SIZE = 200          # rows sent to Neo4j per write
SEED = 42                 # makes the random test profiles repeatable

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
STATE_ALIASES = {
    "pondicherry": "Puducherry", "orissa": "Odisha", "nct of delhi": "Delhi",
    "new delhi": "Delhi", "jammu & kashmir": "Jammu and Kashmir",
    "uttaranchal": "Uttarakhand",
}
# two-letter codes people (and the handoff contract example "MP") use
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
