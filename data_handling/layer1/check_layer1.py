"""
check_layer1.py  -  Final completeness check for Layer 1.

Run after steps A-G:  python check_layer1.py

PASS = requirement met
WARN = a known limit of the data, not a pipeline error (nothing is invented to hide it)
FAIL = something to fix before Layer 2
"""
import pandas as pd

import config

statuses = []


def report(ok: bool, text: str, warn_only: bool = False):
    status = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    statuses.append(status)
    print(f"{status}  {text}")


def failed_in_gemini_log(namespace: str) -> set:
    """Scheme IDs for which Gemini gave no usable answer (from data/interim/llm_failures.log)."""
    ids = set()
    if config.LLM_FAILURE_LOG.exists():
        for line in config.LLM_FAILURE_LOG.read_text(encoding="utf-8").splitlines():
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3 and parts[1] == namespace:
                ids.add(parts[2])
    return ids


def main():
    s = pd.read_csv(config.SCHEMES_FINAL, dtype=str).fillna("")
    c = pd.read_csv(config.CONDITIONS_CSV, dtype=str).fillna("")
    d = pd.read_csv(config.DOCUMENTS_CSV, dtype=str).fillna("")
    a = pd.read_csv(config.APP_STEPS_CSV, dtype=str).fillna("")
    c["valid"] = c["valid"] == "True"
    ids = set(s["scheme_id"])

    report(50 <= len(s) <= 100, f"scheme count is {len(s)} (target 50-100)")
    report(s["scheme_id"].is_unique, "scheme IDs are unique")
    for col in ["scheme_name", "primary_category", "state", "details", "benefits", "official_link"]:
        report((s[col] != "").all(), f"every scheme has {col}")
    unknown = list(s[s["state"] == "UNKNOWN"]["scheme_id"])
    report(not unknown, f"no scheme has an UNKNOWN state ({len(unknown)} left: {unknown})")

    # conditions: separate "Gemini failed" (must fix) from "text has nothing mappable" (data limit)
    no_valid = ids - set(c[c["valid"]]["scheme_id"])
    failed = sorted(no_valid & failed_in_gemini_log("conditions"))
    unmappable = sorted(no_valid - set(failed))
    report(not failed, f"Gemini produced an answer for every scheme's conditions "
                       f"(failed: {failed}; re-run step C to retry them)")
    report(not unmappable,
           f"every scheme has >= 1 valid condition ({len(unmappable)} have none because the text names "
           f"nothing in the attribute vocabulary: {unmappable})", warn_only=True)

    report(ids <= set(d["scheme_id"]), "every scheme has >= 1 document")
    report(ids <= set(a["scheme_id"]), "every scheme has >= 1 application step")
    nodept = list(s[s["department"] == "Not stated"]["scheme_id"])
    report(not nodept, f"every scheme has a department ({len(nodept)} 'Not stated': {nodept})",
           warn_only=True)
    report(config.CONFLICTS_CSV.exists(), "conflicts.csv exists")
    report((config.PROCESSED_DIR / "spot_check_report.txt").exists(),
           "spot-check report exists (Step D: sheet, fill it in, then score)")

    print(f"\n{statuses.count('PASS')} passed, {statuses.count('WARN')} warnings, "
          f"{statuses.count('FAIL')} failed")
    if statuses.count("FAIL") == 0:
        print("Layer 1 is complete. Warnings are limits of the source data and are documented above.")


if __name__ == "__main__":
    main()