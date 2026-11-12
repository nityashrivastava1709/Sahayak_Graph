"""
STEP D  -  Check the quality of the extracted conditions.

Three commands:

  python step_d_spot_check.py checks
      Automatic checks on conditions.csv (every scheme has rules, no invalid rules ...).

  python step_d_spot_check.py sheet [--fresh]
      Picks 20% of the schemes at random and writes data/processed/spot_check_sheet.csv.
      It refuses to overwrite a sheet that already exists (so your filled-in work is safe).
      --fresh  = round 2: keeps the old sheet and report as *_round1/2/..., and draws a NEW
                 sample that excludes the schemes already checked. Use it after improving the
                 Step C prompt, because the old sample guided the changes and would give an
                 over-optimistic score.
      Open it in Excel. For each scheme, compare `eligibility_text` with
      `extracted_conditions` and fill in the five count columns:
          correct          rules that are right
          wrong_value      right attribute, wrong number/value/operator
          wrong_attribute  rule filed under the wrong attribute
          missing          requirements in the text that no row captures. A requirement the
                           vocabulary cannot express is NOT missing if an 'unmapped' row records
                           it. Use the same reading for every scheme and every round.
          hallucinated     rules that are not in the text at all
      (Ignore rows whose source is 'metadata'; those are not shown.)

  python step_d_spot_check.py score
      Reads the filled sheet and prints the error rate.
      If the error rate is above 15%, improve the Step C prompt and re-run Step C.
"""
import argparse

import pandas as pd

import config

SHEET = config.PROCESSED_DIR / "spot_check_sheet.csv"
REPORT = config.PROCESSED_DIR / "spot_check_report.txt"
COUNT_COLS = ["correct", "wrong_value", "wrong_attribute", "missing", "hallucinated"]
ERROR_LIMIT = 0.15


def run_checks():
    schemes = pd.read_csv(config.STEP_A_OUT, dtype=str).fillna("")
    cond = pd.read_csv(config.CONDITIONS_CSV, dtype=str).fillna("")
    cond["valid"] = cond["valid"] == "True"
    llm_valid = cond[(cond["source"] == "llm") & cond["valid"]]

    no_rules = sorted(set(schemes["scheme_id"]) - set(llm_valid["scheme_id"]))
    invalid = cond[(~cond["valid"]) & (cond["attribute"] != "unmapped")]
    unmapped = cond[cond["attribute"] == "unmapped"]
    dupes = cond.duplicated(["scheme_id", "attribute", "operator", "value"]).sum()

    print(f"Schemes: {len(schemes)}   Rules: {len(cond)}   Valid LLM rules: {len(llm_valid)}")
    print(f"Unmapped rules (kept for reference, not used for matching): {len(unmapped)}")
    print(f"Duplicate rules: {dupes}")
    print(f"Schemes with NO valid extracted rule: {len(no_rules)} {no_rules}")
    if len(invalid):
        print(f"\n{len(invalid)} rules broke the vocabulary:")
        print(invalid[["scheme_id", "attribute", "operator", "value", "problems"]]
              .to_string(index=False))
    print("\nRules per attribute:")
    print(llm_valid["attribute"].value_counts().to_string())


def next_round_name(stem: str, suffix: str):
    """spot_check_sheet_round1.csv, _round2.csv ... the first name not used yet."""
    n = 1
    while (config.PROCESSED_DIR / f"{stem}_round{n}{suffix}").exists():
        n += 1
    return config.PROCESSED_DIR / f"{stem}_round{n}{suffix}", n


def make_sheet(fresh: bool = False):
    schemes = pd.read_csv(config.STEP_A_OUT, dtype=str).fillna("")
    cond = pd.read_csv(config.CONDITIONS_CSV, dtype=str).fillna("")
    cond = cond[cond["source"] == "llm"]

    already_checked = set()
    if SHEET.exists():
        if not fresh:
            raise SystemExit(f"{SHEET.name} already exists, so it was NOT overwritten.\n"
                             "  - To start a second round with a NEW sample: python step_d_spot_check.py sheet --fresh\n"
                             "  - To redo round 1 from scratch: delete the file yourself first.")
        old = pd.read_csv(SHEET, dtype=str).fillna("")
        already_checked = set(old["scheme_id"])
        backup, n = next_round_name("spot_check_sheet", ".csv")
        SHEET.rename(backup)
        print(f"Old sheet kept as {backup.name}")
        if REPORT.exists():
            REPORT.rename(config.PROCESSED_DIR / f"spot_check_report_round{n}.txt")
            print(f"Old report kept as spot_check_report_round{n}.txt")

    pool = schemes[~schemes["scheme_id"].isin(already_checked)]
    n = min(max(1, round(len(schemes) * 0.20)), len(pool))
    sample = pool.sample(n=n, random_state=config.SEED).sort_values("scheme_id")

    rows = []
    for _, s in sample.iterrows():
        mine = cond[cond["scheme_id"] == s["scheme_id"]]
        text = "\n".join(f"[{r.group_id}] {r.attribute} {r.operator} {r.value}" for r in mine.itertuples())
        rows.append({"scheme_id": s["scheme_id"], "scheme_name": s["scheme_name"],
                     "eligibility_text": s["eligibility"], "extracted_conditions": text,
                     **{c: "" for c in COUNT_COLS}, "notes": ""})
    pd.DataFrame(rows).to_csv(SHEET, index=False, encoding="utf-8-sig")
    print(f"Wrote {n} schemes to {SHEET}")
    print("Fill the five count columns for every row, save, then run:  python step_d_spot_check.py score")


def score():
    sheet = pd.read_csv(SHEET, dtype=str).fillna("")
    for c in COUNT_COLS:
        sheet[c] = pd.to_numeric(sheet[c], errors="coerce")
    unfilled = sheet[sheet[COUNT_COLS].isna().any(axis=1)]
    if len(unfilled):
        print(f"{len(unfilled)} row(s) are not filled in yet: {list(unfilled['scheme_id'])}")
        print("Enter 0 where there are none. Then run this command again.")
        return

    totals = sheet[COUNT_COLS].sum()
    errors = totals[["wrong_value", "wrong_attribute", "missing", "hallucinated"]].sum()
    total = totals.sum()
    rate = errors / total if total else 0
    sheet["errors"] = sheet[["wrong_value", "wrong_attribute", "missing", "hallucinated"]].sum(axis=1)

    extracted = totals[["correct", "wrong_value", "wrong_attribute", "hallucinated"]].sum()
    true_rules = totals[["correct", "wrong_value", "wrong_attribute", "missing"]].sum()
    precision = totals["correct"] / extracted if extracted else 0     # of what was extracted, how much is right
    recall = totals["correct"] / true_rules if true_rules else 0      # of what the text requires, how much was found
    lines = [f"Schemes checked: {len(sheet)}",
             f"Rules judged   : {int(total)}",
             *[f"  {c:16}: {int(totals[c])}" for c in COUNT_COLS],
             f"Precision      : {precision:.1%}   (extracted rules that are right)",
             f"Recall         : {recall:.1%}   (required rules that were found)",
             f"Error rate     : {rate:.1%}  (limit {ERROR_LIMIT:.0%})",
             "Verdict        : " + ("OK - extraction quality is acceptable"
                                    if rate <= ERROR_LIMIT else
                                    "TOO HIGH - improve the Step C prompt and re-run Step C"),
             "", "Schemes with the most errors:",
             sheet.sort_values("errors", ascending=False)[["scheme_id", "scheme_name", "errors"]]
             .head(5).to_string(index=False)]
    text = "\n".join(lines)
    REPORT.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nSaved {REPORT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["checks", "sheet", "score"])
    parser.add_argument("--fresh", action="store_true",
                        help="with 'sheet': start round 2 with a new sample (keeps the old files)")
    args = parser.parse_args()
    if args.command == "sheet":
        make_sheet(fresh=args.fresh)
    else:
        {"checks": run_checks, "score": score}[args.command]()