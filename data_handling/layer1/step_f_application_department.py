"""
STEP F  -  Application steps, online/offline mode and department.

Input : data/interim/schemes_step_a.csv
Output: data/processed/schemes_clean.csv       (the FINAL scheme table)
        data/processed/application_steps.csv   (one row per step: scheme_id, step_no, step_text)

New columns added to the scheme table
  application_mode   online | offline | both | unclear
  application_steps  3-6 short steps joined with " || "
  department         standard department name, or "Not stated"

How it works
  The LLM reads the application text and the start of the scheme description. If it cannot
  find a department, a simple pattern search looks for "Department of ..." in the text.

Run:  python step_f_application_department.py
"""
import re

import pandas as pd

import config
from llm_utils import cached_llm_json

SYSTEM_PROMPT = """You read an Indian government scheme's description and application text.

Return ONLY a JSON object with exactly these keys:
{
 "mode": "online" | "offline" | "both" | "unclear",
 "steps": ["short step 1", "short step 2", ...],
 "department": "<name of the department/ministry that runs the scheme>" or "" if not stated
}

Rules:
1. "steps": 3 to 6 short imperative steps (under 25 words each) in the order the applicant must do them. Do not invent steps.
2. "mode": "online" if the text only mentions a website/portal, "offline" if only offices/forms, "both" if both are possible.
3. "department": copy the department/ministry name exactly as the text states it (e.g. "Department of Women and Child Development, Government of Rajasthan"). If no department is stated, use an empty string "". Do not guess."""

# Two simple shapes: "Department of Social Welfare"  or  "Social Welfare Department"
# The exact JSON shape Gemini must return (structured output).
APPLICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "mode": {"type": "string", "enum": ["online", "offline", "both", "unclear"]},
        "steps": {"type": "array", "items": {"type": "string"}},
        "department": {"type": "string", "description": "empty string if the text does not state one"},
    },
    "required": ["mode", "steps", "department"],
}

DEPT_PATTERNS = [
    re.compile(r"Department of ([A-Z][A-Za-z&]+(?:\s+(?:and\s+|&\s+)?[A-Z][A-Za-z&]+){0,4})"),
    re.compile(r"((?:[A-Z][A-Za-z&]+\s+(?:and\s+)?){1,5}Department)\b"),
]


def normalise_department(name: str) -> str:
    """Tidy a department name so the same department always looks the same."""
    name = re.sub(r"\s+", " ", str(name)).strip(" .,;")
    name = name.replace("&", "and").replace("Dept.", "Department").replace("Dept ", "Department ")
    name = re.sub(r"^(the|The)\s+", "", name)
    return name


def find_department_by_pattern(*texts) -> str:
    """Fallback when the LLM found no department: look for 'Department of X' in the text."""
    for text in texts:
        for i, pattern in enumerate(DEPT_PATTERNS):
            m = pattern.search(text)
            if m:
                name = m.group(1)
                name = f"Department of {name}" if i == 0 else name
                name = normalise_department(name)
                if name.lower() not in ("department", "the department"):
                    return name
    return ""


def main():
    schemes = pd.read_csv(config.STEP_A_OUT, dtype=str).fillna("")
    modes, steps_joined, departments, step_rows = [], [], [], []

    for i, (_, s) in enumerate(schemes.iterrows(), start=1):
        print(f"[{i}/{len(schemes)}] {s['scheme_id']} {s['scheme_name'][:60]}")
        user = (f"Scheme: {s['scheme_name']}\n\nDescription:\n{s['details'][:1500]}\n\n"
                f"Application text:\n{s['application'][:3000]}\n\nReturn the JSON object now.")
        result = cached_llm_json("application", s["scheme_id"], SYSTEM_PROMPT, user,
                                 schema=APPLICATION_SCHEMA)
        if not isinstance(result, dict):
            result = {}

        mode = str(result.get("mode", "unclear")).lower()
        mode = mode if mode in ("online", "offline", "both", "unclear") else "unclear"
        steps = [str(x).strip() for x in (result.get("steps") or []) if str(x).strip()]
        dept = normalise_department(result.get("department") or "")
        if not dept:
            dept = find_department_by_pattern(s["details"], s["application"], s["scheme_name"])
        dept = dept or "Not stated"

        modes.append(mode)
        steps_joined.append(" || ".join(steps))
        departments.append(dept)
        for n, text in enumerate(steps, start=1):
            step_rows.append({"scheme_id": s["scheme_id"], "step_no": n, "step_text": text})

    schemes["application_mode"] = modes
    schemes["application_steps"] = steps_joined
    schemes["department"] = departments

    schemes.to_csv(config.SCHEMES_FINAL, index=False, encoding="utf-8-sig")
    pd.DataFrame(step_rows).to_csv(config.APP_STEPS_CSV, index=False, encoding="utf-8-sig")

    print(f"\nSaved {config.SCHEMES_FINAL}")
    print(f"Saved {config.APP_STEPS_CSV}  ({len(step_rows)} steps)")
    print("\nApplication mode:\n" + schemes["application_mode"].value_counts().to_string())
    print(f"\nSchemes with no department stated: {(schemes['department'] == 'Not stated').sum()}")
    print("Departments (top 15):\n" + schemes["department"].value_counts().head(15).to_string())


if __name__ == "__main__":
    main()
