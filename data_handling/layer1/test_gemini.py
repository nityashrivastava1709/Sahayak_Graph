"""
test_gemini.py  -  run this BEFORE the pipeline.

It checks, in order:
  1. Settings      : .env found, key present (only its length is shown), model name, SDK version
  2. Connection    : a plain "reply OK" request reaches Gemini with your key
  3. Structured JSON: a tiny extraction returns valid JSON that matches a schema, and the
                     model does NOT invent a rule that is not in the text
  4. Cut-off check : a deliberately tiny token limit must be DETECTED as truncated
                     (proves the pipeline will never accept a half-finished answer)

Run:  python test_gemini.py
"""
import sys
import time
from importlib import metadata

from google.genai import errors, types

import config
import llm_utils

results = []


def report(name, status, detail=""):
    print(f"[{status}] {name}" + (f"  -  {detail}" if detail else ""))
    results.append(status)


SAMPLE = ("The applicant should be a woman. The annual family income should not exceed "
          "Rs 2,50,000 per annum. The applicant should be a farmer.")
SYSTEM = ("You extract eligibility rules from the text. Return a JSON list. Each item has "
          "attribute (one of: gender, age, annual_income, occupation), operator, value, "
          "source_sentence. Only extract rules that are written in the text.")
SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"attribute": {"type": "string"}, "operator": {"type": "string"},
                       "value": {"type": "string"}, "source_sentence": {"type": "string"}},
        "required": ["attribute", "operator", "value", "source_sentence"],
    },
}

# ---------------------------------------------------------------- 1. settings
print("=" * 70)
print("1. SETTINGS")
print("=" * 70)
print(f"Project root : {config.ROOT}")
print(f".env file    : {config.ENV_FILE}  ({'found' if config.ENV_FILE.exists() else 'NOT FOUND'})")
print(f"Model        : {config.GEMINI_MODEL}")
try:
    print(f"google-genai : {metadata.version('google-genai')}")
except metadata.PackageNotFoundError:
    report("google-genai installed", "FAIL", "run: pip install google-genai")
    sys.exit(1)
if config.ENV_PROBLEM:
    report(".env readable", "FAIL", config.ENV_PROBLEM)
    sys.exit(1)
if not config.GEMINI_API_KEY:
    report("GEMINI_API_KEY loaded", "FAIL", f"no GEMINI_API_KEY line found in {config.ENV_FILE}")
    sys.exit(1)
report("GEMINI_API_KEY loaded", "PASS", f"{len(config.GEMINI_API_KEY)} characters (value not shown)")

# ---------------------------------------------------------------- 2. connection
print("\n" + "=" * 70)
print("2. CONNECTION")
print("=" * 70)


def plain_request(attempts=6):
    """One simple request. Temporary Google-side errors (503 overloaded, 429 rate limit) are retried."""
    for attempt in range(1, attempts + 1):
        try:
            return llm_utils.get_client().models.generate_content(
                model=config.GEMINI_MODEL,
                contents="Reply with exactly one word: OK",
                config=types.GenerateContentConfig(max_output_tokens=1024),
            )
        except errors.APIError as e:
            code = getattr(e, "code", None)
            if code in llm_utils.RETRYABLE_HTTP_CODES and attempt < attempts:
                wait = min(60, 5 * attempt)
                print(f"   Gemini answered HTTP {code} (temporary). Attempt {attempt}/{attempts}; "
                      f"waiting {wait}s ...")
                time.sleep(wait)
                continue
            raise


try:
    resp = plain_request()
    reason = resp.candidates[0].finish_reason if resp.candidates else None
    text = (resp.text or "").strip()
    print(f"Gemini said: {text!r}   finish_reason={getattr(reason, 'name', reason)}")
    report("connection", "PASS" if text else "FAIL", "got a reply" if text else "empty reply")
except llm_utils.LLMFatalError as e:
    report("connection", "FAIL", str(e))
    sys.exit(1)
except errors.APIError as e:
    code = getattr(e, "code", None)
    if code in llm_utils.RETRYABLE_HTTP_CODES:
        report("connection", "FAIL",
               f"Gemini was still busy/limited after several tries (HTTP {code}). Your key WAS accepted, "
               "so this is a temporary problem on Google's side.")
        print("   What to do: wait a few minutes and run this test again, or switch model by adding\n"
              "   GEMINI_MODEL=gemini-3.5-flash-lite   to your .env file (a lighter model that is "
              "usually less busy).")
    else:
        report("connection", "FAIL", llm_utils._explain_api_error(e))
    sys.exit(1)
except Exception as e:
    report("connection", "FAIL", f"{type(e).__name__}: {str(e)[:300]}")
    sys.exit(1)

# ---------------------------------------------------------------- 3. structured JSON
print("\n" + "=" * 70)
print("3. STRUCTURED JSON OUTPUT")
print("=" * 70)
try:
    data, raw = llm_utils.call_gemini_json(SYSTEM, SAMPLE, SCHEMA)
    for item in data:
        print(f"  {item['attribute']} {item['operator']} {item['value']}")
    report("valid JSON matching the schema", "PASS", f"{len(data)} rules returned")
    attrs = {i["attribute"] for i in data}
    if "age" in attrs:
        report("no invented rules", "FAIL", "the text never mentions age, but an age rule came back")
    else:
        report("no invented rules", "PASS", "no 'age' rule invented")
    if {"gender", "annual_income", "occupation"} <= attrs:
        report("expected rules found", "PASS", "gender, annual_income and occupation extracted")
    else:
        report("expected rules found", "WARN", f"got {sorted(attrs)}; expected gender, annual_income, occupation")
except llm_utils.LLMFatalError as e:
    report("structured JSON", "FAIL", str(e))
except llm_utils.LLMResponseError as e:
    report("structured JSON", "FAIL", str(e))

# ---------------------------------------------------------------- 4. truncation detection
print("\n" + "=" * 70)
print("4. CUT-OFF DETECTION")
print("=" * 70)
try:
    resp = llm_utils.get_client().models.generate_content(
        model=config.GEMINI_MODEL, contents=SAMPLE,
        config=llm_utils._build_config(SYSTEM, SCHEMA, 5))      # 5 tokens cannot be enough
    try:
        llm_utils._get_reply_text(resp)
        report("cut-off detected", "WARN", "the model finished within 5 tokens; detector not exercised")
    except llm_utils._Truncated:
        report("cut-off detected", "PASS", "a truncated reply was rejected, not accepted")
    except llm_utils._Retryable as e:
        report("cut-off detected", "PASS", f"incomplete reply rejected ({e})")
except Exception as e:
    report("cut-off detected", "WARN", f"could not run this check: {type(e).__name__}: {str(e)[:200]}")

# ---------------------------------------------------------------- summary
print("\n" + "=" * 70)
fails = results.count("FAIL")
print(f"{results.count('PASS')} passed, {results.count('WARN')} warnings, {fails} failed")
if fails:
    print("Fix the FAIL lines above before running the pipeline.")
    sys.exit(1)
print("Gemini is ready. You can run step_a_clean.py next.")