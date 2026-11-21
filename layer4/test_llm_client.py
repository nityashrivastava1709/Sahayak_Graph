"""
test_llm_client.py  -  tests for the Gemini wrapper, using a scripted fake client. No API key or internet needed.

Run:  python test_llm_client.py
Checks that a reply is accepted only when it is complete, valid JSON that matches the schema; that temporary errors are
retried and permanent ones stop the run; and that caching and failure logging behave.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

from google.genai import errors, types

import l4_config as config
import llm_client

time.sleep = lambda s: None                                  # no real waiting
config.GEMINI_API_KEY = "fake-key-for-test"
tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
config.LLM_CACHE_DIR = Path(tmp.name) / "cache"
config.LLM_FAILURE_LOG = Path(tmp.name) / "failures.log"
passed = failed = 0


def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
    else:
        failed += 1
        print(f"FAIL  {name}   {detail}")


def resp(text=None, finish="STOP", no_candidates=False, blocked=False):
    if no_candidates:
        return types.GenerateContentResponse(candidates=[])
    if blocked:
        return types.GenerateContentResponse(prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason=types.BlockedReason.SAFETY))
    content = types.Content(role="model", parts=[types.Part(text=text)] if text is not None else [])
    return types.GenerateContentResponse(candidates=[types.Candidate(content=content, finish_reason=getattr(types.FinishReason, finish))])


class FakeModels:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def generate_content(self, model, contents, config):
        self.calls.append(config)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClient:
    def __init__(self, script):
        self.models = FakeModels(script)


SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}, "n": {"type": "integer"}}, "required": ["a", "n"]}
GOOD = json.dumps({"a": "x", "n": 1})


def run(script, **kw):
    llm_client._client = FakeClient(script)
    return llm_client._client, lambda: llm_client.call_gemini_json("sys", "user", SCHEMA, **kw)


def expect(name, fn, exc=None, check_fn=None):
    try:
        out = fn()
        ok = exc is None and (check_fn(out) if check_fn else True)
    except Exception as e:
        ok, out = (exc is not None and isinstance(e, exc)), f"{type(e).__name__}: {str(e)[:100]}"
    check(name, ok, "" if ok else str(out))


c, f = run([resp(GOOD)])
expect("a valid reply is accepted", f, check_fn=lambda o: o[0] == {"a": "x", "n": 1})
cfg = c.models.calls[0]
check("the request asks for JSON with the schema and the no-guessing rules",
      cfg.response_mime_type == "application/json" and cfg.response_json_schema == SCHEMA and "General rules" in cfg.system_instruction)
c, f = run([resp(None, finish="MAX_TOKENS"), resp(GOOD)])
expect("a cut-off reply (no text) is retried", f, check_fn=lambda o: o[0]["n"] == 1)
check("...with double the token budget", c.models.calls[1].max_output_tokens == 2 * c.models.calls[0].max_output_tokens)
c, f = run([resp('{"a": "x", "n": 1'), resp(GOOD)])
expect("invalid JSON is rejected and retried", f, check_fn=lambda o: len(c.models.calls) == 2)
c, f = run([resp("```json\n" + GOOD + "\n```"), resp(GOOD)])
expect("JSON in a markdown fence is NOT silently repaired", f, check_fn=lambda o: len(c.models.calls) == 2)
c, f = run([resp('{"a": "x"}'), resp('{"a": "x", "n": "one"}'), resp(GOOD)])
expect("a reply that breaks the schema is rejected", f, check_fn=lambda o: len(c.models.calls) == 3)
c, f = run([resp("  "), resp(no_candidates=True), resp(GOOD)])
expect("empty text / no candidates are retried", f, check_fn=lambda o: len(c.models.calls) == 3)
c, f = run([errors.ServerError(503, {"error": {"message": "busy"}}), errors.ClientError(429, {"error": {"message": "quota"}}), resp(GOOD)])
expect("503 and 429 are retried", f, check_fn=lambda o: len(c.models.calls) == 3)
c, f = run([ConnectionError("down"), resp(GOOD)])
expect("a network error is retried", f, check_fn=lambda o: len(c.models.calls) == 2)
for code, label in [(404, "a retired/unknown model"), (403, "a bad key"), (400, "a rejected request")]:
    c, f = run([errors.ClientError(code, {"error": {"message": "x"}})])
    expect(f"{label} (HTTP {code}) stops the run", f, exc=llm_client.LLMFatalError)
c, f = run([resp(None, finish="SAFETY")])
expect("a safety block skips the item without retrying", f, exc=llm_client.LLMResponseError)
check("...after a single call", len(c.models.calls) == 1)
c, f = run([resp("not json")] * config.GEMINI_MAX_RETRIES)
expect("a reply that stays bad ends in an error after all retries", f, exc=llm_client.LLMResponseError)
check("...using every retry", len(c.models.calls) == config.GEMINI_MAX_RETRIES)

# ---- cache and failure log
llm_client._client = FakeClient([resp(GOOD)])
r1 = llm_client.cached_llm_json("t", "K1", "sys", "user", schema=SCHEMA)
r2 = llm_client.cached_llm_json("t", "K1", "sys", "user", schema=SCHEMA)
check("a second identical call comes from the cache", r1 == r2 == json.loads(GOOD) and len(llm_client._client.models.calls) == 1)
llm_client._client = FakeClient([resp("garbage")] * config.GEMINI_MAX_RETRIES)
check("a failed item returns None", llm_client.cached_llm_json("t", "K2", "sys", "user", schema=SCHEMA) is None)
log = config.LLM_FAILURE_LOG.read_text(encoding="utf-8")
check("...is logged, and the key is not in the log", "K2" in log and "fake-key" not in log)
check("...and nothing is cached for it", not list((config.LLM_CACHE_DIR / "t").glob("K2_*.json")))
path = next((config.LLM_CACHE_DIR / "t").glob("K1_*.json"))
path.write_text('{"parsed": {"a": 1}}')
llm_client._client = FakeClient([resp(GOOD)])
check("a damaged cache file is not trusted", llm_client.cached_llm_json("t", "K1", "sys", "user", schema=SCHEMA) == json.loads(GOOD))
config.GEMINI_MODEL, old_model = "another-model", config.GEMINI_MODEL
llm_client._client = FakeClient([resp(GOOD)])
llm_client.cached_llm_json("t", "K1", "sys", "user", schema=SCHEMA)
check("changing the model bypasses old cache entries", len(llm_client._client.models.calls) == 1)
config.GEMINI_MODEL = old_model
llm_client._client, config.GEMINI_API_KEY = None, ""
try:
    llm_client.get_client()
    check("a missing key is a clear fatal error", False)
except llm_client.LLMFatalError as e:
    check("a missing key is a clear fatal error", "GEMINI_API_KEY" in str(e))

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
