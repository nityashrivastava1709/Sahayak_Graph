"""
llm_client.py  -  the ONLY file in Layer 4 that talks to Gemini (google-genai SDK).

    cached_llm_json(namespace, key, system_prompt, user_prompt, schema=...)
returns parsed, schema-validated JSON (or None if that one item could not be processed).

A reply is accepted ONLY if Gemini finished normally (not cut off), the text is non-empty, valid JSON, and matches the
schema. Anything else is retried (a cut-off reply with a bigger token budget). Temporary errors (429, 5xx, network) are
retried with waiting. Permanent errors (bad key, retired model, rejected request) stop the run with a clear message.
Good answers are cached in data/interim/l4_llm_cache/, keyed by model + prompts + schema. The API key is never printed.
"""
import hashlib
import json
import random
import time
from datetime import datetime

import jsonschema
from google import genai
from google.genai import errors, types

import l4_config as config


class LLMError(Exception):
    pass


class LLMResponseError(LLMError):
    """The answer for ONE item was unusable after all retries (that item is skipped)."""


class LLMFatalError(LLMError):
    """Configuration or permission problem; retrying cannot help, so the run stops."""


class _Retryable(Exception):
    pass


class _Truncated(_Retryable):
    pass


class _Blocked(Exception):
    pass


RETRYABLE_HTTP_CODES = {408, 429, 500, 502, 503, 504}
BLOCKED_FINISH_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION", "LANGUAGE", "IMAGE_SAFETY"}

GLOBAL_RULES = """

General rules for every task:
- Use ONLY information that is written in the text you are given. Never use outside knowledge, never guess, never fill gaps.
- If something is not stated in the text, leave it out (empty list, empty string) exactly as the task instructs. Do not invent values.
- Your output must be valid JSON that follows the required schema exactly."""

_client = None
_last_call_time = 0.0


def get_client():
    global _client
    if _client is None:
        if not config.GEMINI_API_KEY:
            hint = config.ENV_PROBLEM or f"Add a line  GEMINI_API_KEY=your-key  to {config.ENV_FILE}  (saved as UTF-8)."
            raise LLMFatalError("GEMINI_API_KEY was not found. " + hint)
        _client = genai.Client(api_key=config.GEMINI_API_KEY,
                               http_options=types.HttpOptions(timeout=int(config.GEMINI_TIMEOUT_SECONDS * 1000)))
    return _client


def _build_config(system, schema, max_tokens):
    kwargs = dict(system_instruction=system + GLOBAL_RULES, response_mime_type="application/json",
                  max_output_tokens=max_tokens)
    if schema is not None:
        field = ("response_json_schema" if "response_json_schema" in types.GenerateContentConfig.model_fields
                 else "response_schema")
        kwargs[field] = schema
    if config.GEMINI_TEMPERATURE is not None:
        kwargs["temperature"] = config.GEMINI_TEMPERATURE
    if config.GEMINI_THINKING_LEVEL:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=config.GEMINI_THINKING_LEVEL.upper())
    return types.GenerateContentConfig(**kwargs)


def _respect_rate_limit():
    global _last_call_time
    wait = config.GEMINI_MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - _last_call_time)
    if wait > 0:
        time.sleep(wait)
    _last_call_time = time.monotonic()


def _get_reply_text(resp) -> str:
    feedback = getattr(resp, "prompt_feedback", None)
    if feedback is not None and getattr(feedback, "block_reason", None):
        raise _Blocked(f"prompt was blocked by Gemini ({feedback.block_reason})")
    candidates = getattr(resp, "candidates", None) or []
    if not candidates:
        raise _Retryable("Gemini returned no candidates")
    cand = candidates[0]
    reason = getattr(cand, "finish_reason", None)
    name = getattr(reason, "name", str(reason)) if reason is not None else "NONE"
    if name == "MAX_TOKENS":
        raise _Truncated("reply was cut off (finish_reason=MAX_TOKENS)")
    if name in BLOCKED_FINISH_REASONS:
        raise _Blocked(f"Gemini refused to answer (finish_reason={name})")
    if name != "STOP":
        raise _Retryable(f"reply did not finish normally (finish_reason={name})")
    parts = getattr(getattr(cand, "content", None), "parts", None) or []
    text = "".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False))
    if not text.strip():
        raise _Retryable("reply was empty")
    return text


def _parse_and_validate(text, schema):
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise _Retryable(f"reply is not valid JSON ({e.msg} at position {e.pos})")
    if schema is not None:
        try:
            jsonschema.validate(data, schema)
        except jsonschema.ValidationError as e:
            raise _Retryable(f"JSON does not match the schema: {e.message[:200]}")
    return data


def _explain_api_error(e) -> str:
    code = getattr(e, "code", None)
    msg = str(getattr(e, "message", "") or e)[:400]
    if code in (401, 403):
        return (f"Gemini rejected the API key or permissions (HTTP {code}). Check GEMINI_API_KEY in {config.ENV_FILE} "
                f"and that the key is active. Details: {msg}")
    if code == 404:
        return (f"Model '{config.GEMINI_MODEL}' was not found (HTTP 404). Set GEMINI_MODEL in .env to a current model "
                f"(see https://ai.google.dev/gemini-api/docs/deprecations). Details: {msg}")
    if code == 400:
        return f"Gemini rejected the request as invalid (HTTP 400). Details: {msg}"
    return f"Gemini API error (HTTP {code}): {msg}"


def call_gemini_json(system, user, schema=None, max_tokens=None):
    """Ask Gemini; return (parsed_json, raw_text). Retries temporary problems."""
    tokens = max(max_tokens or 0, config.GEMINI_MAX_OUTPUT_TOKENS)
    last_problem = "unknown"
    attempts = max(1, config.GEMINI_MAX_RETRIES)
    for attempt in range(1, attempts + 1):
        _respect_rate_limit()
        try:
            resp = get_client().models.generate_content(model=config.GEMINI_MODEL, contents=user,
                                                        config=_build_config(system, schema, tokens))
            text = _get_reply_text(resp)
            return _parse_and_validate(text, schema), text
        except _Truncated as e:
            tokens = min(tokens * 2, config.GEMINI_MAX_OUTPUT_TOKENS_CAP)
            last_problem = f"{e}; retrying with max_output_tokens={tokens}"
        except _Retryable as e:
            last_problem = str(e)
        except _Blocked as e:
            raise LLMResponseError(str(e)) from e
        except LLMFatalError:
            raise
        except errors.APIError as e:
            code = getattr(e, "code", None)
            if code in RETRYABLE_HTTP_CODES:
                last_problem = f"HTTP {code}: {str(getattr(e, 'message', e))[:200]}"
            else:
                raise LLMFatalError(_explain_api_error(e)) from e
        except (ValueError, TypeError) as e:
            raise LLMFatalError(f"Invalid request settings: {e}") from e
        except Exception as e:
            last_problem = f"{type(e).__name__}: {str(e)[:200]}"
        if attempt < attempts:
            wait = min(60.0, 2.0 ** attempt) + random.uniform(0, 1)
            print(f"   attempt {attempt}/{attempts} failed ({last_problem}); waiting {wait:.0f}s")
            time.sleep(wait)
    raise LLMResponseError(f"no usable answer after {attempts} attempts. Last problem: {last_problem}")


def _log_failure(namespace, key, reason):
    config.LLM_FAILURE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(config.LLM_FAILURE_LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')} | {namespace} | {key} | {config.GEMINI_MODEL} | {reason}\n")


def cached_llm_json(namespace, key, system, user, schema=None, max_tokens=None):
    """Validated JSON for one item, from the cache if possible. None if the item could not be processed."""
    fingerprint = json.dumps({"model": config.GEMINI_MODEL, "system": system, "user": user, "schema": schema},
                             sort_keys=True, ensure_ascii=False)
    digest = hashlib.md5(fingerprint.encode("utf-8")).hexdigest()[:10]
    folder = config.LLM_CACHE_DIR / namespace
    folder.mkdir(parents=True, exist_ok=True)
    safe_key = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(key))[:80]
    path = folder / f"{safe_key}_{digest}.json"
    if path.exists():
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))["parsed"]
            if schema is not None:
                jsonschema.validate(parsed, schema)
            return parsed
        except Exception:
            print(f"   cache file for {key} is damaged; asking Gemini again")
    try:
        parsed, raw = call_gemini_json(system, user, schema, max_tokens)
    except LLMResponseError as e:
        _log_failure(namespace, key, str(e))
        print(f"   FAILED {key}: {e}  (logged in {config.LLM_FAILURE_LOG.name})")
        return None
    path.write_text(json.dumps({"parsed": parsed, "raw": raw, "model": config.GEMINI_MODEL,
                                "saved_at": datetime.now().isoformat(timespec="seconds")},
                               ensure_ascii=False, indent=1), encoding="utf-8")
    return parsed
