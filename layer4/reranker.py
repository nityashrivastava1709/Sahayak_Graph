"""
reranker.py  -  the cross-encoder reranker (inference). Layer 5's Discovery agent calls this.

A first-stage search (Layer 3) is fast but rough: it compares one vector per question with one vector per chunk.
A CROSS-ENCODER reads the question and a chunk TOGETHER and outputs a relevance score, which orders the top
candidates much better, at the cost of one model pass per candidate. So we use it only on the top ~20.

    from reranker import CrossEncoderReranker, safe_rerank
    rr = CrossEncoderReranker.load_active()                     # fine-tuned model if it proved better, else pretrained
    ranked, info = safe_rerank(rr, "pension for widows", hits)  # never raises because of the model; falls back to the input order

FIXED FORMAT
  input   candidates = list of dicts, each with at least  "text"  (other keys, e.g. chunk_id / scheme_id / score, are kept)
  output  the same dicts, best first, with three keys added (existing keys of the same name are overwritten):
            "rerank_score"   float 0..1 (None if the model could not be used)
            "original_rank"  1-based position in the input list
            "rank"           1-based position in the output list
  ties keep the original order; top_k cuts the list.
"""
import json
import time

import l4_config as config


class RerankerUnavailable(Exception):
    """The model or PyTorch could not be loaded. The message says what to do."""


def _torch():
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        raise RerankerUnavailable("PyTorch or transformers is not installed. Run:  pip install -r requirements.txt") from e
    return torch, AutoModelForSequenceClassification, AutoTokenizer


def resolve_device(preference=None) -> str:
    torch = _torch()[0]
    pref = (preference or config.RERANKER_DEVICE).lower()
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "cuda" and not torch.cuda.is_available():
        raise RerankerUnavailable("RERANKER_DEVICE=cuda but no GPU is available. Use auto or cpu.")
    if pref not in ("cpu", "cuda"):
        raise RerankerUnavailable("RERANKER_DEVICE must be auto, cpu or cuda.")
    return pref


def score_texts(model, tokenizer, query, texts, device, max_length, batch_size) -> list:
    """Relevance probabilities (0..1) for (query, text) pairs. Shared by inference, training and evaluation."""
    torch = _torch()[0]
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer([query] * len(batch), batch, padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt").to(device)
            logits = model(**enc).logits.reshape(-1)
            out.extend(torch.sigmoid(logits).float().cpu().tolist())
    return out


def _check_candidates(candidates, text_key):
    if not isinstance(candidates, list):
        raise ValueError("candidates must be a list of dicts")
    for i, c in enumerate(candidates):
        if not isinstance(c, dict) or not isinstance(c.get(text_key), str) or not c[text_key].strip():
            raise ValueError(f"candidate #{i} must be a dict with a non-empty '{text_key}' string")


def _check_query(query):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    return query.strip()[:1000]                           # a very long query would crowd out the passage


class CrossEncoderReranker:
    def __init__(self, model_path=None, device=None, max_length=None, batch_size=None):
        torch, Model, Tok = _torch()
        self.model_path = str(model_path or config.RERANKER_BASE_MODEL)
        self.max_length = int(max_length or config.RERANKER_MAX_LENGTH)
        self.batch_size = int(batch_size or config.RERANKER_BATCH_SIZE)
        self.device = resolve_device(device)
        try:
            self.tokenizer = Tok.from_pretrained(self.model_path)
            self.model = Model.from_pretrained(self.model_path)
        except Exception as e:
            raise RerankerUnavailable(
                f"Could not load the reranker '{self.model_path}'. The first use downloads it (about 90 MB) and needs "
                f"internet; or set RERANKER_BASE_MODEL in .env to a model name or a local folder. Details: {e}") from e
        if int(getattr(self.model.config, "num_labels", 0)) != 1:
            raise RerankerUnavailable(f"'{self.model_path}' does not output a single relevance score "
                                      f"(num_labels={self.model.config.num_labels}); a cross-encoder reranker needs num_labels=1.")
        self.model.eval().to(self.device)

    @classmethod
    def load_active(cls, **kwargs):
        """The model chosen by evaluate_reranker.py: the fine-tuned one if it proved better, else the pretrained one."""
        info = active_info()
        return cls(model_path=info["model_path"], **kwargs)

    def score(self, query, texts) -> list:
        query = _check_query(query)
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise ValueError("texts must be a list of strings")
        if not texts:
            return []
        return score_texts(self.model, self.tokenizer, query, texts, self.device, self.max_length, self.batch_size)

    def rerank(self, query, candidates, text_key="text", top_k=None) -> list:
        _check_candidates(candidates, text_key)
        if top_k is not None and (not isinstance(top_k, int) or top_k < 1):
            raise ValueError("top_k must be a whole number >= 1")
        if not candidates:
            return []
        scores = self.score(query, [c[text_key] for c in candidates])
        order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))     # ties keep the input order
        ranked = []
        for new_rank, i in enumerate(order, start=1):
            item = dict(candidates[i])
            item.update({"rerank_score": round(float(scores[i]), 6), "original_rank": i + 1, "rank": new_rank})
            ranked.append(item)
        return ranked[:top_k] if top_k else ranked

    def describe(self) -> dict:
        return {"model_path": self.model_path, "device": self.device, "max_length": self.max_length}


def _passthrough(candidates, top_k):
    out = []
    for i, c in enumerate(candidates, start=1):
        item = dict(c)
        item.update({"rerank_score": None, "original_rank": i, "rank": i})
        out.append(item)
    return out[:top_k] if top_k else out


def safe_rerank(reranker, query, candidates, top_k=None, text_key="text", max_seconds=None):
    """
    Rerank, but NEVER let the model break the pipeline. Returns (results, info).
    If there is no reranker, or it fails for any reason, the input order is returned (same format, rerank_score None)
    and info["used"] is False with the reason. Wrong INPUT (not a list of dicts with text) still raises ValueError:
    that is a bug in the caller, not a model problem.
    """
    _check_candidates(candidates, text_key)
    _check_query(query)
    limit = max_seconds if max_seconds is not None else config.MAX_LATENCY_SECONDS
    info = {"used": False, "reason": None, "latency_s": 0.0, "slow": False, "n": len(candidates),
            "model": getattr(reranker, "model_path", None)}
    if reranker is None:
        info["reason"] = "no reranker configured"
        return _passthrough(candidates, top_k), info
    if not candidates:
        info["used"] = True
        return [], info
    started = time.perf_counter()
    try:
        ranked = reranker.rerank(query, candidates, text_key=text_key, top_k=top_k)
    except Exception as e:                                     # model, memory, device problems
        info["latency_s"] = round(time.perf_counter() - started, 4)
        info["reason"] = f"{type(e).__name__}: {str(e)[:200]}"
        return _passthrough(candidates, top_k), info
    info["latency_s"] = round(time.perf_counter() - started, 4)
    info["used"] = True
    info["slow"] = info["latency_s"] > limit
    return ranked, info


def active_info() -> dict:
    """Which model the agents should use, and whether reranking is recommended at all (set by evaluate_reranker.py)."""
    info = {"active": "pretrained", "model_path": config.RERANKER_BASE_MODEL, "rerank_recommended": None,
            "decided": False}
    if config.ACTIVE_FILE.exists():
        try:
            saved = json.loads(config.ACTIVE_FILE.read_text(encoding="utf-8"))
            info.update({k: saved[k] for k in ("active", "rerank_recommended", "reason", "decided_at") if k in saved})
            info["decided"] = True
            if saved.get("active") == "finetuned" and (config.MODEL_DIR / "config.json").exists():
                info["model_path"] = str(config.MODEL_DIR)
            else:
                info["active"] = "pretrained"
        except (OSError, ValueError):
            pass
    return info
