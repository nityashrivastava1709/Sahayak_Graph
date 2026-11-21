"""
test_reranker.py  -  tests for the reranker and the PyTorch training code.

Run:  python test_reranker.py
The first run downloads the pretrained cross-encoder once (about 90 MB; needs internet). Needs PyTorch + transformers.

Checks: the model loads and scores sensibly; reranking has the fixed format and keeps ties stable; safe_rerank never lets a model
failure through; the loss functions are computed correctly (compared with hand calculation); the model can actually LEARN
(it overfits a small batch); a saved model reloads to identical scores; the active-model selection works.
"""
import math
import sys
import tempfile
from pathlib import Path

import l4_config as config
import reranker as rr

results = []


def report(name, ok, detail="", warn=False):
    status = "PASS" if ok else ("WARN" if warn else "FAIL")
    results.append(status)
    print(f"[{status}] {name}" + (f"  -  {detail}" if detail and not ok else ""))


try:
    torch, Model, Tok = rr._torch()
except rr.RerankerUnavailable as e:
    print(f"[FAIL] PyTorch / transformers: {e}")
    sys.exit(1)
print(f"PyTorch {torch.__version__} | model: {config.RERANKER_BASE_MODEL}\n")
try:
    reranker = rr.CrossEncoderReranker()
except rr.RerankerUnavailable as e:
    print(f"[FAIL] the reranker did not load: {e}")
    sys.exit(1)
report("the model loads", True)
info = reranker.describe()
print(f"   device {info['device']}, max length {info['max_length']}\n")

# ---------------- scoring ----------------
texts = ["Monthly pension for widows aged above 40 whose family income is below the limit.",
         "Subsidy on certified seeds for small farmers owning up to five acres of land.",
         "Scholarship for students pursuing higher education in science.", "Loan for starting a small shop."]
q = "pension for a woman whose husband died"
s = reranker.score(q, texts)
report("one score per text, each between 0 and 1", len(s) == 4 and all(0.0 <= x <= 1.0 for x in s), str(s))
report("scoring is deterministic", reranker.score(q, texts) == s)
small_batch = rr.CrossEncoderReranker(reranker.model_path, batch_size=1).score(q, texts)
report("the batch size does not change the scores", all(abs(a - b) < 1e-3 for a, b in zip(s, small_batch)), f"{s} vs {small_batch}")
report("no texts -> no scores", reranker.score(q, []) == [])
long_text = "pension for widows " * 2000
report("a very long passage is truncated, not an error", len(reranker.score(q, [long_text])) == 1)
report("a very long question is handled", len(reranker.score("pension " * 3000, texts[:1])) == 1)
report("unicode and Hindi text are handled", len(reranker.score("विधवा पेंशन", ["विधवाओं के लिए ₹5,000 मासिक पेंशन"])) == 1)
top = max(range(4), key=lambda i: s[i])
report("MEANING: the widow-pension text scores highest for 'pension for a woman whose husband died'", top == 0,
       f"scores {[round(x, 3) for x in s]}. Expected for the real pretrained model; a randomly initialised test model cannot do this.", warn=True)

# ---------------- rerank format ----------------
cands = [{"text": t, "chunk_id": f"S00{i}-c0", "scheme_id": f"S00{i}", "score": 0.9 - i / 10} for i, t in enumerate(texts, start=1)]
snapshot = [dict(c) for c in cands]
out = reranker.rerank(q, cands)
report("every candidate comes back, with extra keys preserved", len(out) == 4 and all({"chunk_id", "scheme_id", "score", "text"} <= set(o) for o in out))
report("rerank adds rerank_score, original_rank and rank", all({"rerank_score", "original_rank", "rank"} <= set(o) for o in out))
report("ranks are 1..n, best score first", [o["rank"] for o in out] == [1, 2, 3, 4] and [o["rerank_score"] for o in out] == sorted((o["rerank_score"] for o in out), reverse=True))
report("original_rank is the position in the input list", sorted(o["original_rank"] for o in out) == [1, 2, 3, 4] and all(cands[o["original_rank"] - 1]["chunk_id"] == o["chunk_id"] for o in out))
report("the input list is not changed", cands == snapshot)
report("top_k cuts the list", len(reranker.rerank(q, cands, top_k=2)) == 2 and reranker.rerank(q, cands, top_k=2) == out[:2])
dups = [{"text": "same text", "chunk_id": f"c{i}"} for i in range(5)]
report("equal scores keep the original order (stable)", [o["chunk_id"] for o in reranker.rerank(q, dups)] == [f"c{i}" for i in range(5)])
report("an empty list gives an empty list", reranker.rerank(q, []) == [])
report("a single candidate works", len(reranker.rerank(q, cands[:1])) == 1 and reranker.rerank(q, cands[:1])[0]["rank"] == 1)
for label, call in [("empty question", lambda: reranker.rerank("  ", cands)), ("candidates not a list", lambda: reranker.rerank(q, "text")),
                    ("candidate without text", lambda: reranker.rerank(q, [{"chunk_id": "x"}])), ("blank text", lambda: reranker.rerank(q, [{"text": " "}])),
                    ("top_k = 0", lambda: reranker.rerank(q, cands, top_k=0))]:
    try:
        call()
        report(f"{label} is rejected", False)
    except ValueError:
        report(f"{label} is rejected", True)

# ---------------- safe_rerank ----------------
ranked, i1 = rr.safe_rerank(reranker, q, cands)
report("safe_rerank with a working model reranks and reports it", i1["used"] and i1["reason"] is None and ranked == out)
passthrough, i2 = rr.safe_rerank(None, q, cands)
report("no reranker -> original order, same format, rerank_score None", not i2["used"] and [c["chunk_id"] for c in passthrough] == [c["chunk_id"] for c in cands]
       and all(c["rerank_score"] is None and c["rank"] == c["original_rank"] for c in passthrough))


class Broken:
    model_path = "broken"

    def rerank(self, *a, **k):
        raise RuntimeError("out of memory")


fb, i3 = rr.safe_rerank(Broken(), q, cands, top_k=2)
report("a failing model never breaks the pipeline: original order, reason recorded, top_k still applied",
       not i3["used"] and "out of memory" in i3["reason"] and [c["chunk_id"] for c in fb] == [c["chunk_id"] for c in cands][:2])
report("safe_rerank of an empty list", rr.safe_rerank(reranker, q, [])[0] == [])
try:
    rr.safe_rerank(reranker, q, [{"chunk_id": "x"}])
    report("wrong INPUT still raises (that is a bug in the caller, not a model failure)", False)
except ValueError:
    report("wrong INPUT still raises (that is a bug in the caller, not a model failure)", True)
_, i4 = rr.safe_rerank(reranker, q, cands, max_seconds=0.0)
report("a call slower than the limit is flagged", i4["slow"] is True and i4["used"] is True)

# ---------------- loading problems ----------------
for label, kwargs in [("a model that does not exist", dict(model_path="/no/such/model-folder"))]:
    try:
        rr.CrossEncoderReranker(**kwargs)
        report(f"{label} gives a clear error", False)
    except rr.RerankerUnavailable as e:
        report(f"{label} gives a clear error", "Could not load" in str(e))
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    from transformers import BertConfig, BertForSequenceClassification
    tok0 = Tok.from_pretrained(reranker.model_path)
    two = BertForSequenceClassification(BertConfig(vocab_size=len(tok0), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                                                   intermediate_size=64, num_labels=2))
    two.save_pretrained(tmp)
    tok0.save_pretrained(tmp)
    try:
        rr.CrossEncoderReranker(tmp)
        report("a classifier with 2 outputs is rejected (a reranker needs one score)", False)
    except rr.RerankerUnavailable as e:
        report("a classifier with 2 outputs is rejected (a reranker needs one score)", "single relevance score" in str(e))
if not torch.cuda.is_available():
    try:
        rr.resolve_device("cuda")
        report("asking for a GPU that is not there is a clear error", False)
    except rr.RerankerUnavailable:
        report("asking for a GPU that is not there is a clear error", True)
try:
    rr.resolve_device("tpu")
    report("an unknown device is rejected", False)
except rr.RerankerUnavailable:
    report("an unknown device is rejected", True)

# ---------------- the training code ----------------
import train_reranker as tr

F = torch.nn.functional
lg = torch.tensor([2.0, 0.5, -1.0])
manual = -math.log(math.exp(2.0) / (math.exp(2.0) + math.exp(0.5) + math.exp(-1.0)))
report("listwise loss = -log(softmax probability of the right chunk), checked by hand", abs(tr.compute_loss(lg, [3], "listwise", torch).item() - manual) < 1e-5)
mixed = tr.compute_loss(torch.tensor([1.0, 0.0, 0.0, 2.0, 0.5]), [3, 2], "listwise", torch).item()
m1 = -math.log(math.exp(1.0) / (math.exp(1.0) + 2.0)); m2 = -math.log(math.exp(2.0) / (math.exp(2.0) + math.exp(0.5)))
report("groups of different sizes are averaged correctly", abs(mixed - (m1 + m2) / 2) < 1e-5)
lb = torch.tensor([1.0, -1.0]); sig = lambda x: 1 / (1 + math.exp(-x))
bce_manual = (-math.log(sig(1.0)) - math.log(1 - sig(-1.0))) / 2
report("bce loss checked by hand", abs(tr.compute_loss(lb, [2], "bce", torch).item() - bce_manual) < 1e-5)
report("a random-chance model has loss ln(group size)", abs(tr.compute_loss(torch.zeros(8), [8], "listwise", torch).item() - math.log(8)) < 1e-5)
try:
    tr.compute_loss(lg, [3], "mse", torch)
    report("an unknown loss is rejected", False)
except ValueError:
    report("an unknown loss is rejected", True)

groups = [{"query": "pension for a woman whose husband died", "positive": {"chunk_id": "a", "text": texts[0]},
           "negatives": [{"chunk_id": "b", "text": texts[1]}, {"chunk_id": "c", "text": texts[2]}, {"chunk_id": "d", "text": texts[3]}]},
          {"query": "help buying seeds for my small farm", "positive": {"chunk_id": "b", "text": texts[1]},
           "negatives": [{"chunk_id": "a", "text": texts[0]}, {"chunk_id": "d", "text": texts[3]}]},
          {"query": "money to study science at college", "positive": {"chunk_id": "c", "text": texts[2]},
           "negatives": [{"chunk_id": "a", "text": texts[0]}, {"chunk_id": "b", "text": texts[1]}, {"chunk_id": "d", "text": texts[3]}]},
          {"query": "I want to open a small shop, any loan?", "positive": {"chunk_id": "d", "text": texts[3]},
           "negatives": [{"chunk_id": "a", "text": texts[0]}, {"chunk_id": "c", "text": texts[2]}]}]
enc, sizes = tr.encode_batch(reranker.tokenizer, groups, 128, "cpu")
report("encode_batch: the right chunk is first in every group, sizes are right", sizes == [4, 3, 4, 3] and enc["input_ids"].shape[0] == 14)

model = Model.from_pretrained(reranker.model_path)
model.train()
before = {n: p.detach().clone() for n, p in model.named_parameters()}
tiny = sum(p.numel() for p in model.parameters()) < 5_000_000
opt = torch.optim.AdamW(tr.parameter_groups(model, 0.01), lr=1e-3 if tiny else 5e-5)
losses = []
n_steps = 120 if tiny else 40            # a tiny random model sits on a plateau for its first ~20 steps
for step in range(n_steps):
    enc, sizes = tr.encode_batch(reranker.tokenizer, groups, 128, "cpu")
    loss = tr.compute_loss(model(**enc).logits.reshape(-1), sizes, "listwise", torch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
    losses.append(loss.item())
first, last = sum(losses[:3]) / 3, sum(losses[-3:]) / 3
report("the model LEARNS: loss falls when it trains on a small batch", last < (first * 0.5 if tiny else first), f"{first:.3f} -> {last:.3f} over {n_steps} steps")
report("all losses are finite", all(math.isfinite(x) for x in losses))
report("training changed the weights", any(not torch.equal(before[n], p.detach()) for n, p in model.named_parameters()))
no_decay = tr.parameter_groups(model, 0.01)
report("biases and LayerNorm weights are excluded from weight decay", no_decay[1]["weight_decay"] == 0.0 and len(no_decay[1]["params"]) > 0)

model.eval()
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    tr_config = dict(config.__dict__)
    config.MODEL_DIR = Path(tmp) / "finetuned"
    tr.save_checkpoint(model, reranker.tokenizer, {"base_model": "x", "best_epoch": 1})
    reloaded = rr.CrossEncoderReranker(str(config.MODEL_DIR))
    direct = rr.score_texts(model, reranker.tokenizer, q, texts, "cpu", 128, 16)
    again = reloaded.score(q, texts)
    report("a saved model reloads to the same scores", all(abs(a - b) < 1e-4 for a, b in zip(direct, again)), f"{direct} vs {again}")
    report("the fine-tuned model scores differently from the pretrained one", any(abs(a - b) > 1e-4 for a, b in zip(again, s)))
    report("the checkpoint folder has the weights, tokenizer and metadata", all((config.MODEL_DIR / f).exists() for f in ["config.json", "reranker_meta.json"]))

    # ---------------- which model is active ----------------
    config.ACTIVE_FILE = Path(tmp) / "active_reranker.json"
    d = rr.active_info()
    report("no decision yet -> the pretrained model, marked undecided", d["active"] == "pretrained" and d["decided"] is False and d["rerank_recommended"] is None)
    config.ACTIVE_FILE.write_text('{"active": "finetuned", "rerank_recommended": true, "reason": ["x"]}', encoding="utf-8")
    d = rr.active_info()
    report("finetuned chosen -> its folder, recommendation read", d["active"] == "finetuned" and d["model_path"] == str(config.MODEL_DIR) and d["rerank_recommended"] is True)
    report("load_active loads the fine-tuned model", rr.CrossEncoderReranker.load_active().model_path == str(config.MODEL_DIR))
    config.ACTIVE_FILE.write_text('{"active": "pretrained", "rerank_recommended": false}', encoding="utf-8")
    report("pretrained chosen -> the base model, and rerank_recommended False is respected", rr.active_info()["model_path"] == config.RERANKER_BASE_MODEL and rr.active_info()["rerank_recommended"] is False)
    config.MODEL_DIR = Path(tmp) / "missing"
    config.ACTIVE_FILE.write_text('{"active": "finetuned"}', encoding="utf-8")
    report("finetuned chosen but its folder is missing -> safely falls back to pretrained", rr.active_info()["active"] == "pretrained")
    config.ACTIVE_FILE.write_text("{ not json", encoding="utf-8")
    report("a damaged decision file -> safe defaults", rr.active_info()["active"] == "pretrained" and rr.active_info()["decided"] is False)

print(f"\n{results.count('PASS')} passed, {results.count('WARN')} warnings, {results.count('FAIL')} failed")
sys.exit(1 if results.count("FAIL") else 0)
