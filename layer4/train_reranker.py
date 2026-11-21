"""
train_reranker.py  -  fine-tune the cross-encoder reranker with PyTorch.

Run:   python train_reranker.py                         (after make_training_data.py)
       python train_reranker.py --limit-groups 32 --limit-eval 20 --epochs 1      a quick trial

THE TRAINING LOOP (all of it is in this file, nothing is hidden in a library):
  for every epoch:
    shuffle the training groups; take them 4 at a time
    for each group  (a question, 1 right chunk, ~7 wrong chunks)  the model scores every (question, chunk) pair
    LOSS  listwise (default): a softmax over the group's scores; the loss is -log(probability of the right chunk)
          bce: each pair is judged right/wrong on its own
    loss.backward()  ->  clip the gradients  ->  AdamW step  ->  learning-rate schedule step
    then measure on the validation schemes: validation loss and MRR over the first-stage candidates
  keep the checkpoint with the best validation MRR; stop early when it has not improved for TRAIN_PATIENCE epochs.

Epoch 0 is the PRETRAINED model before any training: your safety net and the number to beat.
Time: roughly 15-60 minutes on a laptop CPU for 3 epochs (much less on a GPU). Progress is printed as it runs.
"""
import argparse
import csv
import json
import math
import random
import sys
import time
from datetime import datetime

import l4_config as config
import rerank_data as rd
import rerank_metrics as rm
from reranker import RerankerUnavailable, _torch, resolve_device, score_texts


def set_seed(seed, torch):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_batch(tokenizer, groups, max_length, device):
    """Tokenise every (question, passage) pair of the groups; `sizes` remembers how many passages each group has."""
    queries, texts, sizes = [], [], []
    for g in groups:
        passages = [g["positive"]["text"]] + [n["text"] for n in g["negatives"]]      # the right chunk is ALWAYS first
        queries += [g["query"]] * len(passages)
        texts += passages
        sizes.append(len(passages))
    enc = tokenizer(queries, texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return enc.to(device), sizes


def compute_loss(logits, sizes, loss_type, torch):
    F = torch.nn.functional
    if loss_type == "listwise":
        zero = torch.zeros(1, dtype=torch.long, device=logits.device)                 # index 0 = the right chunk
        return torch.stack([F.cross_entropy(chunk.unsqueeze(0), zero) for chunk in torch.split(logits, sizes)]).mean()
    if loss_type == "bce":
        labels = torch.cat([torch.tensor([1.0] + [0.0] * (s - 1)) for s in sizes]).to(logits.device)
        return F.binary_cross_entropy_with_logits(logits, labels)
    raise ValueError(f"loss must be 'listwise' or 'bce', got '{loss_type}'")


def mean_loss(model, tokenizer, groups, args, device, torch):
    """Average loss without learning (used for validation)."""
    if not groups:
        return float("nan")
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(groups), args.batch_groups):
            batch = groups[i:i + args.batch_groups]
            enc, sizes = encode_batch(tokenizer, batch, args.max_length, device)
            total += compute_loss(model(**enc).logits.reshape(-1), sizes, args.loss, torch).item() * len(batch)
            n += len(batch)
    return total / n


def ranking_metrics(model, tokenizer, items, args, device):
    """Mean ranking metrics of the model's order over the first-stage candidates of the evaluation questions."""
    model.eval()

    def order(item):
        scores = score_texts(model, tokenizer, item["query"], [c["text"] for c in item["candidates"]], device,
                             args.max_length, config.RERANKER_BATCH_SIZE)
        return rm.order_by_scores(scores)

    return rm.evaluate_items(items, order)[0]


def parameter_groups(model, weight_decay):
    no_decay = ("bias", "LayerNorm.weight", "layer_norm", "LayerNorm.bias")
    decay = [p for n, p in model.named_parameters() if p.requires_grad and not any(k in n for k in no_decay)]
    plain = [p for n, p in model.named_parameters() if p.requires_grad and any(k in n for k in no_decay)]
    return [{"params": decay, "weight_decay": weight_decay}, {"params": plain, "weight_decay": 0.0}]


def save_checkpoint(model, tokenizer, meta):
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.MODEL_DIR)
    tokenizer.save_pretrained(config.MODEL_DIR)
    (config.MODEL_DIR / "reranker_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="Fine-tune the cross-encoder reranker.")
    p.add_argument("--base-model", default=config.RERANKER_BASE_MODEL)
    p.add_argument("--epochs", type=int, default=config.TRAIN_EPOCHS)
    p.add_argument("--lr", type=float, default=config.TRAIN_LR)
    p.add_argument("--batch-groups", type=int, default=config.TRAIN_BATCH_GROUPS, help="questions per step")
    p.add_argument("--loss", choices=["listwise", "bce"], default=config.TRAIN_LOSS)
    p.add_argument("--max-length", type=int, default=config.RERANKER_MAX_LENGTH)
    p.add_argument("--patience", type=int, default=config.TRAIN_PATIENCE)
    p.add_argument("--device", default=config.RERANKER_DEVICE)
    p.add_argument("--limit-groups", type=int, help="use only N training groups (trial run)")
    p.add_argument("--limit-eval", type=int, help="use only N validation questions for ranking metrics (trial run)")
    args = p.parse_args()

    try:
        torch, Model, Tok = _torch()
        device = resolve_device(args.device)
    except RerankerUnavailable as e:
        raise SystemExit(f"PYTORCH PROBLEM: {e}")
    train_groups, val_groups = rd.read_jsonl(config.GROUPS_TRAIN), rd.read_jsonl(config.GROUPS_VAL)
    eval_val = rd.read_jsonl(config.EVAL_VAL)
    if not train_groups or not eval_val:
        raise SystemExit("No training data found. Run  python make_training_data.py  first.")
    if args.limit_groups:
        train_groups, val_groups = train_groups[:args.limit_groups], val_groups[:max(4, args.limit_groups // 4)]
    if args.limit_eval:
        eval_val = eval_val[:args.limit_eval]
    set_seed(config.SEED, torch)

    print(f"Base model: {args.base_model} | device: {device} | loss: {args.loss} | lr: {args.lr} | "
          f"{len(train_groups)} training groups, {len(val_groups)} validation groups, {len(eval_val)} validation questions")
    try:
        tokenizer = Tok.from_pretrained(args.base_model)
        model = Model.from_pretrained(args.base_model)
    except Exception as e:
        raise SystemExit(f"Could not load '{args.base_model}': {e}")
    if int(model.config.num_labels) != 1:
        raise SystemExit(f"'{args.base_model}' must output a single relevance score (num_labels=1).")
    model.to(device)

    # ---------------- epoch 0: the pretrained model, before any training ----------------
    history = []
    t0 = time.time()
    base_metrics = ranking_metrics(model, tokenizer, eval_val, args, device)
    base_loss = mean_loss(model, tokenizer, val_groups, args, device, torch)
    history.append({"epoch": 0, "train_loss": "", "val_loss": round(base_loss, 4), "val_mrr": round(base_metrics["mrr"], 4),
                    "val_hit@1": round(base_metrics["hit@1"], 4), "val_hit@3": round(base_metrics["hit@3"], 4),
                    "val_ndcg@5": round(base_metrics["ndcg@5"], 4), "seconds": round(time.time() - t0, 1)})
    first_stage = rm.evaluate_items(eval_val, lambda it: list(range(len(it["candidates"]))))[0]
    print(f"\nepoch 0 (pretrained, no training): val loss {base_loss:.4f} | val MRR {base_metrics['mrr']:.4f} | "
          f"Hit@1 {base_metrics['hit@1']:.3f}   [first-stage order alone: MRR {first_stage['mrr']:.4f}]")

    # ---------------- training ----------------
    optimizer = torch.optim.AdamW(parameter_groups(model, config.TRAIN_WEIGHT_DECAY), lr=args.lr)
    steps_per_epoch = math.ceil(len(train_groups) / args.batch_groups)
    total_steps = max(1, steps_per_epoch * args.epochs)
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, int(config.TRAIN_WARMUP * total_steps), total_steps)
    rng = random.Random(config.SEED)
    best = {"key": None, "epoch": 0, "metrics": None}
    stale, step = 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = train_groups[:]
        rng.shuffle(order)
        running, n_seen, started = 0.0, 0, time.time()
        for i in range(0, len(order), args.batch_groups):
            batch = order[i:i + args.batch_groups]
            enc, sizes = encode_batch(tokenizer, batch, args.max_length, device)
            loss = compute_loss(model(**enc).logits.reshape(-1), sizes, args.loss, torch)
            if not torch.isfinite(loss):
                raise SystemExit("The loss became NaN/infinite. Lower the learning rate (--lr 1e-5) and try again.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item() * len(batch)
            n_seen += len(batch)
            step += 1
            if step % 10 == 0 or i + args.batch_groups >= len(order):
                done = (i + len(batch)) / len(order)
                eta = (time.time() - started) / max(done, 1e-9) * (1 - done)
                print(f"   epoch {epoch}/{args.epochs}  step {step}/{total_steps}  train loss {running / n_seen:.4f}  "
                      f"(~{eta / 60:.1f} min left in this epoch)")
        train_loss = running / max(1, n_seen)
        v_loss = mean_loss(model, tokenizer, val_groups, args, device, torch)
        v = ranking_metrics(model, tokenizer, eval_val, args, device)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4), "val_loss": round(v_loss, 4),
                        "val_mrr": round(v["mrr"], 4), "val_hit@1": round(v["hit@1"], 4), "val_hit@3": round(v["hit@3"], 4),
                        "val_ndcg@5": round(v["ndcg@5"], 4), "seconds": round(time.time() - started, 1)})
        print(f"epoch {epoch}: train loss {train_loss:.4f} | val loss {v_loss:.4f} | val MRR {v['mrr']:.4f} | "
              f"Hit@1 {v['hit@1']:.3f} | Hit@3 {v['hit@3']:.3f}")
        key = (round(v["mrr"], 6), -v_loss)
        if best["key"] is None or key > best["key"]:
            best = {"key": key, "epoch": epoch, "metrics": v}
            stale = 0
            save_checkpoint(model, tokenizer, {
                "base_model": args.base_model, "loss": args.loss, "lr": args.lr, "batch_groups": args.batch_groups,
                "max_length": args.max_length, "best_epoch": epoch, "n_train_groups": len(train_groups),
                "n_val_groups": len(val_groups), "n_val_questions": len(eval_val),
                "val_metrics": v, "pretrained_val_metrics": base_metrics, "first_stage_val_metrics": first_stage,
                "torch": torch.__version__, "seed": config.SEED, "device": device,
                "trained_at": datetime.now().isoformat(timespec="seconds")})
            print("   -> new best, checkpoint saved")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"   no improvement for {args.patience} epoch(s): stopping early")
                break

    with open(config.HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    print(f"\nPretrained val MRR {base_metrics['mrr']:.4f}  ->  best fine-tuned val MRR {best['metrics']['mrr']:.4f} "
          f"(epoch {best['epoch']})   first-stage alone {first_stage['mrr']:.4f}")
    print(f"Saved the best model to {config.MODEL_DIR} and the log to {config.HISTORY_CSV.name}")
    print("Next: python evaluate_reranker.py   (decides whether the fine-tuned model is actually better, on the unseen test schemes)")


if __name__ == "__main__":
    sys.exit(main())
