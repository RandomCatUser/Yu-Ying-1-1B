import json
import math
import os
import time
import torch
from datasets import load_dataset
from common import encode_conversation, env, load_tokenizer, pick_device, synthetic_conversations
from modeling_heoles import HeolesForCausalLM

BASE = env("BASE", "heoles1-1b")
OUT = env("SFT_OUT", "heoles1-1b-chat")
DATASET = env("SFT_DATASET", "HuggingFaceTB/smoltalk")
DATASET_CONFIG = env("SFT_CONFIG", "all")
MAX_LEN = env("MAX_LEN", 2048)
MICRO = env("SFT_MICRO", 4)
ACCUM = env("SFT_ACCUM", 8)
STEPS = env("SFT_STEPS", 3000)
LR = env("SFT_LR", 2e-5)
WARMUP = env("SFT_WARMUP", 100)
SAVE_EVERY = env("SFT_SAVE_EVERY", 500)
MAX_MINUTES = env("MAX_MINUTES", 0)
SYNTHETIC = env("SYNTHETIC", 0)
STATE = os.path.join(OUT, "state.json")


def conversations(seed):
    if SYNTHETIC:
        yield from synthetic_conversations(seed)
        return
    ds = load_dataset(DATASET, DATASET_CONFIG, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10000)
    for ex in ds:
        msgs = ex["messages"]
        if msgs and any(m["role"] == "assistant" for m in msgs):
            yield msgs


def batches(tok, seed):
    rows = []
    for msgs in conversations(seed):
        ids, labels = encode_conversation(tok, msgs, MAX_LEN)
        if all(l == -100 for l in labels):
            continue
        rows.append((ids, labels))
        if len(rows) == MICRO:
            width = max(len(i) for i, _ in rows)
            x = torch.full((MICRO, width), tok.pad_token_id, dtype=torch.long)
            y = torch.full((MICRO, width), -100, dtype=torch.long)
            for r, (i, l) in enumerate(rows):
                x[r, : len(i)] = torch.tensor(i)
                y[r, : len(l)] = torch.tensor(l)
            yield x, y
            rows = []


def lr_at(step):
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    progress = min(1.0, (step - WARMUP) / max(1, STEPS - WARMUP))
    return 0.1 * LR + 0.5 * 0.9 * LR * (1 + math.cos(math.pi * progress))


def save(model, tok, step):
    os.makedirs(OUT, exist_ok=True)
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    with open(STATE, "w") as f:
        json.dump({"step": step, "total": STEPS, "done": step >= STEPS}, f)


def main():
    start = 0
    source = BASE
    if os.path.exists(STATE):
        with open(STATE) as f:
            state = json.load(f)
        if state["done"]:
            print("sft already complete")
            return
        start = state["step"]
        source = OUT
        print(f"resuming sft from step {start}")

    device = pick_device()
    use_cuda = torch.cuda.is_available()
    device_type = "cuda" if use_cuda else "cpu"
    tok = load_tokenizer(source)
    model = HeolesForCausalLM.from_pretrained(source).to(device)
    model.enable_checkpointing(True)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.0, fused=use_cuda)
    data = batches(tok, 7 + start)
    t_run = time.time()
    done_steps = start
    for step in range(start, STEPS):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        running = 0.0
        for _ in range(ACCUM):
            x, y = next(data)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type, dtype=torch.bfloat16, enabled=use_cuda):
                loss = model(input_ids=x, labels=y).loss / ACCUM
            loss.backward()
            running += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        done_steps = step + 1
        if step % 10 == 0:
            print(f"sft step {step} loss {running:.4f} lr {lr_at(step):.2e}", flush=True)
        if done_steps % SAVE_EVERY == 0 and done_steps < STEPS:
            save(model, tok, done_steps)
        if MAX_MINUTES and (time.time() - t_run) / 60 >= MAX_MINUTES:
            break
    save(model, tok, done_steps)
    print(f"sft saved at step {done_steps}/{STEPS}", flush=True)


if __name__ == "__main__":
    main()
