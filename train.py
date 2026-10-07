import contextlib
import json
import math
import os
import random
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from common import build_tokenizer, env, load_tokenizer, pick_device, synthetic_texts
from modeling_heoles import HeolesConfig, HeolesForCausalLM

OUT = env("OUT", "heoles1-1b")
BLOCK = env("BLOCK", 2048)
MICRO = env("MICRO", 4)
ACCUM = env("ACCUM", 16)
STEPS = env("STEPS", 20000)
RUN_STEPS = env("RUN_STEPS", 0)
MAX_MINUTES = env("MAX_MINUTES", 0)
LR = env("LR", 3e-4)
MIN_LR = env("MIN_LR", 3e-5)
WARMUP = env("WARMUP", 500)
SAVE_EVERY = env("SAVE_EVERY", 500)
CHECKPOINTING = env("CHECKPOINTING", 1)
VOCAB = env("VOCAB", 49152)
SEED = env("SEED", 1234)
SYNTHETIC = env("SYNTHETIC", 0)
PRESET = env("PRESET", "full")

PRESETS = {
    "tiny": dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        sliding_window=16,
        global_every=3,
        max_position_embeddings=256,
    )
}

SOURCES = [
    ("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", 0.7),
    ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", 0.3),
]


def make_config():
    return HeolesConfig(vocab_size=VOCAB, **PRESETS.get(PRESET, {}))


def open_source(repo, config, rank, world, seed):
    ds = load_dataset(repo, config, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10000)
    return iter(split_dataset_by_node(ds, rank=rank, world_size=world))


def mixed_texts(rank, world, seed):
    if SYNTHETIC:
        yield from synthetic_texts(seed + rank)
        return
    iters = [open_source(r, c, rank, world, seed) for r, c, _ in SOURCES]
    weights = [w for _, _, w in SOURCES]
    rng = random.Random(seed + rank)
    while iters:
        i = rng.choices(range(len(iters)), weights)[0]
        try:
            yield next(iters[i])["text"]
        except StopIteration:
            iters.pop(i)
            weights.pop(i)


def tokenizer_corpus(per_source):
    if SYNTHETIC:
        for _, text in zip(range(3000), synthetic_texts(99)):
            yield text
        return
    for repo, config, _ in SOURCES:
        ds = load_dataset(repo, config, split="train", streaming=True)
        for i, ex in enumerate(ds):
            if i >= per_source:
                break
            yield ex["text"]


def batches(tok, rank, world, seed):
    texts = mixed_texts(rank, world, seed)
    buf = []
    rows = []
    while True:
        pending = []
        try:
            while len(pending) < 64:
                pending.append(next(texts))
        except StopIteration:
            if not pending:
                return
        for ids in tok(pending, add_special_tokens=False)["input_ids"]:
            buf.append(tok.bos_token_id)
            buf.extend(ids)
            buf.append(tok.eos_token_id)
        while len(buf) >= BLOCK:
            rows.append(buf[:BLOCK])
            buf = buf[BLOCK:]
            if len(rows) == MICRO:
                yield torch.tensor(rows)
                rows = []


def lr_at(step):
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    progress = min(1.0, (step - WARMUP) / max(1, STEPS - WARMUP))
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * progress))


def write_state(step):
    with open(os.path.join(OUT, "state.json"), "w") as f:
        json.dump({"step": step, "total": STEPS, "done": step >= STEPS}, f)


def save_all(model, opt, step, tok):
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, os.path.join(OUT, "ckpt.pt"))
    write_state(step)


def main():
    distributed = "RANK" in os.environ
    use_cuda = torch.cuda.is_available()
    if distributed:
        dist.init_process_group("nccl" if use_cuda else "gloo")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
        if use_cuda:
            torch.cuda.set_device(local)
    else:
        rank, world, local = 0, 1, 0
    device = pick_device(local)
    device_type = "cuda" if use_cuda else "cpu"
    main_proc = rank == 0

    tok_path = os.path.join(OUT, "tokenizer.json")
    if main_proc and not os.path.exists(tok_path):
        build_tokenizer(tokenizer_corpus(150000), OUT, VOCAB)
    if distributed:
        dist.barrier()
    tok = load_tokenizer(OUT)

    torch.manual_seed(SEED)
    model = HeolesForCausalLM(make_config()).to(device)
    model.enable_checkpointing(bool(CHECKPOINTING))
    if main_proc:
        print(f"Heoles1:1B parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.3f}B", flush=True)

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=LR,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=use_cuda,
    )

    start = 0
    ckpt_path = os.path.join(OUT, "ckpt.pt")
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start = state["step"]
        del state
        if main_proc:
            print(f"resumed from step {start}", flush=True)

    if start >= STEPS:
        if main_proc:
            print("training already complete", flush=True)
            write_state(start)
        if distributed:
            dist.destroy_process_group()
        return

    net = DDP(model, device_ids=[local] if use_cuda else None) if distributed else model
    data = batches(tok, rank, world, SEED + start)

    net.train()
    t_run = time.time()
    t0 = time.time()
    done_steps = start
    for step in range(start, STEPS):
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        running = 0.0
        for micro in range(ACCUM):
            x = next(data).to(device)
            sync = micro == ACCUM - 1
            ctx = net.no_sync() if (distributed and not sync) else contextlib.nullcontext()
            with ctx:
                with torch.autocast(device_type, dtype=torch.bfloat16, enabled=use_cuda):
                    loss = net(input_ids=x, labels=x).loss / ACCUM
                loss.backward()
            running += loss.item()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        done_steps = step + 1

        if main_proc and step % 10 == 0:
            elapsed = time.time() - t0
            tokens = MICRO * ACCUM * BLOCK * world * 10
            print(f"step {step} loss {running:.4f} lr {lr_at(step):.2e} tok/s {tokens / max(elapsed, 1e-6):.0f}", flush=True)
            t0 = time.time()
        if main_proc and done_steps % SAVE_EVERY == 0 and done_steps < STEPS:
            save_all(model, opt, done_steps, tok)

        stop = 0
        if RUN_STEPS and done_steps - start >= RUN_STEPS:
            stop = 1
        if MAX_MINUTES and (time.time() - t_run) / 60 >= MAX_MINUTES:
            stop = 1
        if distributed:
            flag = torch.tensor([stop], device=device)
            dist.broadcast(flag, 0)
            stop = int(flag.item())
        if stop:
            break

    if main_proc:
        save_all(model, opt, done_steps, tok)
        print(f"saved at step {done_steps}/{STEPS}", flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
