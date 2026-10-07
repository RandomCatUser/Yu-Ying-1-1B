import math
import os
import torch
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast
from modeling_heoles import HeolesConfig, HeolesForCausalLM

OUT = "heoles1-1b"
BLOCK = 2048
MICRO = 4
ACCUM = 16
STEPS = 20000
LR = 3e-4
WARMUP = 500
DATASET = "HuggingFaceFW/fineweb-edu"
SUBSET = "sample-10BT"


def stream():
    return load_dataset(DATASET, name=SUBSET, split="train", streaming=True)


def text_iter(n):
    for i, ex in enumerate(stream()):
        if i >= n:
            break
        yield ex["text"]


def build_tokenizer():
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/tokenizer.json"
    bt = ByteLevelBPETokenizer()
    bt.train_from_iterator(
        text_iter(200000),
        vocab_size=32000,
        min_frequency=2,
        special_tokens=["<pad>", "<s>", "</s>"],
    )
    bt.save(path)
    tok = PreTrainedTokenizerFast(
        tokenizer_file=path,
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )
    tok.save_pretrained(OUT)
    return tok


def batches(tok):
    ds = stream().shuffle(seed=42, buffer_size=10000)
    buf = []
    rows = []
    for ex in ds:
        buf.extend(tok(ex["text"])["input_ids"])
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
    progress = (step - WARMUP) / max(1, STEPS - WARMUP)
    floor = 0.1 * LR
    return floor + 0.5 * (LR - floor) * (1 + math.cos(math.pi * progress))


def main():
    tok = build_tokenizer()
    device = "cuda"
    model = HeolesForCausalLM(HeolesConfig()).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Heoles1:1B parameters: {total / 1e9:.3f}B")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=LR,
        betas=(0.9, 0.95),
        fused=True,
    )

    data = batches(tok)
    model.train()
    for step in range(STEPS):
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        running = 0.0
        for _ in range(ACCUM):
            x = next(data).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=x, labels=x).loss / ACCUM
            loss.backward()
            running += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 10 == 0:
            print(f"step {step} loss {running:.4f} lr {lr_at(step):.2e}")
        if step > 0 and step % 1000 == 0:
            model.save_pretrained(OUT)

    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)


if __name__ == "__main__":
    main()
