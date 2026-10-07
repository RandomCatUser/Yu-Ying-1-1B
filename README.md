# Yu-Ying 1 : 1B

Yu-Ying 1 : 1B is a 1 billion parameter decoder-only language model written from scratch in PyTorch and compatible with Hugging Face `transformers`.

Developer: **Dihan Ramanayaka**

## What's new in v2

| Feature | Details |
|---|---|
| Local and global attention | 5 sliding-window (1024) layers for every 1 global layer, which keeps long-context memory low |
| Grouped-query attention | 16 query heads, 4 KV heads, head dim 128 |
| QK-norm | RMSNorm on queries and keys for stable training |
| Sandwich norms | RMSNorm before and after every attention and MLP block |
| GeGLU feed-forward | Intermediate size 5632 |
| Dual RoPE | Base 10k for local layers, 1M for global layers |
| Context length | 8192 tokens |
| Vocabulary | 49152 byte-level BPE, with chat special tokens |
| KV cache | Fast generation, with a trimmed cache on local layers |
| Training tools | Multi-GPU (DDP), gradient checkpointing, bf16, resume from checkpoint |
| Chat | Supervised fine-tuning script with assistant-only loss and a streaming chat CLI |

Total parameters: about 1.003B.

## Files

| File | Purpose |
|---|---|
| `modeling_yuying.py` | Config and model |
| `common.py` | Tokenizer building, chat format, conversation encoding |
| `train.py` | Pretraining (tokenizer plus model) |
| `sft.py` | Chat fine-tuning |
| `chat.py` | Interactive streaming chat |
| `publish.py` | Upload to the Hugging Face Hub |
| `requirements.txt` | Dependencies |

## Setup

```bash
pip install -r requirements.txt
```

## Step 1: Pretrain

Single GPU:

```bash
python train.py
```

Multiple GPUs:

```bash
torchrun --nproc_per_node=8 train.py
```

Data is streamed from FineWeb-Edu (70%) and Cosmopedia v2 (30%) via `HuggingFaceTB/smollm-corpus`. Edit `SOURCES` in `train.py` to add code, math, or multilingual datasets.

Settings can be overridden with environment variables, for example:

```bash
STEPS=50000 MICRO=2 ACCUM=32 LR=3e-4 python train.py
```

Checkpoints are saved to `yu-ying-1-1b/`. Running the same command again resumes from the last checkpoint.

## Step 2: Fine-tune for chat

```bash
python sft.py
```

This trains on `HuggingFaceTB/smoltalk` and computes loss only on assistant replies. The result is saved to `yu-ying-1-1b-chat/`.

## Step 3: Chat

```bash
python chat.py yu-ying-1-1b-chat
```

Commands: `/reset` clears history, `/exit` quits.

## Step 4: Publish to Hugging Face

```bash
export HF_TOKEN=your_write_token
python publish.py --dir yu-ying-1-1b-chat --repo Yu-Ying-1-1B
```

Options: `--username`, `--private`, `--fp32` (default upload is bf16, about 2 GB).

Hugging Face repo names cannot contain a colon or spaces, so the repo is `Yu-Ying-1-1B`, while the model is still called Yu-Ying 1 : 1B.

## Load from the Hub

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

repo = "your-username/Yu-Ying-1-1B"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, dtype=torch.bfloat16)

messages = [{"role": "user", "content": "Explain gravity in two sentences."}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids

stop = [tok.eos_token_id, tok.convert_tokens_to_ids("<|end|>")]
out = model.generate(ids, max_new_tokens=200, temperature=0.7, stop_ids=stop)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
```

## Hardware

- Training: one 80 GB GPU works with gradient checkpointing (on by default). 24 to 40 GB cards need a smaller `MICRO` and `BLOCK`.
- Inference: about 2 GB in bf16, runs on a laptop CPU slowly or any modern GPU quickly.

## Honest expectations

The architecture borrows ideas from modern open models such as Gemma, but a model's quality comes mostly from training data and compute, not code. Gemma-class models are trained on trillions of tokens. The default settings here train on roughly 2.6B tokens per 20,000 steps, which gives a small, working model. For results closer to the best 1B models, train on 20B to 100B+ tokens with a stronger data mix (code, math, multilingual) and then fine-tune.

## Testing status

The model code was tested on CPU with a tiny configuration: KV-cache outputs match full-forward outputs, gradient checkpointing matches normal gradients, save and reload is exact, tied embeddings work, and the full 1B config builds to 1.003B parameters. The dataset streaming, GPU training loop, and Hub upload were not run in my environment.

## License

Apache-2.0. See [LICENSE](LICENSE) for the full text.

## Train inside GitHub (Actions)

The repo includes two workflows in `.github/workflows/`.

### smoke-test (free, automatic)

Runs on every push and pull request on a standard GitHub runner. It checks the model code, then runs the full pipeline (tokenizer, pretraining with a stop and resume, fine-tuning, chat) on a tiny model with synthetic data. It does not train a real model.

### train (manual, needs a GPU runner)

GitHub's free runners have no GPU, so a real 1B training run needs one of these:

1. A **larger GitHub-hosted GPU runner** (paid; available on GitHub Team and Enterprise Cloud). Create it in Settings > Actions > Runners > New runner > New GitHub-hosted runner, choose a GPU machine, and give it a name such as `gpu-runner`.
2. A **self-hosted runner** on your own GPU machine or cloud instance, with a label such as `gpu-runner`. For a public repo, be careful about who can trigger workflows on self-hosted machines.

Setup:

1. Push this project to a GitHub repo.
2. Create a Hugging Face token with write access and add it as a repository secret named `HF_TOKEN` (Settings > Secrets and variables > Actions).
3. Open the Actions tab, choose **train**, click **Run workflow**, and set the `runner` input to your runner's name or label.

How it works:

- GitHub jobs are limited to 6 hours on hosted runners, so training runs in chunks. Each run trains for `max_minutes`, saves a checkpoint, and uploads it to a private Hugging Face repo called `<your-username>/yu-ying-1-train-state`.
- With `auto_continue` on, each run starts the next one, which pulls the checkpoint and resumes where the last run stopped. When pretraining finishes it moves on to fine-tuning, and when fine-tuning finishes it publishes the model to `<your-username>/Yu-Ying-1-1B`.
- If a run fails or you cancel it, run the workflow again with the same stage and it resumes from the last saved checkpoint (saved every 500 steps and at the end of each run).
- `stage` lets you run `pretrain`, `sft`, or `publish` on their own.

Notes:

- The data stream restarts with a new shuffle seed on each resume, so a resumed run does not replay the exact same documents.
- Fine-tuning resumes from the saved weights but restarts the optimizer, which is fine for a short run.
- The workflow was validated for syntax and the same code path was tested on CPU, but it has not run on a real GPU runner.

## Train for free

Free cloud GPU options (Kaggle, Hugging Face ZeroGPU, Google Colab) are documented in [FREE_TRAINING.md](FREE_TRAINING.md).
