# Train for free (no GPU on your own PC)

All options below train the model without touching your own machine. Limits first, then steps:

| Option | Free GPU | Realistic for |
|---|---|---|
| Kaggle Notebooks | 30 GPU-hours per week (P100 or 2× T4), sessions up to 12 hours | A real pretraining or fine-tuning run |
| Hugging Face ZeroGPU Space | 5 GPU-minutes per day (RTX Pro 6000 Blackwell, 48 GB) | Tiny runs, pipeline tests, chat demos |
| Google Colab (free tier) | NVIDIA T4 sessions, quotas change often | Quick experiments |
| GitHub Actions | CPU only | Automated smoke tests (see README) |

The practical free answer for a full run is Kaggle plus this repo's `sync.py`, which stores checkpoints in a private Hugging Face repo so training can be paused and resumed across sessions.

## Option 1: Kaggle

1. Push this project to a GitHub repo.
2. On kaggle.com open **Code → New Notebook → Add Input → From GitHub** and pick the repo (or upload the project as a zip).
3. In the right sidebar set **Accelerator** to GPU (T4 ×2 or P100) and turn **Internet** on, because datasets stream from the Hugging Face Hub.
4. Put a Hugging Face write token in the notebook so checkpoints can be saved:

```python
import os
os.environ["HF_TOKEN"] = "hf_..."  # Add-ons > Secrets > Add Secret
```

5. Train in chunks that fit inside one 12-hour session:

```bash
!pip install -r requirements.txt
!python sync.py pull yu-ying-1-1b     # nothing on the first run, starts fresh
!MAX_MINUTES=600 python train.py      # trains ~10 hours, then saves and exits
!python sync.py push yu-ying-1-1b
```

6. Next day, new session, repeat the same four lines. `train.py` resumes from the last checkpoint (saved every 500 steps), so at most 500 steps are lost if a session dies.
7. When `yu-ying-1-1b/state.json` shows `"done": true`, fine-tune the same way with `sync.py push/pull yu-ying-1-1b-chat` and `python sft.py`, then publish with `python publish.py --dir yu-ying-1-1b-chat --repo Yu-Ying-1-1B`.

Notes: a full 20,000-step pretraining run needs on the order of 100 GPU-hours, so expect several weeks of the 30 hour weekly quota. Lower `STEPS` for a faster first model, and use `PRESET=tiny` to check the whole pipeline before spending quota.

## Option 2: Hugging Face ZeroGPU Space

ZeroGPU gives free Spaces a shared RTX Pro 6000 Blackwell. Free accounts can host up to 2 ZeroGPU Spaces (verified email, account older than 30 days), but the daily quota is only 5 GPU-minutes (PRO: 40 minutes), reset 24 hours after your first GPU use. That is enough to validate the pipeline, not enough to pretrain a 1B model (5 minutes a day is about 2.5 hours a month).

1. **New Space → SDK Gradio → Hardware ZeroGPU** (Settings → Hardware).
2. Push this project's files into the Space; `requirements.txt` is installed on build. ZeroGPU supports the Gradio SDK only, Python 3.10/3.12 and PyTorch 2.8+.
3. Wrap training chunks in a GPU function with an explicit quota duration:

```python
import os, subprocess
import spaces

def run(cmd, **env):
    subprocess.run(cmd, shell=True, check=True, env={**os.environ, **env})

@spaces.GPU(duration=300)  # seconds of quota per call
def train_chunk():
    run("python sync.py pull yu-ying-1-1b")
    run("python train.py", MAX_MINUTES="4", PRESET="tiny", STEPS="50")
    run("python sync.py push yu-ying-1-1b")
```

4. Call `train_chunk()` from a Gradio button or at startup, once per day, and the state accumulates in the private HF repo exactly as in the GitHub workflow.

## Option 3: Google Colab

Runtime → Change runtime type → T4 GPU, then clone the repo and run the same `sync.py pull` / `train.py` / `sync.py push` lines as Kaggle. Colab free sessions are shorter and the limits move often, so push state before every session ends; the runtime disk is wiped when it disconnects.

## What does not work for free

- Free GitHub Actions runners and free HF CPU Spaces have no GPU: they can run the smoke tests, but a 1B training run would take months.
- ZeroGPU's 5 minutes a day is for demos and tests, not pretraining.
- Everything else with real free GPU hours (Kaggle, Colab) has session limits, which is why this repo supports checkpoint-everything-and-resume through `sync.py`.
