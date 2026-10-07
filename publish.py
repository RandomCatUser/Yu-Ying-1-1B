import os
import sys
from huggingface_hub import HfApi, login
from transformers import PreTrainedTokenizerFast
from modeling_heoles import HeolesConfig, HeolesForCausalLM

MODEL_DIR = "heoles1-1b"
REPO_NAME = "Heoles1-1B"
DISPLAY_NAME = "Heoles1:1B"
DEVELOPER = "Dihan Ramanayaka"

CARD = f"""---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
language:
- en
tags:
- heoles
- causal-lm
- custom_code
---

# {DISPLAY_NAME}

{DISPLAY_NAME} is a 1 billion parameter decoder-only language model developed by **{DEVELOPER}**.

## Architecture

- Parameters: about 1.0B
- Layers: 22
- Hidden size: 2048
- Attention: 16 heads with 4 key/value heads (grouped-query attention)
- Feed-forward: SwiGLU, intermediate size 5120
- Position encoding: rotary (RoPE)
- Normalization: RMSNorm
- Context length: 2048
- Vocabulary: 32000 (byte-level BPE)

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("YOUR_USERNAME/{REPO_NAME}")
model = AutoModelForCausalLM.from_pretrained("YOUR_USERNAME/{REPO_NAME}", trust_remote_code=True)

ids = tok("The future of AI is", return_tensors="pt").input_ids
out = model.generate(ids, max_new_tokens=64, temperature=0.8)
print(tok.decode(out[0]))
```

## Developer

{DEVELOPER}
"""


def main():
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    else:
        login()

    api = HfApi()
    username = sys.argv[1] if len(sys.argv) > 1 else api.whoami()["name"]
    repo_id = f"{username}/{REPO_NAME}"

    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    HeolesConfig.register_for_auto_class()
    HeolesForCausalLM.register_for_auto_class("AutoModelForCausalLM")

    model = HeolesForCausalLM.from_pretrained(MODEL_DIR)
    tok = PreTrainedTokenizerFast.from_pretrained(MODEL_DIR)

    model.push_to_hub(repo_id, safe_serialization=True)
    tok.push_to_hub(repo_id)

    card = CARD.replace("YOUR_USERNAME", username)
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"Published: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
