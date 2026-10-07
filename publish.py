import argparse
import os
import torch
from huggingface_hub import HfApi, login
from modeling_yuying import YuYingConfig, YuYingForCausalLM
from common import load_tokenizer

DISPLAY_NAME = "Yu-Ying 1 : 1B"
DEVELOPER = "Dihan Ramanayaka"

CARD = """---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
language:
- en
tags:
- yuying
- causal-lm
- custom_code
- chat
---

# Yu-Ying 1 : 1B

Yu-Ying 1 : 1B is a 1 billion parameter decoder-only language model developed by **Dihan Ramanayaka**.

## Architecture

- Parameters: about 1.0B
- Layers: 20, with a 5:1 mix of sliding-window (1024) and global attention layers
- Hidden size: 2048, 16 query heads, 4 key/value heads (grouped-query attention), head dim 128
- Feed-forward: GeGLU, intermediate size 5632
- Norms: RMSNorm with pre and post normalization around every sublayer, plus QK-norm
- Positions: rotary embeddings with separate base frequencies for local (10k) and global (1M) layers
- Context length: 8192
- Vocabulary: 49152 byte-level BPE
- Tied input and output embeddings, KV cache for fast generation

## Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

repo = "{repo_id}"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, dtype=torch.bfloat16)

messages = [{{"role": "user", "content": "Explain gravity in two sentences."}}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids

stop = [tok.eos_token_id, tok.convert_tokens_to_ids("<|end|>")]
out = model.generate(ids, max_new_tokens=200, temperature=0.7, stop_ids=stop)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
```

## Developer

{developer}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="yu-ying-1-1b-chat")
    parser.add_argument("--repo", default="Yu-Ying-1-1B")
    parser.add_argument("--username", default=None)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    login(token=token) if token else login()

    api = HfApi()
    username = args.username or api.whoami()["name"]
    repo_id = f"{username}/{args.repo}"
    api.create_repo(repo_id=repo_id, repo_type="model", private=args.private, exist_ok=True)

    YuYingConfig.register_for_auto_class()
    YuYingForCausalLM.register_for_auto_class("AutoModelForCausalLM")

    model = YuYingForCausalLM.from_pretrained(args.dir)
    if not args.fp32:
        model = model.to(torch.bfloat16)
    tok = load_tokenizer(args.dir)

    model.push_to_hub(repo_id, safe_serialization=True)
    tok.push_to_hub(repo_id)

    api.upload_file(
        path_or_fileobj=CARD.format(repo_id=repo_id, developer=DEVELOPER).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Published: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
