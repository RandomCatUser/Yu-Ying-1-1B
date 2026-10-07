import sys
import torch
from transformers import PreTrainedTokenizerFast
from modeling_heoles import HeolesForCausalLM

path = sys.argv[1] if len(sys.argv) > 1 else "heoles1-1b"
device = "cuda" if torch.cuda.is_available() else "cpu"

tok = PreTrainedTokenizerFast.from_pretrained(path)
model = HeolesForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device)

while True:
    prompt = input("You: ")
    if prompt.strip().lower() in {"exit", "quit"}:
        break
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(ids, max_new_tokens=128, temperature=0.8, top_k=50, top_p=0.95)
    print("Heoles1:1B:", tok.decode(out[0], skip_special_tokens=True))
