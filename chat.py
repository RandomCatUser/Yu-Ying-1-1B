import argparse
import torch
from common import encode_prompt, load_tokenizer
from modeling_yuying import YuYingForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("path", nargs="?", default="yu-ying-1-1b-chat")
parser.add_argument("--system", default="You are Yu-Ying 1 : 1B, a helpful assistant created by Dihan Ramanayaka.")
parser.add_argument("--max_new_tokens", type=int, default=512)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--top_p", type=float, default=0.95)
parser.add_argument("--min_p", type=float, default=0.0)
parser.add_argument("--repetition_penalty", type=float, default=1.1)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
tok = load_tokenizer(args.path)
model = YuYingForCausalLM.from_pretrained(args.path, dtype=dtype).to(device).eval()
stop_ids = [tok.eos_token_id, tok.convert_tokens_to_ids("<|end|>")]


class Printer:
    def __init__(self):
        self.ids = []
        self.shown = ""

    def __call__(self, nxt):
        token = nxt[0, 0].item()
        if token in stop_ids:
            return
        self.ids.append(token)
        text = tok.decode(self.ids)
        if text.endswith("\ufffd"):
            return
        print(text[len(self.shown):], end="", flush=True)
        self.shown = text


history = [{"role": "system", "content": args.system}]
print("Yu-Ying 1 : 1B ready. Commands: /reset  /exit")
while True:
    try:
        user = input("\nYou: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not user:
        continue
    if user == "/exit":
        break
    if user == "/reset":
        history = history[:1]
        print("history cleared")
        continue
    history.append({"role": "user", "content": user})
    ids = encode_prompt(tok, history)[-(model.config.max_position_embeddings - args.max_new_tokens):]
    x = torch.tensor([ids], device=device)
    printer = Printer()
    print("Yu-Ying 1 : 1B: ", end="", flush=True)
    model.generate(
        x,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        stop_ids=stop_ids,
        streamer=printer,
    )
    print()
    history.append({"role": "assistant", "content": printer.shown})
