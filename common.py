import os
import random
import torch
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]

CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}<|end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


def build_tokenizer(texts, out_dir, vocab_size):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "tokenizer.json")
    bt = ByteLevelBPETokenizer()
    bt.train_from_iterator(texts, vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIAL_TOKENS)
    bt.save(path)
    return load_tokenizer(out_dir, path, save=True)


def load_tokenizer(out_dir, path=None, save=False):
    path = path or os.path.join(out_dir, "tokenizer.json")
    tok = PreTrainedTokenizerFast(
        tokenizer_file=path,
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        additional_special_tokens=SPECIAL_TOKENS[3:],
    )
    tok.chat_template = CHAT_TEMPLATE
    if save:
        tok.save_pretrained(out_dir)
    return tok


def encode_turn(tok, role, content):
    head = tok(f"<|{role}|>\n", add_special_tokens=False)["input_ids"]
    body = tok(content + "<|end|>\n", add_special_tokens=False)["input_ids"]
    return head, body


def encode_conversation(tok, messages, max_len):
    ids = [tok.bos_token_id]
    labels = [-100]
    for m in messages:
        head, body = encode_turn(tok, m["role"], m["content"])
        ids += head + body
        labels += [-100] * len(head)
        labels += body if m["role"] == "assistant" else [-100] * len(body)
    return ids[:max_len], labels[:max_len]


def encode_prompt(tok, messages):
    ids = [tok.bos_token_id]
    for m in messages:
        head, body = encode_turn(tok, m["role"], m["content"])
        ids += head + body
    ids += tok("<|assistant|>\n", add_special_tokens=False)["input_ids"]
    return ids


def env(name, default):
    return type(default)(os.environ.get(name, default))


def pick_device(local=0):
    return f"cuda:{local}" if torch.cuda.is_available() else "cpu"


WORDS = (
    "the model learns language from text and data while training on many steps with small batches "
    "of tokens so that attention layers can predict the next word in a long sentence about science "
    "music history code math and everyday life"
).split()


def synthetic_texts(seed=0):
    rng = random.Random(seed)
    while True:
        n = rng.randint(30, 120)
        yield " ".join(rng.choice(WORDS) for _ in range(n)) + "."


def synthetic_conversations(seed=0):
    rng = random.Random(seed)
    while True:
        q = " ".join(rng.choice(WORDS) for _ in range(rng.randint(4, 12))) + "?"
        a = " ".join(rng.choice(WORDS) for _ in range(rng.randint(8, 30))) + "."
        yield [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
