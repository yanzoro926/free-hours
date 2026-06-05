"""Quick test: load model and generate text."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import TrainConfig  # needed for unpickling
from bpe_tokenizer import BPETokenizer
from model import GPT, GPTConfig
import torch

tokenizer = BPETokenizer.load('out/tokenizer.json')
print(f"Tokenizer: {tokenizer.vocab_size} tokens")

config = GPTConfig(block_size=128, vocab_size=tokenizer.vocab_size, 
                    n_layer=4, n_head=4, n_embd=256, dropout=0.1)
model = GPT(config)

ckpt = torch.load('out/best_model.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state'])
model.eval()

print(f"Iter: {ckpt['iter']}, Val loss: {ckpt['val_loss']:.4f}")

prompts = [
    "FIRST CITIZEN:\n",
    "To be, or not to be",
    "KING HENRY:\n",
    "JULIET:\nO Romeo",
]

for prompt in prompts:
    ctx = torch.tensor([tokenizer.encode(prompt)])
    with torch.no_grad():
        gen = model.generate(ctx, max_new_tokens=60, temperature=0.8, top_k=40)
    text = tokenizer.decode(gen[0].tolist())
    print(f"\n{'─'*40}")
    print(text[:250])
