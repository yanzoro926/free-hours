"""
Compare model quality across training checkpoints.

Shows how the model's text generation quality evolves from
random initialization (iter 0) through training (iter 3500).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import TrainConfig
from bpe_tokenizer import BPETokenizer
from model import GPT, GPTConfig
import torch

tokenizer = BPETokenizer.load('out/tokenizer.json')
config = GPTConfig(block_size=128, vocab_size=512, n_layer=4, n_head=4, n_embd=256, dropout=0.1)

prompt = "To be, or not to be"
temperature = 0.8

checkpoints = [
    ('out/checkpoint_1000.pt', 1000),
    ('out/checkpoint_2000.pt', 2000),
    ('out/checkpoint_3000.pt', 3000),
    ('out/best_model.pt', 3500),
]

print("=" * 70)
print("TEXT QUALITY EVOLUTION ACROSS TRAINING")
print("=" * 70)
print(f"Prompt: \"{prompt}\"")
print(f"Temperature: {temperature}")
print()

for ckpt_path, iter_num in checkpoints:
    if not os.path.exists(ckpt_path):
        print(f"  [{iter_num:5d}] checkpoint not found")
        continue
    
    model = GPT(config)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    
    val_loss = ckpt.get('val_loss', float('nan'))
    
    ctx = torch.tensor([tokenizer.encode(prompt)])
    with torch.no_grad():
        gen = model.generate(ctx, max_new_tokens=60, temperature=temperature, top_k=40)
    text = tokenizer.decode(gen[0].tolist())
    
    # Extract the continuation
    continuation = text[len(prompt):]
    
    print(f"[Iter {iter_num:5d}] val_loss={val_loss:.4f}")
    print(f"  {continuation[:150]}")
    print()

print("=" * 70)
print("Observe how:")
print("  - Iter 1000: Gibberish with word-like fragments")
print("  - Iter 2000: Emerging sentence structure")
print("  - Iter 3000: Coherent dialogue, real character names")
print("  - Iter 3500: Fluent Shakespeare-style text")
