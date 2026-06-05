"""
Training script — train a minimal GPT on Shakespeare.

This script:
1. Loads the Shakespeare dataset
2. Trains a BPE tokenizer on it
3. Prepares train/val splits
4. Trains a GPT model with a cosine learning rate schedule
5. Saves checkpoints and generates samples
"""

import os
import sys
import time
import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

# Add project dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bpe_tokenizer import BPETokenizer
from model import GPT, GPTConfig


# ─── Configuration ───────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Data
    data_path: str = "shakespeare.txt"
    vocab_size: int = 512           # BPE vocabulary size
    block_size: int = 128           # Context window
    
    # Model
    n_layer: int = 4                # Transformer layers
    n_head: int = 4                 # Attention heads
    n_embd: int = 256               # Embedding dimension
    
    # Training
    batch_size: int = 32            # Batch size
    max_iters: int = 5000           # Training iterations
    eval_interval: int = 500        # How often to evaluate
    eval_iters: int = 100           # Evaluation iterations
    log_interval: int = 50          # Logging interval
    
    # Optimization
    learning_rate: float = 3e-4     # Max learning rate
    min_lr: float = 1e-5            # Minimum LR for cosine schedule
    weight_decay: float = 1e-1      # Weight decay
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # System
    device: str = 'cpu'             # 'cpu', 'cuda', 'mps'
    dtype: str = 'float32'          # 'float32', 'bfloat16'
    compile: bool = False           # torch.compile()
    
    # Output
    out_dir: str = "out"
    sample_every: int = 1000        # Generate text samples
    save_every: int = 2000          # Save checkpoint


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_data(config: TrainConfig):
    """Load Shakespeare and prepare train/val splits."""
    print("=" * 60)
    print("Loading data...")
    print("=" * 60)
    
    # Load text
    with open(config.data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Raw text: {len(text):,} characters")
    
    # Train BPE tokenizer
    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=config.vocab_size, verbose=True)
    
    # Save tokenizer
    tokenizer.save(os.path.join(config.out_dir, 'tokenizer.json'))
    print(f"Saved tokenizer ({tokenizer.vocab_size} tokens)")
    
    # Encode full text
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"Encoded: {len(data):,} tokens")
    
    # Split: 90% train, 10% val
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(f"Train: {len(train_data):,} tokens, Val: {len(val_data):,} tokens")
    
    return tokenizer, train_data, val_data


def get_batch(split_data, batch_size, block_size, device):
    """Get a random batch of data."""
    data = split_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


# ─── Learning Rate Schedule ──────────────────────────────────────────────────

def get_lr(it, config: TrainConfig):
    """Cosine learning rate schedule with linear warmup."""
    warmup_iters = config.max_iters // 10
    
    # Linear warmup
    if it < warmup_iters:
        return config.learning_rate * it / warmup_iters
    
    # Cosine decay
    if it > config.max_iters:
        return config.min_lr
    
    decay_ratio = (it - warmup_iters) / (config.max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


# ─── Training Loop ────────────────────────────────────────────────────────────

def train(config: TrainConfig):
    """Main training loop."""
    
    # Setup
    os.makedirs(config.out_dir, exist_ok=True)
    device = config.device
    dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[config.dtype]
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[config.dtype]
    ctx = torch.amp.autocast(device_type=device, dtype=ptdtype) if device == 'cuda' else torch.no_grad()
    
    # Load data
    tokenizer, train_data, val_data = load_data(config)
    
    # Create model
    model_config = GPTConfig(
        block_size=config.block_size,
        vocab_size=tokenizer.vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=0.1,
    )
    
    model = GPT(model_config)
    model.to(device)
    
    # Compile if requested
    if config.compile and hasattr(torch, 'compile'):
        print("Compiling model...")
        model = torch.compile(model)
    
    # Optimizer
    optimizer = model.configure_optimizers(
        config.weight_decay, config.learning_rate, (config.beta1, config.beta2), device
    )
    
    # Training state
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Batch loader
    def get_batch_fn():
        return get_batch(train_data, config.batch_size, config.block_size, device)
    
    def get_val_batch_fn():
        return get_batch(val_data, config.batch_size, config.block_size, device)
    
    print("\n" + "=" * 60)
    print("Starting training")
    print("=" * 60)
    t0 = time.time()
    
    for iter_num in range(config.max_iters + 1):
        # Learning rate
        lr = get_lr(iter_num, config) if config.max_iters > 0 else config.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluation
        if iter_num % config.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                val_losses_ = []
                for _ in range(config.eval_iters):
                    X, Y = get_val_batch_fn()
                    _, loss = model(X, Y)
                    val_losses_.append(loss.item())
                val_loss = sum(val_losses_) / len(val_losses_)
                val_losses.append((iter_num, val_loss))
                
                train_losses_ = []
                for _ in range(min(config.eval_iters, 20)):
                    X, Y = get_batch_fn()
                    _, loss = model(X, Y)
                    train_losses_.append(loss.item())
                train_loss = sum(train_losses_) / len(train_losses_)
                train_losses.append((iter_num, train_loss))
                
                elapsed = time.time() - t0
                print(f"Step {iter_num:5d}/{config.max_iters} | "
                      f"train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
                      f"lr {lr:.1e} | time {elapsed:.1f}s")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'iter': iter_num,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'config': config,
                    }, os.path.join(config.out_dir, 'best_model.pt'))
                    print(f"  → Saved best model (val_loss={val_loss:.4f})")
            
            model.train()
        
        # Training step
        X, Y = get_batch_fn()
        _, loss = model(X, Y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        optimizer.step()
        
        # Logging
        if iter_num % config.log_interval == 0:
            print(f"  iter {iter_num:5d} | loss {loss.item():.4f} | lr {lr:.1e}")
        
        # Generate sample
        if iter_num % config.sample_every == 0 and iter_num > 0:
            prompt = "FIRST CITIZEN:\n"
            context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
            model.eval()
            with torch.no_grad():
                generated = model.generate(context, max_new_tokens=100, temperature=0.8, top_k=40)
            generated_text = tokenizer.decode(generated[0].tolist())
            print(f"\n{'─'*40}")
            print(f"Sample at iter {iter_num}:")
            print(generated_text[:300])
            print(f"{'─'*40}\n")
            model.train()
        
        # Save checkpoint
        if iter_num % config.save_every == 0 and iter_num > 0:
            torch.save({
                'iter': iter_num,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config,
            }, os.path.join(config.out_dir, f'checkpoint_{iter_num}.pt'))
    
    # Final save
    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Training complete! Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to {config.out_dir}/best_model.pt")
    print(f"{'='*60}")
    
    # Save losses for plotting
    torch.save({
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, os.path.join(config.out_dir, 'losses.pt'))
    
    return model, tokenizer, train_losses, val_losses


if __name__ == '__main__':
    config = TrainConfig(
        data_path="shakespeare.txt",
        out_dir="out",
        vocab_size=512,
        block_size=128,
        n_layer=4,
        n_head=4,
        n_embd=256,
        max_iters=3500,   # ~20-30 minutes on CPU
        batch_size=32,
        eval_interval=250,
        log_interval=25,
        sample_every=500,
        save_every=1000,
        device='cpu',
    )
    train(config)
