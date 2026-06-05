"""
Sparks of Babel — A Minimal GPT-Style Transformer

A from-scratch implementation of a decoder-only transformer, inspired by:
- "Attention Is All You Need" (Vaswani et al., 2017)
- GPT-2 (Radford et al., 2019)
- nanoGPT (Karpathy, 2023)
- CS336 at Stanford (Spring 2026)

This implementation is educational — every component is written
explicitly with clear variable names and documentation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class GPTConfig:
    """Configuration for a GPT-style transformer."""
    block_size: int = 256       # Maximum context length
    vocab_size: int = 1024      # Vocabulary size
    n_layer: int = 6            # Number of transformer layers
    n_head: int = 6             # Number of attention heads
    n_embd: int = 384           # Embedding dimension
    dropout: float = 0.1        # Dropout rate
    bias: bool = True           # Whether to use bias in Linears
    # n_embd must be divisible by n_head
    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"


class LayerNorm(nn.Module):
    """
    Layer Normalization with optional bias.
    Like PyTorch's LayerNorm but explicit for educational clarity.
    """
    
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.
    
    Causal means each token can only attend to itself and previous tokens.
    This is crucial for autoregressive language modeling.
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        
        # Key, Query, Value projections — all in one matrix for efficiency
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Causal mask (lower triangular)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
        )
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_embd)
        Returns:
            (batch, seq_len, n_embd)
        """
        B, T, C = x.shape
        
        # Compute Q, K, V
        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        # Apply causal mask
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Weighted sum of values
        y = att @ v  # (B, n_head, T, head_dim)
        
        # Re-assemble heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        
        return y


class MLP(nn.Module):
    """
    Feed-forward network with GELU activation.
    Uses 4x expansion as in GPT-2.
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    One transformer block: Attention + MLP with residual connections and LayerNorm.
    Uses pre-normalization (like modern GPTs) rather than post-normalization.
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention with residual
        x = x + self.attn(self.ln_1(x))
        # MLP with residual
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """
    A minimal GPT-style language model.
    
    Architecture:
        Token Embedding → Position Embedding → Dropout →
        [Transformer Block] × n_layer →
        LayerNorm → LM Head → Softmax
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),  # token embedding
            wpe=nn.Embedding(config.block_size, config.n_embd),   # position embedding
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying: share weights between token embedding and LM head
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled init to the residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        
        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPT model: {n_params/1e6:.2f}M parameters")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            idx: (batch, seq_len) input token IDs
            targets: (batch, seq_len) target token IDs, for computing loss
        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: scalar cross-entropy loss (if targets provided)
        """
        device = idx.device
        B, T = idx.shape
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"
        
        # Token + Position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)      # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)       # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)
        
        # Final LayerNorm
        x = self.transformer.ln_f(x)
        
        # LM head
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation.
        
        Args:
            idx: (batch, seq_len) initial context
            max_new_tokens: Number of tokens to generate
            temperature: Softmax temperature (lower = more deterministic)
            top_k: If set, only sample from top-k logits
        Returns:
            (batch, seq_len + max_new_tokens) generated sequence
        """
        for _ in range(max_new_tokens):
            # Crop to block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Softmax and sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    @torch.no_grad()
    def estimate_loss(self, data_loader, eval_iters: int = 50) -> float:
        """Estimate average loss over a data loader."""
        self.eval()
        losses = torch.zeros(eval_iters)
        for k, (X, Y) in enumerate(data_loader):
            if k >= eval_iters:
                break
            _, loss = self(X, Y)
            losses[k] = loss.item()
        self.train()
        return losses.mean().item()
    
    @torch.no_grad()
    def get_attention_maps(self, idx: torch.Tensor) -> list:
        """
        Extract attention maps from the last forward pass (for visualization).
        Returns list of attention weights per layer.
        This is a hook-based approach for simplicity.
        """
        # This requires hooks — we'll implement in the visualization script
        raise NotImplementedError("Use the visualization script for attention extraction")
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configure AdamW optimizer with weight decay separation.
        Only apply weight decay to matmul weights (2D params), not biases/norms.
        """
        # Separate parameters
        decay_params = []
        no_decay_params = []
        for pn, p in self.named_parameters():
            if not p.requires_grad:
                continue
            # Weight decay for weights with dim >= 2
            if p.dim() >= 2:
                decay_params.append(p)
            else:
                no_decay_params.append(p)
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        
        # Use fused AdamW if available (faster on CUDA)
        fused_available = 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else {}
        
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        
        return optimizer
