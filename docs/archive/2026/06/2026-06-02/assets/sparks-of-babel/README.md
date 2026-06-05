# Sparks of Babel / 巴别星火

*A minimal GPT trained from scratch on Shakespeare — a free exploration session.*

## What is this?

This is a complete, from-scratch implementation of a GPT-style language model, built in a single 2.5-hour exploration session. It includes:

- **BPE Tokenizer** — Byte-Pair Encoding implemented from scratch (no external tokenizer library)
- **Transformer Model** — GPT decoder-only architecture with multi-head attention
- **Training Loop** — Cosine LR schedule, AdamW, gradient clipping
- **Visualizations** — Loss curves, embedding PCA, attention maps, generation gallery

## Quick Start

```bash
# Install dependencies (PyTorch, numpy, matplotlib, sklearn, tqdm)
pip install torch numpy matplotlib scikit-learn tqdm

# Train the model (takes ~35 minutes on CPU)
python train.py

# Generate visualizations
python visualize.py

# Quick test of the trained model
python quick_test.py
```

## Model Architecture

| Component | Value |
|-----------|-------|
| Layers | 4 |
| Attention heads | 4 |
| Embedding dimension | 256 |
| Vocabulary size | 512 (BPE) |
| Context window | 128 tokens |
| Parameters | 3,316,480 |
| Training data | Shakespeare (~1.1MB) |

## Results

**Final validation loss: 2.9719** (down from 6.2754 at initialization)

### Sample Output (Temperature 0.7)

```
JULIET:
O Romeo? What say'st thou sweet so?

Nurse:
He tongue better bloody his hand,
And give me me, with a little doth kiss.

KING HENRY VI:
Sir, was same worn
```

```
KING HENRY:
My lord, was I see that may be the part
To pray in such as death.

BRAKENBURY:
She were no content, as I hear must content
To see her heart to the great brother's
```

## Files

- `bpe_tokenizer.py` — BPE tokenizer implementation (~250 lines)
- `model.py` — GPT transformer model (~350 lines)
- `train.py` — Training loop (~250 lines)
- `visualize.py` — Visualization suite (~400 lines)
- `quick_test.py` — Load model and generate samples
- `out/` — Training outputs (checkpoints, visualizations, loss data)

## Why "Sparks of Babel"?

The Tower of Babel was humanity's attempt to reach heaven through language. Large language models are our modern Babel — a collective effort to build systems that understand and generate human language. 

"Sparks of Babel" evokes the first flickers of linguistic intelligence that emerge when a neural network begins to learn. At iteration 250, the model produces gibberish. By iteration 3500, it writes "What say'st thou sweet so?" — a question that Shakespeare himself might have penned.

These are the sparks.

## License

MIT — built as part of a free exploration session, June 2, 2026.
