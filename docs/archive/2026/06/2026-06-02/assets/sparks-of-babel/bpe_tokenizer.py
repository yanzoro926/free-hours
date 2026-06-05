"""
BPE (Byte-Pair Encoding) Tokenizer — from scratch.

This implements the full BPE training and encoding/decoding pipeline:
1. Start with byte-level tokens (0-255)
2. Iteratively merge the most frequent adjacent token pair
3. Build a vocabulary of subword tokens
4. Use the merges to encode new text

Inspired by GPT-2's tokenizer and Sennrich et al. (2016).
"""

import json
from collections import Counter
from typing import List, Dict, Tuple, Optional


class BPETokenizer:
    """
    A minimal but complete BPE tokenizer.
    
    Usage:
        tokenizer = BPETokenizer()
        tokenizer.train(text, vocab_size=1000)
        ids = tokenizer.encode("Hello world")
        text = tokenizer.decode(ids)
    """
    
    def __init__(self):
        self.merges: Dict[Tuple[int, int], int] = {}  # (a,b) -> new_id
        self.vocab: Dict[int, bytes] = {}  # id -> bytes
        self._init_byte_vocab()
    
    def _init_byte_vocab(self):
        """Initialize with single-byte tokens (0-255)."""
        for i in range(256):
            self.vocab[i] = bytes([i])
    
    def _get_stats(self, ids: List[int]) -> Counter:
        """Count frequencies of adjacent pairs."""
        pairs = Counter()
        for pair in zip(ids[:-1], ids[1:]):
            pairs[pair] += 1
        return pairs
    
    def _merge(self, ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        """Replace all occurrences of pair with new_id."""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids
    
    def train(self, text: str, vocab_size: int, verbose: bool = True) -> None:
        """
        Train BPE on text to build a vocabulary of vocab_size tokens.
        
        Args:
            text: Training text (can be large — we work with bytes)
            vocab_size: Target vocabulary size (must be >= 256)
            verbose: Print progress
        """
        assert vocab_size >= 256, "vocab_size must be at least 256 (byte tokens)"
        
        # Convert text to bytes, then to list of ints
        text_bytes = text.encode('utf-8')
        ids = list(text_bytes)
        
        num_merges = vocab_size - 256
        
        if verbose:
            print(f"Training BPE: {len(text_bytes):,} bytes → {vocab_size} tokens ({num_merges} merges)")
            print(f"Initial unique byte values: {len(set(ids))}")
        
        for i in range(num_merges):
            stats = self._get_stats(ids)
            if not stats:
                break
            
            # Find most frequent pair
            top_pair = max(stats, key=stats.get)
            new_id = 256 + i
            
            # Record the merge
            self.merges[top_pair] = new_id
            self.vocab[new_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
            
            # Apply merge to the training data
            ids = self._merge(ids, top_pair, new_id)
            
            if verbose and (i + 1) % max(1, num_merges // 10) == 0:
                pct = (i + 1) / num_merges * 100
                print(f"  Merge {i+1}/{num_merges} ({pct:.0f}%) — "
                      f"pair ({top_pair[0]}, {top_pair[1]}) freq={stats[top_pair]} — "
                      f"new token: {self.vocab[new_id][:20]!r}")
        
        if verbose:
            print(f"Training complete. Vocabulary size: {len(self.vocab)}")
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text into token IDs.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        text_bytes = text.encode('utf-8')
        ids = list(text_bytes)
        
        # Apply merges in order (greedy)
        # We need to apply merges in the order they were learned
        # For each merge, scan and replace
        while len(ids) >= 2:
            # Find the earliest applicable merge
            stats = self._get_stats(ids)
            # Among pairs in stats, find the one with the lowest merge ID
            # (earliest learned merge gets priority)
            best_pair = None
            best_merge_id = float('inf')
            
            for pair in stats:
                if pair in self.merges:
                    merge_id = self.merges[pair]
                    if merge_id < best_merge_id:
                        best_merge_id = merge_id
                        best_pair = pair
            
            if best_pair is None:
                break
            
            ids = self._merge(ids, best_pair, self.merges[best_pair])
        
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        # Concatenate the byte sequences
        text_bytes = b''
        for id_ in ids:
            text_bytes += self.vocab.get(id_, bytes([id_]) if id_ < 256 else b'?')
        
        # Decode UTF-8, replacing errors
        return text_bytes.decode('utf-8', errors='replace')
    
    def save(self, path: str):
        """Save tokenizer state to file."""
        data = {
            'merges': {f'{a},{b}': c for (a,b), c in self.merges.items()},
            'vocab': {str(k): v.decode('latin-1') for k, v in self.vocab.items()},
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'BPETokenizer':
        """Load tokenizer from file."""
        with open(path) as f:
            data = json.load(f)
        
        tokenizer = cls()
        tokenizer.merges = {
            tuple(map(int, k.split(','))): v
            for k, v in data['merges'].items()
        }
        tokenizer.vocab = {
            int(k): v.encode('latin-1')
            for k, v in data['vocab'].items()
        }
        return tokenizer
    
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
    
    def token_bytes(self, id_: int) -> bytes:
        """Get the raw bytes for a token ID."""
        return self.vocab.get(id_, b'?')


if __name__ == '__main__':
    # Quick test
    tokenizer = BPETokenizer()
    sample = "Hello world! This is a test of the BPE tokenizer. " * 100
    tokenizer.train(sample, vocab_size=300, verbose=True)
    
    encoded = tokenizer.encode("Hello world!")
    decoded = tokenizer.decode(encoded)
    print(f"\nEncoded 'Hello world!': {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Matches: {decoded == 'Hello world!'}")
