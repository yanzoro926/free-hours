"""
Markov Muse — A Markov chain text generator that creates
surprisingly coherent prose from codebases and technical documents.

Built for the final hour of the 2026-06-06 exploration session.
"""

import re
import random
from collections import defaultdict
from typing import Optional


class MarkovChain:
    """N-gram Markov chain for text generation."""

    def __init__(self, n: int = 3):
        self.n = n
        self.chain: dict[tuple, list[str]] = defaultdict(list)
        self.starts: list[tuple] = []

    def train(self, text: str):
        """Train on text, building n-gram transitions."""
        # Tokenize: split on word boundaries, preserving punctuation as tokens
        tokens = re.findall(r"[\w']+|[.,!?;:\-—()\[\]{}\"'\n]", text)
        tokens = [t for t in tokens if t.strip()]

        if len(tokens) <= self.n:
            return

        for i in range(len(tokens) - self.n):
            key = tuple(tokens[i : i + self.n])
            next_token = tokens[i + self.n]
            self.chain[key].append(next_token)

            # Record sentence starts (after punctuation or newlines)
            if i == 0 or tokens[i - 1] in {".", "!", "?", "\n"}:
                self.starts.append(key)

        if not self.starts:
            self.starts = [tuple(tokens[: self.n])]

    def generate(self, max_tokens: int = 100, seed: Optional[tuple] = None) -> str:
        """Generate text from the Markov chain."""
        if not self.chain:
            return "(no training data)"

        if seed and seed in self.chain:
            current = seed
        elif self.starts:
            current = random.choice(self.starts)
        else:
            current = random.choice(list(self.chain.keys()))

        result = list(current)

        for _ in range(max_tokens):
            if current not in self.chain:
                break
            next_token = random.choice(self.chain[current])
            result.append(next_token)
            current = tuple(result[-self.n :])

        return self._format_output(result)

    def _format_output(self, tokens: list[str]) -> str:
        """Format tokens into readable text."""
        output = []
        for token in tokens:
            if token in {".", "!", "?", ",", ";", ":", "\n"}:
                if output:
                    output[-1] += token
                else:
                    output.append(token)
            elif token in {"'", '"'}:
                if output:
                    output[-1] += token
            elif output and output[-1] in {"(", "[", "{", '"', "'"}:
                output[-1] += token
            elif token in {")", "]", "}"}:
                if output:
                    output[-1] += token
            else:
                if output and not output[-1].endswith(("\n", "(")):
                    output.append(" ")
                output.append(token)

        text = "".join(output).strip()
        # Capitalize sentences
        text = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
        return text


def train_from_files(file_paths: list[str], n: int = 3) -> MarkovChain:
    """Train a Markov chain from multiple files."""
    chain = MarkovChain(n=n)
    all_text = []
    for path in file_paths:
        try:
            with open(path) as f:
                all_text.append(f.read())
        except Exception:
            pass
    chain.train("\n".join(all_text))
    return chain


def main():
    import sys
    import os

    # Collect Python files from today's projects
    base = os.path.dirname(os.path.abspath(__file__))
    today_dir = os.path.dirname(base)

    py_files = []
    for root, _, files in os.walk(today_dir):
        for f in files:
            if f.endswith(".py"):
                py_files.append(os.path.join(root, f))

    print("=" * 60)
    print("  Markov Muse — Codebase Poetry Generator")
    print("=" * 60)
    print(f"\nTraining on {len(py_files)} Python files...")

    chain = MarkovChain(n=3)
    all_text = []
    for path in py_files:
        try:
            with open(path) as f:
                all_text.append(f.read())
        except Exception:
            pass
    chain.train("\n".join(all_text))

    print(f"Chain size: {len(chain.chain):,} states")
    print(f"Start states: {len(chain.starts):,}")
    print()

    # Generate multiple samples
    titles = [
        "A Durable Dream",
        "Whispers in the Token Stream",
        "Vector Nights",
        "The Checkpoint's Lament",
        "What the Retry Saw",
    ]

    for i, title in enumerate(titles, 1):
        print(f"── {i}. {title} ──")
        text = chain.generate(max_tokens=60)
        print(text)
        print()

    # Bonus: generate from a specific seed
    print("── Bonus: 'Durable execution' seed ──")
    seed = ("Durable", "execution", "is")
    if seed in chain.chain:
        text = chain.generate(max_tokens=50, seed=seed)
    else:
        text = chain.generate(max_tokens=50)
    print(text)


if __name__ == "__main__":
    main()
