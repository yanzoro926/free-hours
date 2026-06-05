"""
Token Whisperer — Intelligent prompt compression for LLM input optimization.

Strategies:
1. Whitespace normalization
2. Comment/docstring removal (code-aware)
3. Redundancy detection (near-duplicate sentences)
4. Abbreviation expansion reversal
5. Token count estimation (BPE-like heuristic)

Inspired by Lowfat CLI and the growing need for token-efficient LLM usage.
"""

import re
import sys
from typing import Optional
from dataclasses import dataclass, field
from difflib import SequenceMatcher


# Approximate BPE token counts per language (tokens per char)
TOKEN_DENSITY = {
    "english": 0.25,  # ~4 chars per token
    "code_python": 0.55,  # ~1.8 chars per token
    "code_javascript": 0.50,
    "code_json": 0.40,
    "mixed": 0.35,
}


@dataclass
class CompressionResult:
    original: str
    compressed: str
    original_tokens: int
    compressed_tokens: int
    strategies_applied: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def savings_pct(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return (1 - self.compressed_tokens / self.original_tokens) * 100

    def report(self) -> str:
        lines = [
            f"Original:  {self.original_tokens:>6} tokens ({len(self.original)} chars)",
            f"Compressed: {self.compressed_tokens:>6} tokens ({len(self.compressed)} chars)",
            f"Saved:     {self.original_tokens - self.compressed_tokens:>6} tokens ({self.savings_pct:.1f}%)",
            "",
            f"Strategies: {', '.join(self.strategies_applied)}",
            f"Stats: {self.stats}",
        ]
        return "\n".join(lines)


class TokenWhisperer:
    """
    Compresses prompts to reduce LLM token usage while preserving semantic content.

    Usage:
        tw = TokenWhisperer()
        result = tw.compress("Your long prompt here...")
        print(result.report())
        print(result.compressed)
    """

    def __init__(self, aggressive: bool = False):
        self.aggressive = aggressive
        self.strategies = [
            self._normalize_whitespace,
            self._remove_comments,
            self._remove_redundancy,
            self._compress_repetition,
            self._abbreviate_common,
        ]
        if aggressive:
            self.strategies.append(self._summarize_bullets)
            self.strategies.append(self._remove_filler)

    def estimate_tokens(self, text: str, lang: str = "english") -> int:
        """Estimate token count using language-specific heuristics."""
        density = TOKEN_DENSITY.get(lang, 0.25)
        return max(1, int(len(text) * density))

    def compress(self, text: str, lang: str = "mixed") -> CompressionResult:
        """Apply all compression strategies in sequence."""
        original_tokens = self.estimate_tokens(text, lang)
        compressed = text
        applied = []
        stats = {}

        for strategy in self.strategies:
            before = len(compressed)
            compressed = strategy(compressed)
            after = len(compressed)
            if before != after:
                name = strategy.__name__.replace("_", " ").strip()
                applied.append(name)
                stats[name] = before - after

        compressed_tokens = self.estimate_tokens(compressed, lang)

        # Safety: never return empty or whitespace-only
        if not compressed.strip():
            stripped_original = text.strip()
            compressed = stripped_original if stripped_original else "(empty)"
            compressed_tokens = original_tokens

        return CompressionResult(
            original=text,
            compressed=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            strategies_applied=applied,
            stats=stats,
        )

    # === Strategy implementations ===

    def _normalize_whitespace(self, text: str) -> str:
        """Collapse multiple spaces/newlines, trim lines."""
        # Collapse 3+ blank lines to 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Collapse multiple spaces (but not in code blocks)
        lines = []
        in_code = False
        for line in text.split("\n"):
            if line.strip().startswith("```"):
                in_code = not in_code
                lines.append(line)
            elif in_code:
                lines.append(line)
            else:
                lines.append(re.sub(r" {2,}", " ", line).rstrip())
        return "\n".join(lines)

    def _remove_comments(self, text: str) -> str:
        """Remove code comments and docstrings."""
        # Python: # comments
        text = re.sub(r"^(\s*)#.*$", r"\1", text, flags=re.MULTILINE)
        # Python: triple-quoted docstrings (simple case)
        text = re.sub(r'""".*?"""', "", text, flags=re.DOTALL)
        text = re.sub(r"'''.*?'''", "", text, flags=re.DOTALL)
        # JS/TS: // comments
        text = re.sub(r"^(\s*)//.*$", r"\1", text, flags=re.MULTILINE)
        # HTML: <!-- comments -->
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        return text

    def _remove_redundancy(self, text: str) -> str:
        """Remove near-duplicate sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) < 2:
            return text

        keep = [sentences[0]]
        for i in range(1, len(sentences)):
            # Check similarity with most recent kept sentence
            sim = SequenceMatcher(None, keep[-1].lower(), sentences[i].lower()).ratio()
            if sim < 0.85:  # Keep if less than 85% similar
                keep.append(sentences[i])

        return " ".join(keep)

    def _compress_repetition(self, text: str) -> str:
        """Compress repeated patterns in structured text."""
        # Compress repeated JSON-like structures
        # e.g., '{"key": "val"}' repeated many times
        lines = text.split("\n")
        if len(lines) < 3:
            return text

        compressed = []
        repeat_count = 1
        for i in range(1, len(lines)):
            if lines[i] == lines[i - 1]:
                repeat_count += 1
            else:
                if repeat_count > 2:
                    compressed.append(lines[i - 1])
                    compressed.append(f"  [repeated {repeat_count - 1} more times]")
                else:
                    for _ in range(repeat_count):
                        compressed.append(lines[i - 1])
                repeat_count = 1

        # Handle last line
        if repeat_count > 2:
            compressed.append(lines[-1])
            compressed.append(f"  [repeated {repeat_count - 1} more times]")
        else:
            for _ in range(repeat_count):
                compressed.append(lines[-1])

        return "\n".join(compressed)

    def _abbreviate_common(self, text: str) -> str:
        """Replace common verbose phrases with abbreviations."""
        replacements = {
            r"\bplease note that\b": "Note:",
            r"\bit is important to note that\b": "Note:",
            r"\bin order to\b": "to",
            r"\bdue to the fact that\b": "because",
            r"\bat the present time\b": "now",
            r"\bin the event that\b": "if",
            r"\ba large number of\b": "many",
            r"\bhas the ability to\b": "can",
            r"\bwith regard to\b": "about",
            r"\bit should be noted that\b": "",
            r"\bI would like to\b": "I want to",
            r"\bthank you very much\b": "thanks",
            r"\bI hope this helps\b": "",
            r"\bplease let me know if you have any questions\b": "",
            r"\bfeel free to reach out\b": "",
            r"\bdo not hesitate to\b": "",
        }
        result = text
        for pattern, replacement in replacements.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        return result

    def _summarize_bullets(self, text: str) -> str:
        """Aggressively summarize bullet points."""
        lines = text.split("\n")
        result = []
        bullets = []
        in_bullets = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("- ", "* ", "• ", "1. ", "2. ")):
                if not in_bullets:
                    in_bullets = True
                bullets.append(re.sub(r"^[-*•\d.]+\s*", "", stripped))
            else:
                if in_bullets and len(bullets) > 3:
                    # Keep first and last, summarize middle
                    result.append(f"- {bullets[0]}")
                    result.append(f"  [... {len(bullets)-2} more items ...]")
                    result.append(f"- {bullets[-1]}")
                    bullets = []
                elif bullets:
                    for b in bullets:
                        result.append(f"- {b}")
                    bullets = []
                in_bullets = False
                result.append(line)

        # Handle trailing bullets
        if bullets:
            for b in bullets:
                result.append(f"- {b}")

        return "\n".join(result)

    def _remove_filler(self, text: str) -> str:
        """Remove common LLM filler phrases (aggressive mode only)."""
        fillers = [
            r"\bplease note that\b",
            r"\bit is important to note that\b",
            r"\bi hope this helps\b",
            r"\bplease let me know if you have any questions\b",
            r"\bfeel free to reach out\b",
            r"\bdo not hesitate to\b",
        ]
        result = text
        for f in fillers:
            result = re.sub(f, "", result, flags=re.IGNORECASE)

        # Clean up resulting artifacts
        result = re.sub(r" {2,}", " ", result)
        result = re.sub(r"\.\s*\.", ".", result)
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result.strip()


# === CLI ===

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Token Whisperer — compress prompts to save LLM tokens"
    )
    parser.add_argument("file", nargs="?", help="File to compress (or stdin)")
    parser.add_argument(
        "--aggressive", "-a", action="store_true", help="Aggressive compression"
    )
    parser.add_argument(
        "--lang",
        default="mixed",
        choices=["english", "code_python", "code_javascript", "mixed"],
        help="Language hint for token estimation",
    )
    parser.add_argument(
        "--output", "-o", help="Write compressed output to file"
    )
    parser.add_argument(
        "--stats-only", "-s", action="store_true", help="Only show stats, not compressed text"
    )

    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    tw = TokenWhisperer(aggressive=args.aggressive)
    result = tw.compress(text, lang=args.lang)

    print(result.report())
    print()

    if not args.stats_only:
        if args.output:
            with open(args.output, "w") as f:
                f.write(result.compressed)
            print(f"Compressed output written to {args.output}")
        else:
            print("─" * 60)
            print(result.compressed)
            print("─" * 60)


if __name__ == "__main__":
    main()
