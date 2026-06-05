"""Quick tests for Token Whisperer."""
import unittest
from token_whisperer import TokenWhisperer, CompressionResult


class TestTokenWhisperer(unittest.TestCase):
    def setUp(self):
        self.tw = TokenWhisperer()

    def test_whitespace_normalization(self):
        result = self.tw.compress("hello    world\n\n\n\nfoo")
        self.assertLess(len(result.compressed), len(result.original))

    def test_abbreviation(self):
        result = self.tw.compress("due to the fact that it is raining")
        self.assertIn("because", result.compressed.lower())

    def test_redundancy_removal(self):
        text = "Hello world. Hello world. Something new."
        result = self.tw.compress(text)
        # Should have fewer sentences
        self.assertLess(result.compressed_tokens, result.original_tokens)

    def test_comment_removal(self):
        text = "# This is a comment\nx = 1  # inline comment"
        result = self.tw.compress(text)
        self.assertNotIn("This is a comment", result.compressed)

    def test_code_block_preservation(self):
        text = "```python\n# comment\nx = 1\n```\nNormal text."
        result = self.tw.compress(text)
        # Code blocks preserved (whitespace normalization doesn't touch them)
        self.assertIn("x = 1", result.compressed)

    def test_never_empty(self):
        result = self.tw.compress("   ")
        self.assertTrue(len(result.compressed.strip()) > 0)

    def test_token_estimation(self):
        tokens = self.tw.estimate_tokens("hello world", "english")
        self.assertGreater(tokens, 0)

    def test_aggressive_mode(self):
        tw_agg = TokenWhisperer(aggressive=True)
        text = "Please note that I would like to request your help. Thank you!"
        result = tw_agg.compress(text)
        self.assertLess(result.compressed_tokens, result.original_tokens)

    def test_stats(self):
        result = self.tw.compress("due to the fact that x")
        self.assertIn("abbreviate common", result.strategies_applied)


if __name__ == "__main__":
    unittest.main()
