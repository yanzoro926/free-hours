# Token Whisperer / 令牌低语者

> *"Speak less, say more."*

A prompt compression tool that reduces LLM input tokens while preserving semantic meaning. Built during exploration of the HN front-page story about Lowfat (91.8% token savings).

## Quick Start

```bash
# Compress a file
python token_whisperer.py my_prompt.txt

# Compress with aggressive mode
python token_whisperer.py my_prompt.txt --aggressive

# Only show stats
python token_whisperer.py code.py --lang code_python --stats-only

# Pipe from stdin
cat long_prompt.txt | python token_whisperer.py
```

## Compression Strategies

| Strategy | Safe Mode | Aggressive Mode | Best For |
|----------|:---------:|:---------------:|----------|
| Whitespace normalization | ✅ | ✅ | All text |
| Comment/docstring removal | ✅ | ✅ | Code |
| Redundancy detection | ✅ | ✅ | Long-form text |
| Repetition compression | ✅ | ✅ | Structured data |
| Common abbreviation | ✅ | ✅ | English prose |
| Bullet summarization | ❌ | ✅ | Lists/bullet points |
| Filler removal | ❌ | ✅ | Verbose English |

## Results

```
Python code:   40.3% saved (6,809 → 4,064 tokens)
English text:  17.1% saved (333 → 276 tokens)
```

## Architecture

```
Input Text → [Whitespace] → [Comments] → [Redundancy] → [Repetition] → [Abbrev] → [Bullets*] → [Fillers*] → Output
                                                                                          * aggressive only
```

## Limitations

- Token counts are heuristic estimates (chars × density), not actual BPE counts
- Code comment removal uses regex — complex languages may need AST-aware removal
- Redundancy detection uses simple string similarity, not semantic similarity
- Aggressive mode may alter meaning — review before sending to LLMs

## License

MIT
