#!/usr/bin/env python3
"""Cosmic Haiku — a generative astronomy-themed haiku engine.

Generates haiku (5-7-5) using template-based composition with
real astronomical vocabulary. Each run produces a unique poem.

Usage:
    conda run -n hermesauto python cosmic_haiku.py
"""

import random
from pathlib import Path

OUT = Path("/home/yanyj/VibeCoding/autonomy/2026-06-01/cosmic_haiku")
OUT.mkdir(parents=True, exist_ok=True)

# Syllable-counted astronomy vocabulary
N1 = {  # 1-syllable
    "star", "dust", "light", "void", "dark", "red", "blue", "core",
    "space", "wave", "spin", "flare", "pulse", "beam", "sky", "ghost",
    "brane", "glow", "弦", "辉", "暗"
}

N2 = {  # 2-syllable
    "photon", "cosmos", "nebula", "quasar", "pulsar", "redshift",
    "dark flow", "event", "horizon", "supernova", "cluster", "galaxy",
    "引力", "粒子", "星团", "光子"
}

N3 = {  # 3-syllable
    "universe", "singularity", "parallax", "infinity", "radiation",
    "stellar wind", "dark matter", "cosmic web", "light echo",
    "neutrino", "magnetar", "equinox",
    "黑洞视界", "宇宙网", "超新星"
}

N4 = {  # 4-syllable
    "gravitation", "constellation", "interstellar", "annihilation",
    "cosmic microwave", "dark energy field", "spiral arm",
    "accretion disk", "event horizon",
    "暗物质晕", "星系团"
}

N5 = {  # 5-syllable
    "cosmic inflation", "stellar nucleo-synthesis",
    "gravitational wave", "intergalactic medium",
    "large-scale structure", "primordial black hole",
}

# Phrase templates with slot for astronomical terms
LINE1_TMPL = [
    "{N2} {N3}",
    "{N1} {N4}",
    "{N5}",
    "{N2} in the {N3}",
    "cold {N2} {N2}",
]

LINE2_TMPL = [
    "{N3} {N4}",
    "{N2} through {N5}",
    "{N1} {N1} {N5}",
    "where {N2} {N2} drift",
    "echo of {N4}",
]

LINE3_TMPL = [
    "{N3} {N2}",
    "{N2} {N1} {N2}",
    "{N5}",
    "into the {N3}",
    "only {N2} remain",
]


def pick_word(syllables):
    pool = {1: N1, 2: N2, 3: N3, 4: N4, 5: N5}[syllables]
    return random.choice(list(pool))


def fill_template(tmpl):
    result = tmpl
    for syl in [5, 4, 3, 2, 1]:
        result = result.replace(f"{{N{syl}}}", pick_word(syl))
    return result


def generate_haiku():
    l1 = fill_template(random.choice(LINE1_TMPL))
    l2 = fill_template(random.choice(LINE2_TMPL))
    l3 = fill_template(random.choice(LINE3_TMPL))
    return l1, l2, l3


def render_haiku(poems, filename):
    """Render haikus as a beautiful text file."""
    lines = ["╔══════════════════════════╗",
             "║   ✦ COSMIC HAIKU ✦    ║",
             "║   generative poetry    ║",
             "╚══════════════════════════╝",
             ""]

    for i, (l1, l2, l3) in enumerate(poems, 1):
        lines.append(f"  #{i}")
        lines.append(f"    {l1}")
        lines.append(f"    {l2}")
        lines.append(f"    {l3}")
        lines.append("")

    text = "\n".join(lines)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def main():
    print("Cosmic Haiku — astronomy-themed generative poetry")
    print("=" * 50)

    random.seed()

    poems = [generate_haiku() for _ in range(12)]

    text = render_haiku(poems, OUT / "haiku.txt")

    # Also save as simple text for sharing
    simple = []
    for i, (l1, l2, l3) in enumerate(poems, 1):
        simple.append(f"{l1}")
        simple.append(f"{l2}")
        simple.append(f"{l3}")
        simple.append("")
    with open(OUT / "haiku_plain.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(simple))

    print(text)
    print(f"\nSaved to {OUT / 'haiku.txt'}")


if __name__ == "__main__":
    main()
