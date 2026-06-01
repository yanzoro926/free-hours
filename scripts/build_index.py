#!/usr/bin/env python3
"""Regenerate docs/index.html from metadata/days.json.

Reads metadata/days.json and embeds the data into the index.html page.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DAYS_PATH = ROOT / "metadata" / "days.json"
INDEX_PATH = ROOT / "docs" / "index.html"


def main():
    days = json.loads(DAYS_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(days)} day entries.")

    html = INDEX_PATH.read_text(encoding="utf-8")

    # Embed days data into the <script id="days-data"> tag
    import re
    days_json = json.dumps(days, ensure_ascii=False)
    html = re.sub(
        r'(<script id="days-data" type="application/json">).*?(</script>)',
        rf'\1{days_json}\2',
        html,
        flags=re.DOTALL,
    )

    INDEX_PATH.write_text(html, encoding="utf-8")
    print(f"Updated {INDEX_PATH} with {len(days)} embedded entries.")


if __name__ == "__main__":
    main()
