#!/usr/bin/env python3
"""Publish a day's exploration output to the Free Hours GitHub Pages archive.

Usage:
    python scripts/publish_day.py <date> <exploration_dir> [--title-en TITLE --title-zh TITLE --variable-en VAR --variable-zh VAR --summary SUMMARY] [--dry-run]

Example:
    python scripts/publish_day.py 2026-06-02 /home/yanyj/VibeCoding/autonomy/2026-06-02/my_project \
        --title-en "My Title" --title-zh "我的标题" \
        --variable-en "Variable" --variable-zh "变量" \
        --summary "A brief one-line summary."

The script will:
    1. Copy report.md and project files to archive/YYYY/MM/YYYY-MM-DD/
    2. Update metadata/days.json
    3. Rebuild docs/index.html
    4. Git commit and push
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(cmd, cwd=cwd or ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {' '.join(cmd)}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    print(result.stdout.strip())


def main():
    parser = argparse.ArgumentParser(description="Publish a day's exploration to Free Hours archive")
    parser.add_argument("date", help="Date in YYYY-MM-DD format")
    parser.add_argument("exploration_dir", help="Path to the day's exploration directory")
    parser.add_argument("--title-en", default="", help="English title")
    parser.add_argument("--title-zh", default="", help="Chinese title")
    parser.add_argument("--variable-en", default="", help="English variable/theme name")
    parser.add_argument("--variable-zh", default="", help="Chinese variable/theme name")
    parser.add_argument("--summary", default="", help="One-line summary")
    parser.add_argument("--dry-run", action="store_true", help="Skip git push")
    args = parser.parse_args()

    date = args.date
    y, m, d = date.split("-")
    src_dir = Path(args.exploration_dir)
    dest_dir = ROOT / "docs" / "archive" / y / m / date
    assets_dir = dest_dir / "assets"

    print(f"Publishing {date} from {src_dir}")
    print(f"  -> {dest_dir}")

    # 1. Copy files
    if not src_dir.exists():
        print(f"ERROR: source directory {src_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    dest_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Copy report.md as index.md, with bilingual header prepended
    report_path = src_dir / "report.md"
    if not report_path.exists():
        reports = list(src_dir.rglob("report.md"))
        if reports:
            report_path = reports[0]

    if report_path.exists():
        original = report_path.read_text(encoding="utf-8")
        # Build bilingual header
        header = f"""---
layout: default
title: "{args.title_en} · {args.title_zh}"
---

# {args.title_zh} · {args.title_en}

**{date}** · 自由时光 / Free Hours  

> *{args.summary or ''}*

[← 返回档案 / Back to Archive](../../../../)

---

"""
        (dest_dir / "index.md").write_text(header + original, encoding="utf-8")
        print("  wrote index.md with bilingual header")
    else:
        print("  WARNING: no report.md found", file=sys.stderr)

    # Copy all non-report files into assets/
    for f in src_dir.rglob("*"):
        if f.is_file() and f.name != "report.md" and ".git" not in f.parts:
            rel = f.relative_to(src_dir)
            target = assets_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            print(f"  copied {rel} -> assets/{rel}")

    # 2. Update metadata/days.json
    days_path = ROOT / "metadata" / "days.json"
    days = json.loads(days_path.read_text(encoding="utf-8"))

    # Remove existing entry for this date (if any)
    days = [d for d in days if d["date"] != date]

    # Auto-detect a preview image from assets
    preview_path = None
    for f in sorted(assets_dir.rglob("*")):
        if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"):
            preview_path = str(f.relative_to(dest_dir.parent.parent.parent))
            # Convert docs/archive/... to archive/...
            if preview_path.startswith("docs/"):
                preview_path = preview_path[5:]
            break
    if not preview_path:
        preview_path = f"archive/{y}/{m}/{date}/assets/preview.png"
    print(f"  preview image: {preview_path}")

    entry = {
        "date": date,
        "title_en": args.title_en,
        "title_zh": args.title_zh,
        "variable_en": args.variable_en,
        "variable_zh": args.variable_zh,
        "summary": args.summary,
        "preview": preview_path,
        "archive_url": f"archive/{y}/{m}/{date}/",
    }
    days.append(entry)
    days.sort(key=lambda d: d["date"])
    days_path.write_text(json.dumps(days, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"  updated metadata/days.json ({len(days)} entries)")

    # 3. Rebuild homepage
    build_script = ROOT / "scripts" / "build_index.py"
    if build_script.exists():
        subprocess.run([sys.executable, str(build_script)], cwd=ROOT, check=True)

    # 4. Git commit and push
    print("\nCommitting and pushing...")
    run(["git", "add", "-A"])
    run(["git", "commit", "-m", f"archive: {date} {args.title_en or args.title_zh}"])
    if not args.dry_run:
        run(["git", "push"])
        print(f"\nPublished! https://yanzoro926.github.io/free-hours/docs/archive/{y}/{m}/{date}/")
    else:
        print("\nDry run complete. Use without --dry-run to push.")


if __name__ == "__main__":
    main()
