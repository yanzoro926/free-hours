#!/usr/bin/env python3
"""CodeGlyph — generate a beautiful codebase structure visualization.

Creates a text-based visual map of a Python project's structure,
showing modules, classes, functions, and their relationships.

Usage:
    conda run -n hermesauto python codeglyph.py [path]
"""

import ast
from pathlib import Path
from collections import defaultdict

OUT = Path("/home/yanyj/VibeCoding/autonomy/2026-06-01/codeglyph")
OUT.mkdir(parents=True, exist_ok=True)


def analyze_file(filepath):
    """Extract structure from a Python file."""
    try:
        with open(filepath) as f:
            tree = ast.parse(f.read())
    except (SyntaxError, UnicodeDecodeError):
        return {"classes": [], "functions": [], "imports": []}

    result = {"classes": [], "functions": [], "imports": []}

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = [n.name for n in ast.walk(node) if isinstance(n, ast.FunctionDef)]
            result["classes"].append({"name": node.name, "methods": methods, "line": node.lineno})
        elif isinstance(node, ast.FunctionDef) and not _is_method(node, tree):
            result["functions"].append({"name": node.name, "line": node.lineno})
        elif isinstance(node, ast.Import):
            for alias in node.names:
                result["imports"].append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                result["imports"].append(node.module)

    return result


def _is_method(func_node, tree):
    """Check if a function is a method inside a class."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for child in ast.walk(node):
                if child is func_node:
                    return True
    return False


def build_tree(root_path, max_depth=4):
    """Build a directory tree with code structure."""
    root = Path(root_path)
    tree = {}

    py_files = sorted(root.rglob("*.py"))
    py_files = [f for f in py_files if "__pycache__" not in str(f) and ".egg-info" not in str(f)]
    py_files = py_files[:50]  # limit

    for f in py_files:
        rel = f.relative_to(root)
        depth = len(rel.parts)
        if depth > max_depth:
            continue

        structure = analyze_file(f)
        tree[str(rel)] = {
            "classes": structure["classes"],
            "functions": structure["functions"],
            "size": f.stat().st_size,
            "lines": len(f.read_text().splitlines()) if f.exists() else 0,
        }

    return tree


def render_tree(tree, root_name):
    """Render the tree as formatted text."""
    lines = [
        "╔══════════════════════════════════════════╗",
        "║         ✦ CODEGLYPH ✦                 ║",
        f"║   {root_name:<34} ║",
        "╚══════════════════════════════════════════╝",
        "",
    ]

    # Group by directory
    dirs = defaultdict(list)
    for path, info in sorted(tree.items()):
        parent = str(Path(path).parent)
        dirs[parent].append((Path(path).name, info))

    for d in sorted(dirs):
        lines.append(f"📁 {d if d != '.' else root_name}/")
        for name, info in sorted(dirs[d]):
            size_kb = info["size"] / 1024
            lines.append(f"  📄 {name}  ({info['lines']} lines, {size_kb:.1f} KB)")

            for cls in info["classes"]:
                methods_str = ", ".join(cls["methods"][:5])
                if len(cls["methods"]) > 5:
                    methods_str += f", ... (+{len(cls['methods'])-5})"
                lines.append(f"      ◈ class {cls['name']} → {methods_str}")

            for fn in info["functions"]:
                lines.append(f"      ● def {fn['name']}()")

        lines.append("")

    # Summary
    total_files = len(tree)
    total_lines = sum(i["lines"] for i in tree.values())
    total_classes = sum(len(i["classes"]) for i in tree.values())
    total_funcs = sum(len(i["functions"]) for i in tree.values())

    lines.append("─" * 44)
    lines.append(f"  {total_files} files · {total_lines} lines · {total_classes} classes · {total_funcs} functions")
    lines.append("")

    return "\n".join(lines)


def main():
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "."

    print(f"CodeGlyph — analyzing {target}...")
    tree = build_tree(target)
    output = render_tree(tree, Path(target).name)

    out_path = OUT / f"codeglyph_{Path(target).name}.txt"
    out_path.write_text(output)
    print(output)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
