#!/usr/bin/env python3
"""
Workflow visualization tool for Durable Mini.

Generates:
1. ASCII art workflow DAG
2. Step execution timeline
3. Colorized status dashboard

Usage:
    python visualize.py <instance_id>
    python visualize.py <instance_id> --format html > workflow.html
"""

import sys
import os
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from durable_mini import DurableEngine
from durable_mini.backend import DurableBackend

# ANSI color codes
C = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "green": "\033[32m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "cyan": "\033[36m",
    "magenta": "\033[35m",
    "white": "\033[37m",
}

STATUS_ICONS = {
    "pending": "⏳",
    "running": "🔄",
    "completed": "✅",
    "failed": "❌",
}

STATUS_COLORS = {
    "pending": C["dim"],
    "running": C["yellow"],
    "completed": C["green"],
    "failed": C["red"],
}


class WorkflowVisualizer:
    """Visualize workflow state from the database."""

    def __init__(self, db_path: str):
        self.backend = DurableBackend(db_path)

    def dag(self, instance_id: str) -> str:
        """Generate ASCII art DAG of the workflow."""
        instance = self.backend.get_instance(instance_id)
        if not instance:
            return f"Instance '{instance_id}' not found"

        workflow = self.backend.get_workflow(instance["workflow_name"])
        if not workflow:
            return f"Workflow '{instance['workflow_name']}' not found"

        steps = self.backend.get_all_steps(instance_id)
        step_map = {s.step_name: s for s in steps}

        lines = []
        lines.append(f"{C['bold']}╔══════════════════════════════════════╗{C['reset']}")
        lines.append(
            f"{C['bold']}║  Workflow: {instance['workflow_name']:<25s} ║{C['reset']}"
        )
        lines.append(
            f"{C['bold']}║  Instance: {instance_id:<25s} ║{C['reset']}"
        )
        lines.append(
            f"{C['bold']}║  Status:   {instance['status']:<25s} ║{C['reset']}"
        )
        lines.append(f"{C['bold']}╚══════════════════════════════════════╝{C['reset']}")
        lines.append("")

        # Build dependency tree
        step_defs = workflow["steps"]
        order = workflow.get("order", list(step_defs.keys()))

        # Find root steps (no dependencies)
        roots = [
            name for name in order if not step_defs.get(name, {}).get("depends_on", [])
        ]
        non_roots = [name for name in order if name not in roots]

        # Render
        for i, name in enumerate(order):
            sdef = step_defs.get(name, {})
            deps = sdef.get("depends_on", [])
            is_fanout = sdef.get("fan_out", False)
            step_rec = step_map.get(name)

            # Status
            if step_rec:
                icon = STATUS_ICONS.get(step_rec.status, "❓")
                color = STATUS_COLORS.get(step_rec.status, "")
            else:
                icon = "⬚"
                color = C["dim"]

            prefix = ""
            if deps:
                prefix = "  │" * (len(deps) - 1) + "  ├─"

            fan_tag = " ⟐" if is_fanout else ""
            step_info = f"{color}{icon} {name}{fan_tag}{C['reset']}"
            if step_rec and step_rec.status == "failed":
                step_info += f" {C['red']}[attempt {step_rec.attempt}]{C['reset']}"
            elif step_rec and step_rec.attempt > 1:
                step_info += f" {C['yellow']}[attempt {step_rec.attempt}]{C['reset']}"

            lines.append(f"  {prefix}{step_info}")

            # Show fan-out sub-steps
            if is_fanout:
                fan_steps = [
                    s
                    for s in steps
                    if s.step_name.startswith(f"{name}[") and s.step_name != name
                ]
                for fs in fan_steps[:5]:  # Show first 5
                    ficon = STATUS_ICONS.get(fs.status, "⬚")
                    fcolor = STATUS_COLORS.get(fs.status, "")
                    lines.append(f"  │    {fcolor}└─ {ficon} {fs.step_name}{C['reset']}")
                if len(fan_steps) > 5:
                    lines.append(f"  │    └─ ... ({len(fan_steps)} total)")

        # Stats footer
        completed = sum(1 for s in steps if s.status == "completed")
        failed = sum(1 for s in steps if s.status == "failed")
        lines.append("")
        lines.append(
            f"  {C['dim']}Steps: {completed} completed, {failed} failed, {len(steps)} total{C['reset']}"
        )

        return "\n".join(lines)

    def timeline(self, instance_id: str) -> str:
        """Generate execution timeline."""
        steps = self.backend.get_all_steps(instance_id)
        if not steps:
            return "No steps found"

        lines = []
        lines.append(f"\n{C['bold']}Execution Timeline{C['reset']}")
        lines.append("─" * 50)

        # Calculate time range
        started_times = [s.started_at for s in steps if s.started_at]
        completed_times = [s.completed_at for s in steps if s.completed_at]
        if not started_times:
            return "No execution data"

        t_min = min(started_times)
        t_max = max(completed_times + started_times)
        total_duration = t_max - t_min

        bar_width = 40

        for s in sorted(steps, key=lambda s: s.started_at or 0):
            if not s.started_at:
                continue

            offset = s.started_at - t_min
            duration = (
                (s.completed_at - s.started_at) if s.completed_at else total_duration - offset
            )

            bar_start = int(offset / total_duration * bar_width) if total_duration > 0 else 0
            bar_len = max(1, int(duration / total_duration * bar_width)) if total_duration > 0 else 1

            bar = " " * bar_start + "█" * bar_len
            icon = STATUS_ICONS.get(s.status, "⬚")
            color = STATUS_COLORS.get(s.status, "")

            name = s.step_name[:20].ljust(20)
            duration_str = (
                f"{(s.completed_at - s.started_at)*1000:.0f}ms"
                if s.completed_at and s.started_at
                else "—"
            )

            lines.append(f"  {color}{icon} {name}{C['reset']} {C['dim']}{bar}{C['reset']} {duration_str}")

        lines.append("─" * 50)
        lines.append(f"  Total: {total_duration*1000:.0f}ms")
        return "\n".join(lines)

    def dashboard(self, instance_id: str) -> str:
        """Full color dashboard."""
        parts = [self.dag(instance_id), self.timeline(instance_id)]
        return "\n".join(parts)

    def html(self, instance_id: str) -> str:
        """Generate an HTML visualization."""
        instance = self.backend.get_instance(instance_id)
        if not instance:
            return f"<p>Instance '{instance_id}' not found</p>"

        workflow = self.backend.get_workflow(instance["workflow_name"])
        if not workflow:
            return f"<p>Workflow not found</p>"

        steps = self.backend.get_all_steps(instance_id)
        order = workflow.get("order", list(workflow["steps"].keys()))

        status_class = {"completed": "ok", "failed": "fail", "running": "run", "pending": "pend"}
        status_emoji = {"completed": "✅", "failed": "❌", "running": "🔄", "pending": "⏳"}

        step_html = ""
        for name in order:
            sdef = workflow["steps"].get(name, {})
            rec = next((s for s in steps if s.step_name == name), None)
            st = rec.status if rec else "pending"
            cls = status_class.get(st, "pend")
            emoji = status_emoji.get(st, "⬚")
            is_fan = sdef.get("fan_out", False)
            attempt = f" (attempt {rec.attempt})" if rec and rec.attempt > 1 else ""

            # Sub-steps for fan-out
            fan_html = ""
            if is_fan:
                fan_steps = [s for s in steps if s.step_name.startswith(f"{name}[")]
                for fs in fan_steps:
                    fst = fs.status
                    fcls = status_class.get(fst, "pend")
                    femoji = status_emoji.get(fst, "⬚")
                    fan_html += f'<div class="substep {fcls}">{femoji} {fs.step_name}</div>'

            step_html += f"""
            <div class="step {cls}">
              <div class="step-header">{emoji} <strong>{name}</strong>{' ⟐' if is_fan else ''}{attempt}</div>
              {fan_html}
            </div>"""

        stats = self.backend.get_stats()
        completed = sum(1 for s in steps if s.status == "completed")
        failed_count = sum(1 for s in steps if s.status == "failed")

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Durable Mini — {instance['workflow_name']}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    background: #0d1117; color: #c9d1d9; padding: 2rem;
  }}
  .header {{
    border: 1px solid #30363d; border-radius: 8px; padding: 1.5rem;
    margin-bottom: 1.5rem; background: #161b22;
  }}
  .header h1 {{ font-size: 1.4rem; color: #58a6ff; }}
  .header .meta {{ color: #8b949e; margin-top: 0.5rem; font-size: 0.85rem; }}
  .stats {{ display: flex; gap: 1rem; margin: 1.5rem 0; }}
  .stat {{
    flex: 1; text-align: center; padding: 1rem;
    border: 1px solid #30363d; border-radius: 6px; background: #161b22;
  }}
  .stat .num {{ font-size: 2rem; font-weight: bold; }}
  .stat .label {{ color: #8b949e; font-size: 0.8rem; margin-top: 0.25rem; }}
  .ok .num {{ color: #3fb950; }}
  .fail .num {{ color: #f85149; }}
  .step {{
    border: 1px solid #30363d; border-radius: 6px; padding: 0.75rem 1rem;
    margin-bottom: 0.5rem; background: #161b22;
  }}
  .step.ok {{ border-left: 3px solid #3fb950; }}
  .step.fail {{ border-left: 3px solid #f85149; }}
  .step.run {{ border-left: 3px solid #d2991d; }}
  .step.pend {{ border-left: 3px solid #30363d; opacity: 0.6; }}
  .substep {{ margin: 0.25rem 0 0.25rem 2rem; font-size: 0.85rem; color: #8b949e; }}
  .substep.ok {{ color: #3fb950; }}
  .substep.fail {{ color: #f85149; }}
  .footer {{ margin-top: 2rem; color: #484f58; font-size: 0.8rem; text-align: center; }}
</style>
</head>
<body>
<div class="header">
  <h1>⚙️ {instance['workflow_name']}</h1>
  <div class="meta">
    Instance: {instance_id}<br>
    Status: <span style="color: {'#3fb950' if instance['status']=='completed' else '#f85149'}">{instance['status'].upper()}</span>
  </div>
</div>
<div class="stats">
  <div class="stat ok"><div class="num">{completed}</div><div class="label">Completed</div></div>
  <div class="stat fail"><div class="num">{failed_count}</div><div class="label">Failed</div></div>
  <div class="stat"><div class="num">{len(steps)}</div><div class="label">Total Steps</div></div>
</div>
<h2 style="color:#8b949e;margin-bottom:0.75rem;">Steps</h2>
{step_html}
<div class="footer">
  Durable Mini · Clockwork Resurrections / 钟表复活 · {time.strftime('%Y-%m-%d %H:%M')}
</div>
</body>
</html>"""

    def close(self):
        self.backend.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize Durable Mini workflows")
    parser.add_argument("db_path", help="Path to the SQLite database")
    parser.add_argument("instance_id", help="Workflow instance ID")
    parser.add_argument(
        "--format",
        choices=["dag", "timeline", "dashboard", "html"],
        default="dashboard",
        help="Output format",
    )
    parser.add_argument(
        "-o", "--output", help="Output file (for HTML format)"
    )
    args = parser.parse_args()

    viz = WorkflowVisualizer(args.db_path)

    if args.format == "dag":
        output = viz.dag(args.instance_id)
    elif args.format == "timeline":
        output = viz.timeline(args.instance_id)
    elif args.format == "dashboard":
        output = viz.dashboard(args.instance_id)
    elif args.format == "html":
        output = viz.html(args.instance_id)

    if args.output and args.format == "html":
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Saved to {args.output}")
    else:
        print(output)

    viz.close()


if __name__ == "__main__":
    main()
