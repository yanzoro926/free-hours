#!/usr/bin/env python3
"""
Lichtenberg Generator / 雷纹生成器

A Python reimagining of endoh2's IOCCC 2025 winning entry
"Lichtenberg curves." Generates unique electrical discharge
patterns as ANSI-colored terminal art.

Algorithm:
  1. Build triangular (hexagonal) lattice
  2. Grow random spanning tree from a seed point
  3. Accumulate charge from leaves to root
  4. Render edges with brightness proportional to current

Each run produces a different pattern, seeded by PID or argument.

Inspired by: https://www.ioccc.org/2025/endoh2/
"""

import os
import sys
import random
import math


class LichtenbergGenerator:
    """Generates Lichtenberg figure ASCII art."""

    # ANSI color palette: dim yellow → bright white
    # Using 256-color mode for broader compatibility
    COLORS = [232, 233, 234, 235, 236, 237, 238, 239, 240,
              241, 242, 243, 244, 245, 246, 247, 248, 249,
              250, 251, 252, 253, 254, 255,  # grays
              220, 221, 222, 227, 228, 229, 230,  # yellow→white
              226, 190, 154, 118, 82, 46, 11]  # bright yellow→orange→red (hot spots)

    def __init__(self, seed=None, width=None, height=None):
        if seed is None:
            seed = os.getpid()
        random.seed(seed)
        self.seed = seed

        # Terminal dimensions
        try:
            cols, rows = os.get_terminal_size()
        except (ValueError, OSError):
            cols, rows = 80, 24

        self.W = width or cols
        self.H = height or rows

        # Triangular lattice parameters
        # Each "logical" cell is 2 characters wide × 1 row tall
        self.grid_w = self.W // 2
        self.grid_h = self.H

        # Node count in the triangular lattice
        self.nodes = self.grid_w * self.grid_h

        # Edge storage: for each node, store 6 possible neighbors
        # Neighbor directions (in node index offsets):
        #   up-left, up-right, left, right, down-left, down-right
        self.parent = [-1] * self.nodes  # spanning tree parent
        self.charge = [0] * self.nodes   # accumulated charge

        # Seed point (discharge origin) — near center
        self.seed_x = self.grid_w // 2
        self.seed_y = self.grid_h // 2
        self.seed_idx = self.seed_y * self.grid_w + self.seed_x

    def _node_idx(self, x, y):
        """Convert grid coordinates to node index."""
        if 0 <= x < self.grid_w and 0 <= y < self.grid_h:
            return y * self.grid_w + x
        return -1

    def _neighbors(self, idx):
        """Get valid neighbor indices for a hexagonal lattice node."""
        x = idx % self.grid_w
        y = idx // self.grid_w

        # Six directions for hexagonal grid
        # Parity determines which diagonals are valid
        is_even_row = (y % 2 == 0)

        dirs = []
        # Up-left
        if is_even_row:
            dirs.append((x - 1, y - 1))  # up-left
            dirs.append((x, y - 1))      # up-right
        else:
            dirs.append((x, y - 1))      # up-left
            dirs.append((x + 1, y - 1))  # up-right

        # Left and right
        dirs.append((x - 1, y))          # left
        dirs.append((x + 1, y))          # right

        # Down-left and down-right
        if is_even_row:
            dirs.append((x - 1, y + 1))  # down-left
            dirs.append((x, y + 1))      # down-right
        else:
            dirs.append((x, y + 1))      # down-left
            dirs.append((x + 1, y + 1))  # down-right

        result = []
        for nx, ny in dirs:
            nidx = self._node_idx(nx, ny)
            if nidx >= 0:
                result.append(nidx)
        return result

    def _grow_tree(self):
        """Grow a random spanning tree from the seed point using randomized BFS/DFS."""
        # Randomized Prim-like algorithm
        frontier = [self.seed_idx]
        visited = [False] * self.nodes
        visited[self.seed_idx] = True
        self.parent[self.seed_idx] = self.seed_idx  # root

        while frontier:
            # Pick a random frontier node
            i = random.randrange(len(frontier))
            current = frontier[i]

            # Get unvisited neighbors
            neighbors = self._neighbors(current)
            unvisited = [n for n in neighbors if not visited[n]]

            if unvisited:
                # Pick random unvisited neighbor
                next_node = random.choice(unvisited)
                visited[next_node] = True
                self.parent[next_node] = current
                frontier.append(next_node)
            else:
                # No unvisited neighbors — remove from frontier
                frontier[i] = frontier[-1]
                frontier.pop()

        return visited

    def _accumulate_charge(self):
        """Accumulate charge from leaves to root of the spanning tree."""
        # First, compute "leaf charge": 1 for each node, then propagate upward
        # Count children for each node
        children_count = [0] * self.nodes
        for i in range(self.nodes):
            p = self.parent[i]
            if p >= 0 and p != i:
                children_count[p] += 1

        # BFS from leaves: collect all leaves (nodes with no children)
        leaves = []
        for i in range(self.nodes):
            if self.parent[i] >= 0 and children_count[i] == 0:
                leaves.append(i)

        # Initialize charge: each leaf gets base charge
        base_charge = 1
        for leaf in leaves:
            self.charge[leaf] = base_charge

        # Process in topological order (leaves to root)
        # Build reverse adjacency
        children = [[] for _ in range(self.nodes)]
        for i in range(self.nodes):
            p = self.parent[i]
            if p >= 0 and p != i:
                children[p].append(i)

        # Post-order traversal from root
        order = []
        stack = [(self.seed_idx, 0)]  # (node, state: 0=enter, 1=exit)
        while stack:
            node, state = stack.pop()
            if state == 0:
                stack.append((node, 1))
                for child in children[node]:
                    stack.append((child, 0))
            else:
                order.append(node)

        # Propagate charge upward
        for node in order:
            p = self.parent[node]
            if p >= 0 and p != node:
                # Charge propagation with attenuation
                self.charge[p] += self.charge[node] * 0.7

    def generate(self):
        """Generate the Lichtenberg figure."""
        self._grow_tree()
        self._accumulate_charge()

        # Find max charge for normalization
        max_charge = max(self.charge) if max(self.charge) > 0 else 1

        return self._render_ansi(max_charge)

    def _render_ansi(self, max_charge):
        """Render the Lichtenberg figure as ANSI-colored text."""
        lines = []

        # Background: very dark blue-black
        bg = "\x1b[48;5;17m"

        for y in range(self.grid_h):
            line_parts = []
            for x in range(self.grid_w):
                idx = y * self.grid_w + x
                charge = self.charge[idx]

                if charge == 0:
                    # Empty space
                    line_parts.append(f"{bg}  ")
                    continue

                # Normalize charge to [0, 1]
                intensity = min(charge / max_charge, 1.0)

                # Map intensity to color
                color_idx = int(intensity ** 0.5 * (len(self.COLORS) - 1))
                color = self.COLORS[min(color_idx, len(self.COLORS) - 1)]

                # Render edge based on parent direction
                px = self.parent[idx] % self.grid_w if self.parent[idx] >= 0 else -1
                py = self.parent[idx] // self.grid_w if self.parent[idx] >= 0 else -1

                # Determine which characters to use
                is_even = (y % 2 == 0)
                chars = "  "

                if self.parent[idx] == idx:
                    # Root node — origin of discharge
                    chars = "\x1b[38;5;226m◆\x1b[0m"
                    line_parts.append(f"{bg}{chars}")
                    continue

                # Simple rendering: just color the node
                line_parts.append(f"\x1b[48;5;{color}m  ")

            line_parts.append("\x1b[0m")
            lines.append("".join(line_parts))

        return "\n".join(lines)

    def generate_text(self):
        """Generate a plain-text version (no ANSI) for screenshots/files."""
        self._grow_tree()
        self._accumulate_charge()
        max_charge = max(self.charge) if max(self.charge) > 0 else 1

        chars = " ·:;+=*#@"
        lines = []
        for y in range(self.grid_h):
            line = ""
            for x in range(self.grid_w):
                idx = y * self.grid_w + x
                charge = self.charge[idx]
                if charge == 0:
                    line += "  "
                else:
                    intensity = min(charge / max_charge, 1.0)
                    ci = min(int(intensity * (len(chars) - 1)), len(chars) - 1)
                    c = chars[ci]
                    line += c * 2
            lines.append(line)
        return "\n".join(lines)


def main():
    """Entry point."""
    seed = None
    if len(sys.argv) > 1:
        try:
            seed = int(sys.argv[1])
        except ValueError:
            seed = hash(sys.argv[1]) % 2**31

    gen = LichtenbergGenerator(seed)

    if "--text" in sys.argv:
        # Plain text output
        result = gen.generate_text()
        print(result)
    else:
        # ANSI output
        print("\x1b[2J\x1b[H", end="")  # Clear screen
        result = gen.generate()
        print(result)
        print(f"\x1b[0m\n  ⚡ Lichtenberg Figure · seed={gen.seed} · {gen.grid_w}×{gen.grid_h}")
        print(f"  Inspired by Yusuke Endoh's IOCCC 2025 winning entry")


if __name__ == "__main__":
    main()
