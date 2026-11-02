"""Render every markdown table in README.md as a PNG and show it in the README.

    python scripts/render_readme_tables.py

Each table stays in the README as the editable source, collapsed under its image.
Edit the numbers there and rerun to refresh the images; already-wrapped tables are
not wrapped again.
"""
from __future__ import annotations

import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
FIGURES = "reports/figures"

MAX_WIDTH_PX = 880  # GitHub's README column
LOGICAL_DPI = 100
SAVE_DPI = 200
FONT_SIZE = 11
LINE_PX = 19
PAD_X_PX = 12
PAD_Y_PX = 9
INK = "#1f2328"
RULE = "#d1d9e0"
HEADER_RULE = "#59636e"
SURFACE = "#ffffff"

Style = dict[str, str]


def split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_cell(cell: str, row_label: bool = False) -> tuple[str, Style]:
    """Strip markdown marks; a wholly bold or code cell, or a row label starting with code, keeps that style."""
    style: Style = {}
    if row_label and cell.startswith("`"):
        style["fontfamily"] = "monospace"
    if re.fullmatch(r"\*\*[^*]+\*\*", cell):
        style["fontweight"] = "bold"
    if re.fullmatch(r"`[^`]+`", cell):
        style["fontfamily"] = "monospace"
    return cell.replace("**", "").replace("`", ""), style


def is_numeric_column(rows: list[list[str]], col: int) -> bool:
    return all(re.match(r"[~*]*\d|—$", row[col]) for row in rows)


def text_width(fig: plt.Figure, text: str, style: Style, bold: bool = False) -> float:
    kwargs = {**style, "fontweight": "bold"} if bold else style
    artist = Text(0, 0, text, figure=fig, fontsize=FONT_SIZE, parse_math=False, **kwargs)
    return artist.get_window_extent(fig.canvas.get_renderer()).width


def wrap_table(fig: plt.Figure, header: list[str], rows: list[list[str]]) -> tuple[list[list[list[str]]], list[float]]:
    """Wrap the widest column until the table fits. Returns wrapped lines per cell, and column widths."""
    table = [header] + rows
    parsed = [[parse_cell(cell, row_label=r > 0 and c == 0) for c, cell in enumerate(row)] for r, row in enumerate(table)]
    limits = [max(len(text) for text, _ in col) for col in zip(*parsed)]
    while True:
        wrapped = [
            [textwrap.wrap(text, limit, break_long_words=False, break_on_hyphens=False) or [""] for (text, _), limit in zip(row, limits)]
            for row in parsed
        ]
        widths = [
            max(
                text_width(fig, line, parsed[r][c][1], bold=r == 0)
                for r in range(len(table))
                for line in wrapped[r][c]
            ) + 2 * PAD_X_PX
            for c in range(len(header))
        ]
        if sum(widths) <= MAX_WIDTH_PX:
            return wrapped, widths
        order = sorted(range(len(widths)), key=lambda c: widths[c], reverse=True)
        narrower = next((c for c in order if limits[c] > 12), None)
        if narrower is None:
            return wrapped, widths
        limits[narrower] -= 2


def render_table(header: list[str], rows: list[list[str]], path: Path) -> int:
    """Draw one table to path. Returns its width in CSS pixels."""
    fig = plt.figure(dpi=LOGICAL_DPI, facecolor=SURFACE)
    fig.canvas.get_renderer()
    wrapped, widths = wrap_table(fig, header, rows)
    heights = [max(len(lines) for lines in row) * LINE_PX + 2 * PAD_Y_PX for row in wrapped]
    total_w, total_h = sum(widths), sum(heights)

    fig.set_size_inches(total_w / LOGICAL_DPI, total_h / LOGICAL_DPI)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, total_w)
    ax.set_ylim(total_h, 0)
    ax.axis("off")

    table = [header] + rows
    right = [c > 0 and is_numeric_column(rows, c) for c in range(len(header))]
    y = 0.0
    for r, row in enumerate(wrapped):
        x = 0.0
        for c, lines in enumerate(row):
            _, style = parse_cell(table[r][c], row_label=r > 0 and c == 0)
            kwargs = {"fontweight": "bold"} if r == 0 else {}
            anchor = x + widths[c] - PAD_X_PX if right[c] else x + PAD_X_PX
            for i, line in enumerate(lines):
                ax.text(
                    anchor, y + PAD_Y_PX + (i + 0.5) * LINE_PX, line,
                    ha="right" if right[c] else "left", va="center", fontsize=FONT_SIZE,
                    color=INK, parse_math=False, **{**style, **kwargs},
                )
            x += widths[c]
        y += heights[r]
        is_header = r == 0
        if r < len(wrapped) - 1:
            ax.plot([0, total_w], [y, y], color=HEADER_RULE if is_header else RULE, linewidth=1.0 if is_header else 0.6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=SAVE_DPI, facecolor=SURFACE)
    plt.close(fig)
    return round(total_w)


def image_tag(name: str, alt: str, width: int) -> str:
    return f'<img src="{FIGURES}/{name}" alt="{alt}" width="{width}">'


def main() -> None:
    lines = README.read_text().splitlines()
    out: list[str] = []
    heading = "README"
    number = 0
    in_fence = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            in_fence = not in_fence
        if not in_fence and line.startswith("#"):
            heading = line.lstrip("# ")
        is_table = (
            not in_fence and line.startswith("|")
            and i + 1 < len(lines) and re.match(r"\|[\s:|-]+\|$", lines[i + 1].strip())
        )
        if not is_table:
            out.append(line)
            i += 1
            continue

        end = i
        while end < len(lines) and lines[end].startswith("|"):
            end += 1
        header = split_row(lines[i])
        rows = [split_row(row) for row in lines[i + 2:end]]
        number += 1
        name = f"table_{number:02d}.png"
        width = render_table(header, rows, ROOT / FIGURES / name)

        alt = f"{heading}: {', '.join(parse_cell(c)[0] for c in header if c)}".replace('"', "'")
        wrapped_already = any(prev.startswith("<details>") for prev in out[-2:])
        if wrapped_already:
            out[-4 if out[-1] == "" else -3] = image_tag(name, alt, width)
        else:
            out += [image_tag(name, alt, width), "", "<details><summary>Table as text</summary>", ""]
        out += lines[i:end]
        if not wrapped_already:
            out += ["", "</details>"]
        i = end
    README.write_text("\n".join(out) + "\n")
    print(f"rendered {number} tables to {FIGURES}/")


if __name__ == "__main__":
    main()
