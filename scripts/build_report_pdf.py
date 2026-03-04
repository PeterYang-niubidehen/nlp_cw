from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render REPORT_FINAL.md to a readable PDF.")
    parser.add_argument("--input", type=Path, default=Path("reports/REPORT_FINAL.md"))
    parser.add_argument("--output", type=Path, default=Path("reports/REPORT_FINAL.pdf"))
    return parser.parse_args()


def clean_inline(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    return text


def split_heading(line: str) -> tuple[int, str] | None:
    stripped = line.lstrip()
    if not stripped.startswith("#"):
        return None
    level = len(stripped) - len(stripped.lstrip("#"))
    title = stripped[level:].strip()
    if not title:
        return None
    return level, clean_inline(title)


def styled_lines(md_text: str) -> list[tuple[str, int, str, str]]:
    """
    Return tuples: (text, font_size, font_weight, font_family)
    """
    out: list[tuple[str, int, str, str]] = []
    in_code = False

    for raw in md_text.splitlines():
        line = raw.rstrip("\n")

        if line.strip().startswith("```"):
            in_code = not in_code
            if not in_code:
                out.append(("", 9, "normal", "DejaVu Sans"))
            continue

        if in_code:
            cleaned = line.replace("\t", "    ")
            out.append((cleaned, 8, "normal", "DejaVu Sans Mono"))
            continue

        if line.strip() == "---":
            out.append(("", 9, "normal", "DejaVu Sans"))
            continue

        heading = split_heading(line)
        if heading is not None:
            level, title = heading
            if level == 1:
                out.append((title, 16, "bold", "DejaVu Sans"))
            elif level == 2:
                out.append((title, 13, "bold", "DejaVu Sans"))
            else:
                out.append((title, 11, "bold", "DejaVu Sans"))
            out.append(("", 9, "normal", "DejaVu Sans"))
            continue

        stripped = line.strip()
        if stripped.startswith("- "):
            out.append(("• " + clean_inline(stripped[2:]), 10, "normal", "DejaVu Sans"))
            continue

        if re.match(r"^\d+\.\s+", stripped):
            out.append((clean_inline(stripped), 10, "normal", "DejaVu Sans"))
            continue

        out.append((clean_inline(line), 10, "normal", "DejaVu Sans"))

    return out


def wrap_for_size(text: str, size: int, family: str) -> list[str]:
    if text == "":
        return [""]
    if family == "DejaVu Sans Mono":
        width = 96
    elif size >= 14:
        width = 70
    elif size >= 12:
        width = 80
    else:
        width = 95
    return textwrap.wrap(text, width=width, replace_whitespace=False, drop_whitespace=False) or [""]


def line_step(size: int) -> float:
    if size >= 16:
        return 0.028
    if size >= 13:
        return 0.023
    if size >= 11:
        return 0.020
    return 0.018


def render_pdf(lines: list[tuple[str, int, str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))  # A4
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        y = 0.97
        page = 1

        def new_page() -> tuple[plt.Figure, plt.Axes, float]:
            nonlocal page
            pdf.savefig(fig)
            plt.close(fig)
            page += 1
            f = plt.figure(figsize=(8.27, 11.69))
            a = f.add_axes([0, 0, 1, 1])
            a.axis("off")
            return f, a, 0.97

        for text, size, weight, family in lines:
            wrapped = wrap_for_size(text, size, family)
            step = line_step(size)
            for piece in wrapped:
                if y < 0.05:
                    fig, ax, y = new_page()
                ax.text(
                    0.06,
                    y,
                    piece,
                    fontsize=size,
                    fontweight=weight,
                    fontfamily=family,
                    va="top",
                    ha="left",
                )
                y -= step

        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    md_text = args.input.read_text(encoding="utf-8")
    lines = styled_lines(md_text)
    render_pdf(lines, args.output)
    print(args.output)
    print(f"bytes {args.output.stat().st_size}")


if __name__ == "__main__":
    main()
