#!/usr/bin/env python3
"""Generate the report's evidence-bearing SVG figures using only the stdlib."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "mirror-geometry3k"
OUT = REPORT / "images"
DATA = json.loads((REPORT / "results.json").read_text())

INK = "#18202A"
MUTED = "#637083"
GRID = "#DCE3EA"
BLUE = "#2F6FED"
TEAL = "#1B9E77"
ORANGE = "#E07A2D"
PURPLE = "#7E57C2"
RED = "#D64545"


def esc(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def frame(title: str, subtitle: str, body: str, width: int = 900, height: int = 520) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">
<rect width="100%" height="100%" fill="#FFFFFF" rx="14"/>
<text x="54" y="48" font-family="Inter,system-ui,sans-serif" font-size="24" font-weight="700" fill="{INK}">{esc(title)}</text>
<text x="54" y="75" font-family="Inter,system-ui,sans-serif" font-size="14" fill="{MUTED}">{esc(subtitle)}</text>
{body}
</svg>
"""


def grouped_bars(
    title: str,
    subtitle: str,
    groups: list[str],
    series: list[tuple[str, list[float], str]],
    path: Path,
    ymax: float,
    note: str,
) -> None:
    left, top, width, height = 80, 115, 760, 300
    parts: list[str] = []
    for tick in range(6):
        value = ymax * tick / 5
        y = top + height - height * tick / 5
        parts.append(f'<line x1="{left}" y1="{y}" x2="{left+width}" y2="{y}" stroke="{GRID}"/>')
        parts.append(
            f'<text x="{left-12}" y="{y+5}" text-anchor="end" font-family="system-ui" font-size="12" fill="{MUTED}">{value*100:.0f}%</text>'
        )
    group_w = width / len(groups)
    bar_w = min(46, group_w / (len(series) + 1))
    for gi, group in enumerate(groups):
        center = left + group_w * (gi + 0.5)
        parts.append(
            f'<text x="{center}" y="{top+height+27}" text-anchor="middle" font-family="system-ui" font-size="13" fill="{INK}">{esc(group)}</text>'
        )
        for si, (name, values, color) in enumerate(series):
            x = center + (si - (len(series) - 1) / 2) * (bar_w + 8) - bar_w / 2
            bar_h = height * values[gi] / ymax
            y = top + height - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" rx="4" fill="{color}"/>')
            parts.append(
                f'<text x="{x+bar_w/2:.1f}" y="{y-7:.1f}" text-anchor="middle" font-family="system-ui" font-size="11" fill="{INK}">{values[gi]*100:.1f}</text>'
            )
    legend_x = left
    for name, _, color in series:
        parts.append(f'<rect x="{legend_x}" y="92" width="13" height="13" rx="2" fill="{color}"/>')
        parts.append(
            f'<text x="{legend_x+19}" y="103" font-family="system-ui" font-size="12" fill="{INK}">{esc(name)}</text>'
        )
        legend_x += 145
    parts.append(
        f'<text x="{left}" y="487" font-family="system-ui" font-size="12" fill="{MUTED}">{esc(note)}</text>'
    )
    path.write_text(frame(title, subtitle, "\n".join(parts)))


def baseline() -> None:
    b = DATA["baseline"]
    grouped_bars(
        "Equivalent views expose different successes",
        "Qwen3-VL-4B-Instruct on 96 held-out Geometry3K problems (pass@1)",
        ["Text", "Diagram", "Combined", "Any view"],
        [
            (
                "Correct",
                [
                    b["textAccuracy"],
                    b["imageAccuracy"],
                    b["combinedAccuracy"],
                    b["oracleAnyViewCorrect"],
                ],
                BLUE,
            )
        ],
        OUT / "baseline_views.svg",
        0.45,
        "62.5% disagreed across views; 29.2% had a complementary success (some but not all views correct).",
    )


def seeds() -> None:
    rows = DATA["seedComparisons"]
    grouped_bars(
        "The MIRROR advantage changes sign across matched seeds",
        "Mean exact-answer accuracy over text, diagram, and combined views",
        [f"Seed {row['seed']}" for row in rows],
        [
            ("Matched SFT", [row["sftMeanAccuracy"] for row in rows], BLUE),
            ("Adaptive MIRROR", [row["mirrorMeanAccuracy"] for row in rows], TEAL),
            ("No reciprocal", [row["noRklMeanAccuracy"] for row in rows], ORANGE),
        ],
        OUT / "headline_seeds.svg",
        0.55,
        "MIRROR − SFT: +1.04, −1.04, +0.35 percentage points; average +0.12 points.",
    )


def teachers() -> None:
    t = DATA["teacherSelectionSeed17"]
    labels = ["Adaptive", "Fixed combined", "Fixed text", "Fixed image"]
    keys = ["adaptive", "fixedCombined", "fixedText", "fixedImage"]
    grouped_bars(
        "Adaptive teachers narrowly lead fixed teachers",
        "Seed 17; same examples, optimizer steps, LoRA rank, and evaluation set",
        labels,
        [
            ("Mean accuracy", [t[key]["meanAccuracy"] for key in keys], PURPLE),
            ("All views correct", [t[key]["allCorrect"] for key in keys], TEAL),
        ],
        OUT / "teacher_selection.svg",
        0.55,
        "Adaptive selection used text/image/combined teachers on 32.8%/32.8%/34.4% of training problems.",
    )


def kl_sweep() -> None:
    rows = DATA["klSweepSeed17"]
    labels = ["0", ".001", ".005", ".01", ".05"]
    grouped_bars(
        "Reverse-KL strength is not monotonic",
        "Seed 17 adaptive selection; λ=0 is the no-reciprocal control",
        labels,
        [
            ("Mean accuracy", [row["meanAccuracy"] for row in rows], BLUE),
            ("All views correct", [row["allCorrect"] for row in rows], ORANGE),
        ],
        OUT / "kl_sensitivity.svg",
        0.55,
        "λ=.05 is best here, unlike the paper’s λ=.01 optimum; this is a short frozen-teacher LoRA approximation.",
    )


def consistency() -> None:
    seed = DATA["seedComparisons"][0]
    b = DATA["baseline"]
    grouped_bars(
        "Training improves consistency, but reciprocal loss is not required",
        "Seed 17 held-out sample",
        ["Base", "SFT", "MIRROR", "No reciprocal"],
        [
            (
                "All-view agreement",
                [
                    1 - b["disagreement"],
                    seed["sftAgreement"],
                    seed["mirrorAgreement"],
                    seed["noRklAgreement"],
                ],
                TEAL,
            ),
            (
                "All views correct",
                [
                    b["allViewCorrect"],
                    seed["sftAllCorrect"],
                    seed["mirrorAllCorrect"],
                    seed["noRklAllCorrect"],
                ],
                ORANGE,
            ),
        ],
        OUT / "consistency.svg",
        0.85,
        "All methods resolve much baseline disagreement; the no-reciprocal control matches or exceeds λ=.01.",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    baseline()
    seeds()
    teachers()
    kl_sweep()
    consistency()
    print(f"Wrote 5 figures to {OUT}")


if __name__ == "__main__":
    main()
