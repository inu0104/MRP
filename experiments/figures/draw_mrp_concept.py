"""Draw a compact MRP concept diagram for the paper.

The figure is intentionally vector-like and editable through this script.
It avoids a stage-by-stage tutorial layout and instead tracks the same
candidate predictions through calibrated confidence, label-wise reliability
projection, and risk-based ordering.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper" / "figure"


COLORS = {
    "A": "#4C2A85",
    "B": "#2B7BB9",
    "C": "#5E9E3D",
    "D": "#C77C2A",
}


SAMPLES = [
    {
        "id": "A",
        "item": "Candidate A",
        "snippet": "directly answers the query",
        "label": "Exact",
        "c": 0.72,
        "q": 0.88,
        "y": 0.74,
    },
    {
        "id": "B",
        "item": "Candidate B",
        "snippet": "partially relevant evidence",
        "label": "Partial",
        "c": 0.58,
        "q": 0.61,
        "y": 0.56,
    },
    {
        "id": "C",
        "item": "Candidate C",
        "snippet": "ambiguous relevance signal",
        "label": "Low",
        "c": 0.66,
        "q": 0.39,
        "y": 0.38,
    },
    {
        "id": "D",
        "item": "Candidate D",
        "snippet": "mostly off-topic result",
        "label": "Irrel.",
        "c": 0.45,
        "q": 0.52,
        "y": 0.20,
    },
]


def add_round_box(ax, xy, width, height, fc="white", ec="#CCCCCC", lw=0.9, r=0.025):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.01,rounding_size={r}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    return patch


def add_arrow(ax, start, end, color="#999999", lw=0.9, alpha=1.0, ms=8):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        alpha=alpha,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arr)
    return arr


def rel_curve(label: str, x: np.ndarray) -> np.ndarray:
    if label == "Exact":
        return 0.18 + 0.78 * (1 - np.exp(-3.2 * x)) / (1 - np.exp(-3.2))
    if label == "Partial":
        return 0.08 + 0.82 * x**0.92
    if label == "Low":
        return 0.04 + 0.54 * x**1.45
    return 0.05 + 0.52 * x**1.05


def draw_candidate_column(ax):
    ax.text(0.02, 0.94, "Relevance prediction task", fontsize=8.8, weight="bold", va="top")
    add_round_box(ax, (0.02, 0.79), 0.27, 0.12, fc="#FAFAFA")
    ax.text(
        0.035,
        0.845,
        "How do noise cancelling\nheadphones reduce noise?",
        fontsize=7.5,
        va="center",
    )

    for sample in SAMPLES:
        y = sample["y"]
        add_round_box(ax, (0.02, y - 0.045), 0.25, 0.075, fc="white")
        ax.scatter(0.04, y - 0.007, s=150, color=COLORS[sample["id"]], zorder=5)
        ax.text(0.04, y - 0.007, sample["id"], ha="center", va="center", color="white", fontsize=8, weight="bold")
        ax.text(0.075, y + 0.006, sample["item"], fontsize=7.2, va="center")
        ax.text(0.075, y - 0.020, sample["snippet"], fontsize=6.2, color="#777777", va="center")


def draw_calibrated_column(ax):
    ax.text(0.34, 0.94, "Calibrated prediction", fontsize=8.8, weight="bold", va="top")

    for sample in SAMPLES:
        y = sample["y"]
        add_arrow(ax, (0.275, y - 0.008), (0.325, y - 0.008), color="#C7C7C7", lw=0.8, ms=7)
        add_round_box(ax, (0.34, y - 0.047), 0.23, 0.078, fc="#FFFFFF")
        ax.text(
            0.355,
            y + 0.009,
            rf"$d={sample['label']}$",
            fontsize=7.0,
            color=COLORS[sample["id"]],
            weight="bold",
            va="center",
        )
        ax.text(0.355, y - 0.022, rf"$\hat{{c}}^A={sample['c']:.2f}$", fontsize=7.0, va="center")
        bar_x, bar_y = 0.475, y - 0.024
        add_round_box(ax, (bar_x, bar_y), 0.075, 0.015, fc="#F0F0F0", ec="#BBBBBB", lw=0.5, r=0.004)
        ax.add_patch(
            FancyBboxPatch(
                (bar_x, bar_y),
                0.075 * sample["c"],
                0.015,
                boxstyle="round,pad=0.0,rounding_size=0.003",
                linewidth=0,
                facecolor=COLORS[sample["id"]],
                alpha=0.85,
            )
        )


def draw_curve_panel(ax):
    ax.text(0.615, 0.94, "Label-wise reliability", fontsize=8.8, weight="bold", va="top")

    px, py, pw, ph = 0.61, 0.22, 0.22, 0.58
    add_round_box(ax, (px - 0.01, py - 0.03), pw + 0.02, ph + 0.06, fc="#FFFFFF", ec="#DDDDDD", lw=0.8, r=0.015)

    ax.plot([px, px], [py, py + ph], color="#333333", lw=0.7)
    ax.plot([px, px + pw], [py, py], color="#333333", lw=0.7)
    ax.text(px + pw * 0.48, py - 0.055, r"calibrated confidence $c$", fontsize=6.6, ha="center")
    ax.text(px - 0.044, py + ph * 0.52, r"correctness $q$", fontsize=6.6, rotation=90, va="center")

    xs = np.linspace(0, 1, 160)
    labels = ["Exact", "Partial", "Low", "Irrel."]
    curve_colors = ["#4C2A85", "#2B7BB9", "#5E9E3D", "#777777"]
    for label, color in zip(labels, curve_colors):
        ys = rel_curve(label, xs)
        ax.plot(px + xs * pw, py + ys * ph, color=color, lw=1.6, alpha=0.9)

    label_pos = {
        "Exact": (0.80, 0.93),
        "Partial": (0.64, 0.78),
        "Low": (0.74, 0.55),
        "Irrel.": (0.66, 0.38),
    }
    for label, (lx, ly) in label_pos.items():
        ax.text(px + lx * pw, py + ly * ph, rf"$T_{{\mathrm{{{label}}}}}$", fontsize=6.6, color="#555555")

    for sample in SAMPLES:
        xcoord = px + sample["c"] * pw
        ycoord = py + sample["q"] * ph
        src = (0.57, sample["y"] - 0.008)
        add_arrow(ax, src, (xcoord - 0.008, ycoord), color=COLORS[sample["id"]], lw=0.7, alpha=0.5, ms=6)
        ax.plot([xcoord, xcoord], [py, ycoord], color=COLORS[sample["id"]], lw=0.65, alpha=0.35, ls="--")
        ax.scatter(xcoord, ycoord, s=35, color=COLORS[sample["id"]], zorder=6)
        ax.text(xcoord + 0.006, ycoord + 0.012, sample["id"], fontsize=7, color=COLORS[sample["id"]], weight="bold")


def draw_ranking_panel(ax):
    ax.text(0.86, 0.94, "Risk order", fontsize=8.8, weight="bold", va="top")

    add_round_box(ax, (0.86, 0.50), 0.12, 0.34, fc="#F4FBF1", ec="#9CCB88", lw=0.9)
    ax.text(0.87, 0.81, "Automatic use", fontsize=7.5, color="#3D8737", weight="bold")
    add_round_box(ax, (0.86, 0.17), 0.12, 0.20, fc="#FFF4F0", ec="#D99A84", lw=0.9)
    ax.text(0.87, 0.34, "Fallback / review", fontsize=7.5, color="#B45843", weight="bold")

    ordered = sorted(SAMPLES, key=lambda s: 1 - s["q"])
    y_positions = {"A": 0.73, "B": 0.63, "D": 0.53, "C": 0.26}
    for sample in ordered:
        y = y_positions[sample["id"]]
        add_round_box(ax, (0.875, y - 0.038), 0.09, 0.062, fc="white", ec="#D5D5D5", lw=0.7, r=0.012)
        ax.scatter(0.889, y - 0.008, s=95, color=COLORS[sample["id"]], zorder=5)
        ax.text(0.889, y - 0.008, sample["id"], ha="center", va="center", fontsize=6.3, color="white", weight="bold")
        ax.text(0.905, y + 0.004, rf"$q={sample['q']:.2f}$", fontsize=6.7, va="center")
        ax.text(0.905, y - 0.022, rf"risk={1 - sample['q']:.2f}", fontsize=6.3, color="#777777", va="center")

    ax.annotate(
        "",
        xy=(0.845, 0.81),
        xytext=(0.845, 0.19),
        arrowprops=dict(arrowstyle="<|-|>", color="#999999", lw=0.9),
    )
    ax.text(0.832, 0.50, "higher reliability", rotation=90, fontsize=6.3, color="#777777", ha="center", va="center")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(12.0, 3.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_candidate_column(ax)
    draw_calibrated_column(ax)
    draw_curve_panel(ax)
    draw_ranking_panel(ax)

    ax.text(
        0.50,
        0.055,
        "Predictions stay fixed; label-wise correctness reliability changes the order.",
        ha="center",
        fontsize=8.3,
        weight="bold",
        color="#444444",
    )

    pdf_path = OUT_DIR / "mrp_concept_diagram.pdf"
    png_path = OUT_DIR / "mrp_concept_diagram.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png_path, dpi=240, bbox_inches="tight", pad_inches=0.03)
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
