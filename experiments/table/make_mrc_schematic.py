#!/usr/bin/env python3
"""Draw a schematic for MRP-induced reliability calibration on a simplex."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyArrowPatch, PathPatch
from matplotlib.patches import Polygon

COLORS = {
    "uncal": "#0072B2",  # Okabe-Ito blue
    "cal": "#E69F00",  # Okabe-Ito orange
    "temperature": "#C44E52",  # muted red temperature path
    "mrp": "#6A3D9A",  # muted purple for the reliability contour
    "mrc": "#8E1B54",  # plum
}


def add_step_marker(ax, xy: np.ndarray, number: int, dx: float = 0.0, dy: float = 0.0) -> None:
    ax.text(
        xy[0] + dx,
        xy[1] + dy,
        str(number),
        ha="center",
        va="center",
        fontsize=7.8,
        fontweight="bold",
        color="black",
        bbox={
            "boxstyle": "circle,pad=0.17",
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": 0.75,
        },
        zorder=7,
    )


def bary_to_xy(top: float, runner: float, rest: float) -> np.ndarray:
    vertices = np.asarray(
        [
            [0.5, np.sqrt(3) / 2.0],  # top label
            [0.0, 0.0],  # runner-up
            [1.0, 0.0],  # rest
        ],
        dtype=float,
    )
    weights = np.asarray([top, runner, rest], dtype=float)
    weights = weights / weights.sum()
    return weights @ vertices


def contour_for_top(top_mass: float) -> tuple[np.ndarray, np.ndarray]:
    # All points with fixed top mass have runner+rest = 1-top.
    left = bary_to_xy(top_mass, 1.0 - top_mass, 0.0)
    right = bary_to_xy(top_mass, 0.0, 1.0 - top_mass)
    return left, right


def project_point_to_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    segment = end - start
    t = np.dot(point - start, segment) / np.dot(segment, segment)
    t = np.clip(t, 0.0, 1.0)
    return start + t * segment


def cubic_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: float) -> np.ndarray:
    return (
        (1 - t) ** 3 * p0
        + 3 * (1 - t) ** 2 * t * p1
        + 3 * (1 - t) * t**2 * p2
        + t**3 * p3
    )


def main() -> None:
    out_dir = Path("paper/figure")
    out_dir.mkdir(parents=True, exist_ok=True)

    top_v = bary_to_xy(1, 0, 0)
    run_v = bary_to_xy(0, 1, 0)
    rest_v = bary_to_xy(0, 0, 1)

    center = np.asarray([1 / 3, 1 / 3, 1 / 3])
    sample_point = np.asarray([0.72, 0.24, 0.04])
    center_xy = bary_to_xy(*center)
    sample_xy = bary_to_xy(*sample_point)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10.4,
            "axes.linewidth": 0.8,
        }
    )
    fig, ax = plt.subplots(figsize=(4.8, 4.05))
    triangle = Polygon([top_v, run_v, rest_v], closed=True, facecolor="#f7f7f7", edgecolor="black", linewidth=2.2, zorder=1)
    ax.add_patch(triangle)
    left_foot = project_point_to_segment(center_xy, top_v, run_v)
    right_foot = project_point_to_segment(center_xy, top_v, rest_v)
    decision_region = Polygon(
        [top_v, right_foot, center_xy, left_foot],
        closed=True,
        facecolor="#dddddd",
        edgecolor="none",
        zorder=1.5,
    )
    ax.add_patch(decision_region)
    ax.plot([center_xy[0], left_foot[0]], [center_xy[1], left_foot[1]], color="#8a8a8a", linewidth=0.7, zorder=2)
    ax.plot([center_xy[0], right_foot[0]], [center_xy[1], right_foot[1]], color="#8a8a8a", linewidth=0.7, zorder=2)
    mrp_left, mrp_right = contour_for_top(0.52)
    mrp_y = mrp_left[1]
    ax.plot(
        [mrp_left[0], mrp_right[0]],
        [mrp_left[1], mrp_right[1]],
        color=COLORS["mrp"],
        linewidth=2.0,
        linestyle=(0, (1.3, 2.0)),
        zorder=3,
    )
    ax.text(
        mrp_left[0] - 0.035,
        mrp_left[1],
        r"$\mathbf{MRP}$ $q_{\mathrm{MRP}}^{\mathcal{A}}(x)$",
        ha="right",
        va="center",
        fontsize=10.6,
        fontweight="bold",
        color=COLORS["mrp"],
    )
    curve_c1 = bary_to_xy(0.43, 0.08, 0.49)
    curve_c2 = bary_to_xy(0.76, 0.02, 0.22)
    cal_xy = cubic_bezier(center_xy, curve_c1, curve_c2, top_v, 0.62)
    curve_samples = np.asarray([cubic_bezier(center_xy, curve_c1, curve_c2, top_v, t) for t in np.linspace(0, 1, 400)])
    mrc_xy = curve_samples[np.argmin(np.abs(curve_samples[:, 1] - mrp_y))]
    temp_curve = MplPath(
        [center_xy, curve_c1, curve_c2, top_v],
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
    )
    ax.add_patch(PathPatch(temp_curve, facecolor="none", edgecolor=COLORS["temperature"], linewidth=1.7, zorder=3))
    ax.text(
        mrc_xy[0] - 0.020,
        mrc_xy[1] - 0.110,
        "temperature path",
        ha="left",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        color=COLORS["temperature"],
        zorder=4,
    )
    triangle_outline = Polygon([top_v, run_v, rest_v], closed=True, facecolor="none", edgecolor="black", linewidth=2.2, zorder=4)
    ax.add_patch(triangle_outline)
    ax.add_patch(
        FancyArrowPatch(
            posA=(sample_xy[0] + 0.012, sample_xy[1]),
            posB=(cal_xy[0] - 0.012, cal_xy[1]),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.8,
            color="black",
            zorder=4.5,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            posA=(cal_xy[0], cal_xy[1] - 0.015),
            posB=(mrc_xy[0], mrc_xy[1] + 0.014),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.8,
            color="black",
            zorder=4.5,
        )
    )
    ax.scatter([sample_xy[0]], [sample_xy[1]], s=50, color=COLORS["uncal"], edgecolor="black", linewidth=1.1, zorder=5)
    add_step_marker(ax, sample_xy, 1, dx=-0.020, dy=0.037)
    ax.text(sample_xy[0] - 0.06, sample_xy[1] + 0.002, r"Uncal. $\hat{\mathbf{p}}^{0}(x)$", ha="right", va="center", fontsize=10.4, fontweight="bold", color=COLORS["uncal"])
    ax.scatter([cal_xy[0]], [cal_xy[1]], s=50, color=COLORS["cal"], edgecolor="black", linewidth=1.1, zorder=5)
    add_step_marker(ax, cal_xy, 2, dx=0.035, dy=0.040)
    ax.text(cal_xy[0] + 0.195, cal_xy[1] + 0.055, r"Calibrator $\mathcal{A}$ $\hat{\mathbf{p}}^{\mathcal{A}}(x)$", ha="center", va="bottom", fontsize=10.4, fontweight="bold", color=COLORS["cal"])
    ax.scatter([mrc_xy[0]], [mrc_xy[1]], s=56, color=COLORS["mrc"], edgecolor="black", linewidth=1.1, zorder=6)
    add_step_marker(ax, mrc_xy, 4, dx=0.034, dy=0.037)
    ax.text(mrc_xy[0] + 0.030, mrc_xy[1] - 0.005, r"$\mathbf{MRC}$ $\hat{\mathbf{p}}^{\mathrm{MRC}}(x)$", ha="left", va="top", fontsize=10.6, fontweight="bold", color=COLORS["mrc"])
    add_step_marker(ax, np.asarray([mrp_left[0], mrp_left[1]]), 3, dx=-0.025, dy=0.035)
    ax.scatter([center_xy[0]], [center_xy[1]], s=18, color="black", edgecolor="black", linewidth=0.0, zorder=5)
    ax.scatter([top_v[0]], [top_v[1]], s=18, color="black", edgecolor="black", linewidth=0.0, zorder=6)
    ax.text(top_v[0] + 0.045, top_v[1] + 0.005, r"Decision label $d(x)$", ha="left", va="top", fontsize=10.4)
    ax.text(
        center_xy[0],
        0.15,
        "Outside the\nfixed-decision area",
        ha="center",
        va="center",
        fontsize=11.4,
        fontweight="bold",
        linespacing=1.05,
        color="#555555",
    )

    ax.set_xlim(-0.10, 1.12)
    ax.set_ylim(-0.035, 0.945)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(out_dir / "mrc_simplex_schematic.pdf", bbox_inches="tight")
    plt.close(fig)
    print(out_dir / "mrc_simplex_schematic.pdf")


if __name__ == "__main__":
    main()
