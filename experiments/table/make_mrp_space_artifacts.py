#!/usr/bin/env python3
"""Create paper tables/figures for MRP reliability-space analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASET_ORDER = [
    "ESCI-Rerank-US",
    "MSLR-WEB10K",
    "Amazon ESCI",
    "SciDocs",
    "WANDS",
    "Alloprof-Rerank",
]

BASE_ORDER = ["Uncal.", "TS", "DIAG", "Spline", "h-cal", "SMART"]

K_BY_DATASET = {
    "Amazon ESCI": 4,
    "MSLR-WEB10K": 5,
    "Alloprof-Rerank": 2,
    "ESCI-Rerank-US": 2,
    "WANDS": 3,
    "SciDocs": 2,
}

DATASET_TEX = {
    "ESCI-Rerank-US": r"\shortstack{ESCI-Rerank\\US}",
    "MSLR-WEB10K": r"\shortstack{MSLR-\\WEB10K}",
    "Amazon ESCI": r"\shortstack{Amazon\\ESCI}",
    "SciDocs": "SciDocs",
    "WANDS": "WANDS",
    "Alloprof-Rerank": r"\shortstack{Alloprof-\\Rerank}",
}


def fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def fmt_pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}\\%"


def write_agreement_table(agreement_dir: Path, table_dir: Path) -> None:
    dataset = pd.read_csv(agreement_dir / "dataset_summary.csv")
    overall = pd.read_csv(agreement_dir / "overall.csv")

    rows = []
    for name in DATASET_ORDER:
        group = dataset[dataset["Dataset"] == name].set_index("Score")
        rows.append(
            [
                name,
                fmt(group.loc["confidence", "Pearson_mean"]),
                fmt(group.loc["q", "Pearson_mean"]),
                fmt(group.loc["confidence", "MAE_mean"], 4),
                fmt(group.loc["q", "MAE_mean"], 4),
            ]
        )

    overall = overall.set_index("Score")
    rows.append(
        [
            r"\textbf{Avg.}",
            r"\textbf{" + fmt(overall.loc["confidence", "Pearson_mean"]) + "}",
            r"\textbf{" + fmt(overall.loc["q", "Pearson_mean"]) + "}",
            r"\textbf{" + fmt(overall.loc["confidence", "MAE_mean"], 4) + "}",
            r"\textbf{" + fmt(overall.loc["q", "MAE_mean"], 4) + "}",
        ]
    )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Sample-wise agreement across base calibrators. For each dataset, we compare every pair of base calibrators on the same test samples. Pearson and MAE are averaged over calibrator pairs and protocol seeds. After MRP, the correctness reliability \(q^{\mathcal{A}}\) is much less sensitive to the choice of base calibrator than the calibrated confidence \(\hat{c}^{\mathcal{A}}\).}",
        r"\label{tab:cross_calibrator_agreement}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\multirow{2}{*}{Dataset} & \multicolumn{2}{c}{Pearson \(\uparrow\)} & \multicolumn{2}{c}{MAE \(\downarrow\)} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r" & \(\hat{c}^{\mathcal{A}}\) & \(q^{\mathcal{A}}\) & \(\hat{c}^{\mathcal{A}}\) & \(q^{\mathcal{A}}\) \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (table_dir / "cross_calibrator_agreement.tex").write_text("\n".join(lines), encoding="utf-8")


def write_simplex_table(projection_dir: Path, table_dir: Path) -> None:
    summary = pd.read_csv(projection_dir / "simplex_projection_summary.csv")

    def mean_row(dataset_name: str, base_name: str, variant: str) -> pd.Series:
        subset = summary[
            (summary["Dataset"] == dataset_name)
            & (summary["Base"] == base_name)
            & (summary["Variant"] == variant)
        ]
        if subset.empty:
            raise ValueError(f"missing simplex projection row: {dataset_name} / {base_name} / {variant}")
        return subset.mean(numeric_only=True)

    def fmt_delta(value: float, digits: int = 4) -> str:
        rounded = round(float(value), digits)
        if rounded == 0:
            return f"{0.0:.{digits}f}"
        text = f"{rounded:+.{digits}f}"
        if rounded < 0:
            return r"\mrcimprove{" + text + "}"
        return text

    rows = []
    for dataset_name in DATASET_ORDER:
        dataset_rows = []
        for base_name in BASE_ORDER:
            if summary[
                (summary["Dataset"] == dataset_name)
                & (summary["Base"] == base_name)
                & (summary["Variant"] == "Base")
            ].empty:
                continue
            base = mean_row(dataset_name, base_name, "Base")
            mrc = mean_row(dataset_name, base_name, "MRP-PowerTS")
            dataset_rows.append(
                [
                    base_name,
                    fmt(base["FixedECE_mean"], 4),
                    fmt(mrc["FixedECE_mean"], 4),
                    fmt_delta(mrc["FixedECE_mean"] - base["FixedECE_mean"]),
                    fmt(base["TopECE_mean"], 4),
                    fmt(mrc["TopECE_mean"], 4),
                    fmt_delta(mrc["TopECE_mean"] - base["TopECE_mean"]),
                    fmt(base["NLL_mean"], 4),
                    fmt(mrc["NLL_mean"], 4),
                    fmt_delta(mrc["NLL_mean"] - base["NLL_mean"]),
                    fmt(base["Brier_mean"], 4),
                    fmt(mrc["Brier_mean"], 4),
                    fmt_delta(mrc["Brier_mean"] - base["Brier_mean"]),
                    fmt_pct(mrc["Power_q_below_uniform_rate_mean"]),
                    fmt_pct(mrc["Power_solved_rate_mean"]),
                ]
            )
        if dataset_rows:
            rows.append((dataset_name, dataset_rows))

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{1.9pt}",
        r"\renewcommand{\arraystretch}{0.92}",
        r"\caption{Top-label simplex projection analysis. Base is the original calibrated vector, and MRC is the MRP-induced power-temperature projection. For Fixed ECE, Top ECE, NLL, and Brier, \(\Delta\) is MRC minus Base. Blue bold negative \(\Delta\) values indicate improvements, and lower is better for all four metrics. The \(q<1/K\) column reports infeasible targets. Solved gives exact rates.}",
        r"\label{tab:simplex_projection}",
        r"\newcommand{\mrcimprove}[1]{\textcolor{blue}{\textbf{#1}}}",
        r"\begin{tabular}{lclcccccccccccccc}",
        r"\toprule",
        r"\multirow{2}{*}{Dataset} & \multirow{2}{*}{\(K\)} & \multirow{2}{*}{Cal.} & \multicolumn{3}{c}{Fixed ECE\(\downarrow\)} & \multicolumn{3}{c}{Top ECE\(\downarrow\)} & \multicolumn{3}{c}{NLL\(\downarrow\)} & \multicolumn{3}{c}{Brier\(\downarrow\)} & \multirow{2}{*}{\(q<1/K\)} & \multirow{2}{*}{Solved} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}\cmidrule(lr){13-15}",
        r" & & & Base & MRC & \(\Delta\) & Base & MRC & \(\Delta\) & Base & MRC & \(\Delta\) & Base & MRC & \(\Delta\) & & \\",
        r"\midrule",
    ]
    for dataset_idx, (dataset_name, dataset_rows) in enumerate(rows):
        span = len(dataset_rows)
        for row_idx, row in enumerate(dataset_rows):
            prefix = (
                [rf"\multirow{{{span}}}{{*}}{{{DATASET_TEX[dataset_name]}}}", rf"\multirow{{{span}}}{{*}}{{{K_BY_DATASET[dataset_name]}}}"]
                if row_idx == 0
                else ["", ""]
            )
            lines.append(" & ".join(prefix + row) + r" \\")
        if dataset_idx != len(rows) - 1:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\renewcommand{\arraystretch}{1.0}",
        r"\end{table*}",
        "",
    ]
    (table_dir / "simplex_projection.tex").write_text("\n".join(lines), encoding="utf-8")


def heatmap_matrix(pair_summary: pd.DataFrame, dataset: str, score: str) -> np.ndarray:
    matrix = np.zeros((len(BASE_ORDER), len(BASE_ORDER)), dtype=np.float64)
    subset = pair_summary[(pair_summary["Dataset"] == dataset) & (pair_summary["Score"] == score)]
    lookup = {(row.BaseLeft, row.BaseRight): row.MAE_mean for row in subset.itertuples()}
    for i, left in enumerate(BASE_ORDER):
        for j, right in enumerate(BASE_ORDER):
            if i == j:
                matrix[i, j] = 0.0
            else:
                key = (left, right) if (left, right) in lookup else (right, left)
                matrix[i, j] = float(lookup.get(key, np.nan))
    return matrix


def write_heatmap(agreement_dir: Path, figure_dir: Path) -> None:
    pair_summary = pd.read_csv(agreement_dir / "pairwise_summary.csv")
    dataset = "MSLR-WEB10K"
    matrices = [
        (r"Calibrated confidence $\hat{c}^{\mathcal{A}}$", heatmap_matrix(pair_summary, dataset, "confidence")),
        (r"MRP reliability $q^{\mathcal{A}}$", heatmap_matrix(pair_summary, dataset, "q")),
    ]
    vmax = max(float(np.nanmax(mat)) for _, mat in matrices)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    images = []
    for ax, (title, mat) in zip(axes, matrices):
        im = ax.imshow(mat, vmin=0.0, vmax=vmax, cmap="YlOrRd")
        images.append(im)
        ax.set_title(title)
        ax.set_xticks(range(len(BASE_ORDER)))
        ax.set_yticks(range(len(BASE_ORDER)))
        ax.set_xticklabels(BASE_ORDER, rotation=35, ha="right")
        ax.set_yticklabels(BASE_ORDER)
        for i in range(len(BASE_ORDER)):
            for j in range(len(BASE_ORDER)):
                color = "white" if mat[i, j] > vmax * 0.55 else "black"
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(images[0], ax=axes, shrink=0.86, label="sample-wise MAE")
    pdf = figure_dir / "mrp_mslr_calibrator_agreement_heatmap.pdf"
    png = figure_dir / "mrp_mslr_calibrator_agreement_heatmap.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agreement-dir", default="results/analysis/mrp_cross_calibrator_agreement_gpu0")
    parser.add_argument("--projection-dir", default="results/tables/mrp_simplex_projection_gpu0")
    parser.add_argument("--table-dir", default="paper/table")
    parser.add_argument("--figure-dir", default="paper/figure")
    parser.add_argument("--write-agreement", action="store_true", help="Also write optional cross-calibrator agreement table/heatmap.")
    args = parser.parse_args()

    agreement_dir = Path(args.agreement_dir)
    projection_dir = Path(args.projection_dir)
    table_dir = Path(args.table_dir)
    figure_dir = Path(args.figure_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    write_simplex_table(projection_dir, table_dir)
    if args.write_agreement:
        write_agreement_table(agreement_dir, table_dir)
        write_heatmap(agreement_dir, figure_dir)
        print(table_dir / "cross_calibrator_agreement.tex")
        print(figure_dir / "mrp_mslr_calibrator_agreement_heatmap.pdf")
    print(table_dir / "simplex_projection.tex")


if __name__ == "__main__":
    main()
