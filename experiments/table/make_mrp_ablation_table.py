#!/usr/bin/env python3
"""Build the TeX ablation table for monotone reliability projection."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ORDER = [
    "M0_confidence",
    "Shared1D",
    "LabelConstant",
    "PerLabelIsotonic",
    "Label1D",
    "Label2D",
]

DISPLAY = {
    "M0_confidence": r"Conf.",
    "Shared1D": "Shared 1D",
    "LabelConstant": "Label-only intercept",
    "PerLabelIsotonic": "Per-label isotonic",
    "Label1D": r"\textbf{MRP (Label-wise 1D)}",
    "Label2D": "Label-wise 2D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overall", default="results/tables/monotone_reliability_projection_ablation_no_anchor_gpu0/mrp_overall.csv")
    parser.add_argument("--summary", default="results/tables/monotone_reliability_projection_ablation_no_anchor_gpu0/mrp_summary.csv")
    parser.add_argument("--tex", default="paper/table/mrp_ablation_results.tex")
    parser.add_argument("--csv", default="results/tables/monotone_reliability_projection_ablation_no_anchor_gpu0/mrp_ablation_compact.csv")
    return parser.parse_args()


def fmt(x: float) -> str:
    return f"{float(x):.4f}"


def delta(x: float, good: str) -> str:
    if abs(float(x)) < 5e-5:
        return "+0.0000"
    ok = x > 0 if good == "positive" else x < 0
    text = f"{float(x):+.4f}"
    return rf"\textbf{{{text}}}" if ok else text


def best_text(value: float, best_value: float) -> str:
    text = fmt(value)
    return rf"\textbf{{{text}}}" if abs(float(value) - float(best_value)) < 5e-5 else text


def main() -> None:
    args = parse_args()
    overall = pd.read_csv(args.overall).set_index("Variant")
    m0 = overall.loc["M0_confidence"]
    rows = []
    for variant in ORDER:
        if variant not in overall.index:
            continue
        row = overall.loc[variant]
        rows.append(
            {
                "Variant": variant,
                "Display": DISPLAY[variant],
                "Correct_NLL": row["Binary_NLL_mean"],
                "Delta_Correct_NLL": row["Binary_NLL_mean"] - m0["Binary_NLL_mean"],
                "Wrong_AUPRC": row["Wrong_AUPRC_mean"],
                "Delta_Wrong_AUPRC": row["Wrong_AUPRC_mean"] - m0["Wrong_AUPRC_mean"],
                "AURC": row["AURC_mean"],
                "Delta_AURC": row["AURC_mean"] - m0["AURC_mean"],
                "SelAcc50": row["SelAcc50_mean"],
                "Delta_SelAcc50": row["SelAcc50_mean"] - m0["SelAcc50_mean"],
            }
        )

    compact = pd.DataFrame(rows)
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    compact.to_csv(args.csv, index=False)
    best_correct_nll = compact["Correct_NLL"].min()
    best_wrong_auprc = compact["Wrong_AUPRC"].max()
    best_aurc = compact["AURC"].min()
    best_selacc50 = compact["SelAcc50"].max()

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Structural ablation averaged over all relevance-prediction settings. C-NLL denotes \(\mathrm{NLL}_{\mathrm{correct}}\). Variants are defined in Section~\ref{sec:ablation_variants}. MRP denotes the main label-wise 1D projection.}",
        r"\label{tab:mrp_ablation_results}",
        r"\setlength{\tabcolsep}{2.7pt}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Variant & C-NLL$\downarrow$ & AUPR-Error$\uparrow$ & AURC$\downarrow$ & SelAcc@50$\uparrow$ \\",
        r"\midrule",
    ]
    for _, row in compact.iterrows():
        lines.append(
            " & ".join(
                [
                    row["Display"],
                    best_text(row["Correct_NLL"], best_correct_nll),
                    best_text(row["Wrong_AUPRC"], best_wrong_auprc),
                    best_text(row["AURC"], best_aurc),
                    best_text(row["SelAcc50"], best_selacc50),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])

    tex_path = Path(args.tex)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(args.csv)
    print(args.tex)


if __name__ == "__main__":
    main()
