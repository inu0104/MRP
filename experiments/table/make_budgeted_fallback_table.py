#!/usr/bin/env python3
"""Build the budgeted fallback table from MRP summary results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DATASET_ORDER = [
    "ESCI-Rerank-US",
    "MSLR-WEB10K",
    "Amazon ESCI",
    "SciDocs",
    "WANDS",
    "Alloprof-Rerank",
]

DATASET_TEX = {
    "ESCI-Rerank-US": r"\shortstack{ESCI-Rerank\\US}",
    "MSLR-WEB10K": r"\shortstack{MSLR-\\WEB10K}",
    "Amazon ESCI": r"\shortstack{Amazon\\ESCI}",
    "SciDocs": "SciDocs",
    "WANDS": "WANDS",
    "Alloprof-Rerank": r"\shortstack{Alloprof-\\Rerank}",
}

COVERAGES = [10, 50, 70, 90]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/tables/fixed_decision_mrp_main_no_anchor_gpu0/mrp_summary.csv")
    parser.add_argument("--tex", default="paper/table/budgeted_fallback.tex")
    parser.add_argument("--csv", default="results/tables/fixed_decision_mrp_main_no_anchor_gpu0/budgeted_fallback_deltas.csv")
    return parser.parse_args()


def fmt_delta(value: float) -> str:
    text = f"{float(value):+0.3f}"
    return rf"\textbf{{{text}}}" if value > 0 else text


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary)
    ref = summary[summary["Variant"] == "M0_confidence"].set_index(["Dataset", "Base"])
    mrp = summary[summary["Variant"] == "Label1D"].copy()

    rows = []
    for item in mrp.itertuples(index=False):
        base = ref.loc[(item.Dataset, item.Base)]
        row = {"Dataset": item.Dataset, "Base": item.Base}
        for coverage in COVERAGES:
            row[f"Delta_SelAcc{coverage}"] = getattr(item, f"SelAcc{coverage}_mean") - base[f"SelAcc{coverage}_mean"]
        rows.append(row)
    deltas = pd.DataFrame(rows)
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    deltas.to_csv(args.csv, index=False)

    dataset_rows = []
    for dataset in DATASET_ORDER:
        group = deltas[deltas["Dataset"] == dataset]
        if group.empty:
            continue
        dataset_rows.append(
            {
                "Dataset": dataset,
                **{f"Delta_SelAcc{coverage}": float(group[f"Delta_SelAcc{coverage}"].mean()) for coverage in COVERAGES},
            }
        )
    dataset_table = pd.DataFrame(dataset_rows)
    avg = {f"Delta_SelAcc{coverage}": float(dataset_table[f"Delta_SelAcc{coverage}"].mean()) for coverage in COVERAGES}

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Budgeted fallback simulation. At automatic coverage \(\tau\), each reliability score keeps the lowest-risk \(\tau\) fraction for automatic use and routes the remaining highest-risk predictions to fallback or review. Values compare MRP with the no-projection confidence baseline \(\hat{q}_{\varnothing}^{\mathcal{A}}=\hat{c}^{\mathcal{A}}\). Positive values indicate a better reliability reranking.}",
        r"\label{tab:budgeted_fallback}",
        r"\setlength{\tabcolsep}{3.8pt}",
        r"\begin{tabular}{@{}p{0.31\linewidth}rrrr@{}}",
        r"\toprule",
        r"\multirow{2}{*}{Dataset} & \multicolumn{4}{c}{\(\Delta\mathrm{SelAcc}@\tau\)} \\",
        r"\cmidrule(l){2-5}",
        r" & 10\% & 50\% & 70\% & 90\% \\",
        r"\midrule",
    ]
    for row in dataset_rows:
        lines.append(
            " & ".join(
                [DATASET_TEX[row["Dataset"]]]
                + [fmt_delta(row[f"Delta_SelAcc{coverage}"]) for coverage in COVERAGES]
            )
            + r" \\"
        )
    lines += [
        r"\midrule",
        " & ".join(
            [r"\textbf{Average}"]
            + [fmt_delta(avg[f"Delta_SelAcc{coverage}"]) for coverage in COVERAGES]
        )
        + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    tex_path = Path(args.tex)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(args.csv)
    print(args.tex)


if __name__ == "__main__":
    main()
