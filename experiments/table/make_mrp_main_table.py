#!/usr/bin/env python3
"""Build the main table for monotone reliability projection.

The main table has one row per dataset/calibrator.  Accuracy is computed from a
fixed raw top-label decision produced by a single saved model member.  Each
calibrator only changes the confidence assigned to that fixed decision.  MRP
then changes neither the decision nor the calibrated confidence; it only
replaces the confidence reliability baseline q=c with q=T_k(c).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.table.evaluate_monotone_reliability_projection import (  # noqa: E402
    fixed_decision_info,
    select_member_logits,
    select_member_temperatures,
)
from experiments.table.mrp_final_utils import (  # noqa: E402
    BASE_LABELS,
    BASES,
    dataset_config,
    ece_from_conf,
    fit_calibrators,
    load_meta_args,
    split_indices,
    subset_logits,
    top_info,
)
from experiments.reproducibility import seed_everything  # noqa: E402


DATASETS = [
    ("esci_reranking_us", "ESCI-Rerank-US"),
    ("mslr", "MSLR-WEB10K"),
    ("esci", "Amazon ESCI"),
    ("scidocs", "SciDocs"),
    ("wands", "WANDS"),
    ("alloprof", "Alloprof-Rerank"),
]

DEFAULT_BASES = ["raw", "ts", "diag", "spline", "hcal", "smart"]
BASE_TO_LABEL = {key: BASE_LABELS[key] for key in BASES}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/tables/monotone_reliability_projection_main_gpu0/mrp_summary.csv")
    parser.add_argument("--datasets", nargs="+", default=[item[0] for item in DATASETS])
    parser.add_argument("--bases", nargs="+", default=DEFAULT_BASES, choices=BASES)
    parser.add_argument("--model-run-index", type=int, default=0)
    parser.add_argument("--member-index", type=int, default=0)
    parser.add_argument("--calibration-seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--max-seeds", type=int, default=3, help="Deprecated; kept only for old command compatibility.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/tables/mrp_main_table")
    parser.add_argument("--tex", default="paper/table/main_mrp_table.tex")
    parser.add_argument("--refresh-full-metrics", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def require_device(requested: str) -> None:
    requested = str(requested)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for main-table generation, but torch.cuda.is_available() is False.")


def selected_datasets(args: argparse.Namespace):
    requested = set(args.datasets)
    valid = {item[0] for item in DATASETS}
    unknown = requested - valid
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    return [item for item in DATASETS if item[0] in requested]


def prediction_metrics(probs: np.ndarray, labels: np.ndarray, raw_info) -> dict[str, float]:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    info = fixed_decision_info(probs, labels, raw_info)
    confidence = info.confidence
    correct = info.correctness.astype(np.float64)
    return {
        "Acc": float(correct.mean()),
        "ECE": ece_from_conf(confidence, correct),
    }


def summarize_full_metrics(full_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = full_rows.groupby(["DatasetKey", "Dataset", "BaseKey", "Base"], as_index=False)
    for keys, group in grouped:
        dataset_key, dataset, base_key, base = keys
        row = {
            "DatasetKey": dataset_key,
            "Dataset": dataset,
            "BaseKey": base_key,
            "Base": base,
        }
        for metric in ["Acc", "ECE"]:
            vals = group[metric].astype(float)
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def load_full_metrics(args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    cache_path = output_dir / "full_metrics.csv"
    summary_path = output_dir / "full_metrics_summary.csv"
    if cache_path.exists() and summary_path.exists() and not args.refresh_full_metrics:
        return pd.read_csv(summary_path)

    rows: list[dict[str, object]] = []
    requested = set(args.bases)
    for dataset_key, dataset_label in selected_datasets(args):
        config = dataset_config(dataset_key)
        run_dirs = sorted(path for path in Path(".").glob(config["glob"]) if path.name.rsplit("seed", 1)[-1].isdigit())
        if not run_dirs:
            raise FileNotFoundError(f"No run directories found for {dataset_key}: {config['glob']}")
        if args.model_run_index < 0 or args.model_run_index >= len(run_dirs):
            raise IndexError(
                f"model_run_index={args.model_run_index} is out of range for {dataset_key}; found {len(run_dirs)} run dirs"
            )
        run_dir = run_dirs[args.model_run_index]

        for run_idx, calibration_seed in enumerate(args.calibration_seeds):
            seed_everything(args.seed + int(calibration_seed))
            print(
                f"[full] {dataset_label} model_run={args.model_run_index} member={args.member_index} "
                f"cal_seed={calibration_seed} {run_dir}",
                flush=True,
            )
            data = np.load(run_dir / "logits_and_labels.npz")
            meta = load_meta_args(run_dir, list(requested), args.device)
            meta.methods = [method for method in BASES if method in requested]

            val_logits = select_member_logits(data["val_logits"], args.member_index)
            val_labels = data["val_labels"]
            test_logits = select_member_logits(data[config["test_logits_key"]], args.member_index)
            test_labels = data[config["test_labels_key"]]
            temperatures = select_member_temperatures(data["temperatures"], args.member_index)
            split_seed = args.seed + int(calibration_seed) * 1009
            cal_idx, _fit_idx, _val_idx = split_indices(len(val_labels), 0.5, 0.5, split_seed)
            calibrators = fit_calibrators(config, subset_logits(val_logits, cal_idx), val_labels[cal_idx], temperatures, meta)
            _raw_logits, raw_probs, _raw_members = config["view"]("raw", test_logits, temperatures, calibrators)
            raw_info = top_info(raw_probs, test_labels)

            for base in [method for method in BASES if method in requested]:
                _logits, probs, _members = config["view"](base, test_logits, temperatures, calibrators)
                rows.append(
                    {
                        "DatasetKey": dataset_key,
                        "Dataset": dataset_label,
                        "Run": run_idx,
                        "CalibrationSeed": calibration_seed,
                        "BaseKey": base,
                        "Base": BASE_TO_LABEL[base],
                        **prediction_metrics(probs, test_labels, raw_info),
                    }
                )

    full = pd.DataFrame(rows)
    full.to_csv(cache_path, index=False)
    summary = summarize_full_metrics(full)
    summary.to_csv(summary_path, index=False)
    return summary


def pm(mean: float, std: float, digits: int = 3) -> str:
    return f"{float(mean):.{digits}f}{{\\tiny $\\pm$ {float(std):.{digits}f}}}"


def fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def delta(value: float, *, good_when: str, digits: int = 3) -> str:
    good = value > 0 if good_when == "positive" else value < 0
    text = f"{float(value):+.{digits}f}"
    return f"\\textbf{{{text}}}" if good else text


def dataset_cell(label: str) -> str:
    replacements = {
        "ESCI-Rerank-US": "\\shortstack{ESCI-Rerank\\\\US}",
        "MSLR-WEB10K": "\\shortstack{MSLR-\\\\WEB10K}",
        "Amazon ESCI": "\\shortstack{Amazon\\\\ESCI}",
    }
    return replacements.get(label, label)


def build_rows(args: argparse.Namespace, full: pd.DataFrame, mrp: pd.DataFrame) -> pd.DataFrame:
    full_lookup = {
        (str(row["Dataset"]), str(row["Base"])): row
        for _, row in full.iterrows()
    }
    mrp_lookup = {
        (str(row["Dataset"]), str(row["Base"]), str(row["Variant"])): row
        for _, row in mrp.iterrows()
    }
    rows = []
    for _dataset_key, dataset_label in selected_datasets(args):
        for base_key in args.bases:
            base_label = BASE_TO_LABEL[base_key]
            full_row = full_lookup.get((dataset_label, base_label))
            m0 = mrp_lookup.get((dataset_label, base_label, "M0_confidence"))
            mrp_row = mrp_lookup.get((dataset_label, base_label, "Label1D"))
            if full_row is None or m0 is None or mrp_row is None:
                print(f"[warn] missing dataset={dataset_label} base={base_label}", flush=True)
                continue
            out = {
                "Dataset": dataset_label,
                "Base": base_label,
                "Acc_mean": full_row["Acc_mean"],
                "Acc_std": full_row["Acc_std"],
                "ECE_mean": full_row["ECE_mean"],
                "ECE_std": full_row["ECE_std"],
            }
            for metric in ["Binary_NLL", "Wrong_AUPRC", "AURC"]:
                out[f"M0_{metric}_mean"] = m0[f"{metric}_mean"]
                out[f"M0_{metric}_std"] = m0[f"{metric}_std"]
                out[f"MRP_{metric}_mean"] = mrp_row[f"{metric}_mean"]
                out[f"MRP_{metric}_std"] = mrp_row[f"{metric}_std"]
                out[f"Delta_{metric}"] = mrp_row[f"{metric}_mean"] - m0[f"{metric}_mean"]
            rows.append(out)
    return pd.DataFrame(rows)


def write_tex(table: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Post-calibration reliability reranking results under the fixed-decision protocol. Acc. and ECE are base-calibrator values and are unchanged by MRP. Reliability metrics compare the confidence baseline \(\\hat q=\\hat c^{\mathcal{A}}\) with MRP. Blue bold \(\\Delta\) values indicate improvements in the direction of each metric.}",
        "\\label{tab:main_mrp}",
        "\\setlength{\\tabcolsep}{3.2pt}",
        "\\begin{tabular}{llcc|ccc|ccc|ccc}",
        "\\toprule",
        "\\multirow{2}{*}{Dataset} & \\multirow{2}{*}{Calibrator} & \\multirow{2}{*}{Acc.} & \\multirow{2}{*}{ECE} & \\multicolumn{3}{c|}{$\\mathrm{NLL}_{\\mathrm{correct}}\\downarrow$} & \\multicolumn{3}{c|}{AUPR-Error$\\uparrow$} & \\multicolumn{3}{c}{AURC$\\downarrow$} \\\\",
        " & & & & Conf. & MRP & $\\Delta$ & Conf. & MRP & $\\Delta$ & Conf. & MRP & $\\Delta$ \\\\",
        "\\midrule",
    ]
    for dataset_idx, (dataset, group) in enumerate(table.groupby("Dataset", sort=False)):
        if dataset_idx:
            lines.append("\\midrule")
        dataset_text = f"\\multirow{{{len(group)}}}{{*}}{{{dataset_cell(dataset)}}}"
        for row_idx, (_, row) in enumerate(group.iterrows()):
            dataset_cell_text = dataset_text if row_idx == 0 else ""
            base_label = str(row["Base"])
            acc_text = fmt(row["Acc_mean"])
            ece_text = fmt(row["ECE_mean"])
            lines.append(
                " & ".join(
                    [
                        dataset_cell_text,
                        base_label,
                        acc_text,
                        ece_text,
                        pm(row["M0_Binary_NLL_mean"], row["M0_Binary_NLL_std"]),
                        pm(row["MRP_Binary_NLL_mean"], row["MRP_Binary_NLL_std"]),
                        delta(row["Delta_Binary_NLL"], good_when="negative"),
                        pm(row["M0_Wrong_AUPRC_mean"], row["M0_Wrong_AUPRC_std"]),
                        pm(row["MRP_Wrong_AUPRC_mean"], row["MRP_Wrong_AUPRC_std"]),
                        delta(row["Delta_Wrong_AUPRC"], good_when="positive"),
                        pm(row["M0_AURC_mean"], row["M0_AURC_std"]),
                        pm(row["MRP_AURC_mean"], row["MRP_AURC_std"]),
                        delta(row["Delta_AURC"], good_when="negative"),
                    ]
                )
                + " \\\\"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    require_device(args.device)
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mrp = pd.read_csv(args.summary)
    full = load_full_metrics(args, output_dir)
    table = build_rows(args, full, mrp)
    table.to_csv(output_dir / "main_mrp_table.csv", index=False)
    write_tex(table, Path(args.tex))
    print(output_dir / "main_mrp_table.csv")
    print(args.tex)


if __name__ == "__main__":
    main()
