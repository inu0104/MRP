#!/usr/bin/env python3
"""Structural diagnostics for label-wise monotone reliability projection.

This script checks the assumptions behind q = T_k(c):

1. Label-conditional residual structure:
   within confidence bands, do predicted labels have different residuals?
2. Learned reliability curves:
   what does T_k(c) actually do compared with the identity q=c?
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import colormaps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.table.evaluate_monotone_reliability_projection import (  # noqa: E402
    EPS,
    bern_kl,
    clip01,
    eval_q,
    fit_lattice,
    predict_lattice,
    top_runner_gap,
)
from experiments.table.mrp_final_utils import (  # noqa: E402
    BASE_LABELS,
    BASES,
    cap_indices,
    dataset_config,
    fit_calibrators,
    load_meta_args,
    split_indices,
    subset_logits,
    top_info,
)
from experiments.reproducibility import seed_everything  # noqa: E402


LABEL_DISPLAY = {
    "Amazon ESCI": {0: "Irrel.", 1: "Comp.", 2: "Sub.", 3: "Exact"},
    "MSLR-WEB10K": {0: "Bad", 1: "Fair", 2: "Good", 3: "Excellent", 4: "Perfect"},
    "WANDS": {0: "Irrel.", 1: "Partial", 2: "Exact"},
    "ESCI-Rerank-US": {0: "Non-rel.", 1: "Rel."},
    "SciDocs": {0: "Non-rel.", 1: "Rel."},
    "Alloprof-Rerank": {0: "rel=0", 1: "rel=1"},
}


def relevance_color(label: int, n_classes: int):
    if n_classes <= 1:
        return colormaps["viridis"](0.75)
    # Light-to-dark relevance-grade ramp.  The lower end is kept away from
    # white so all curves remain visible in print.
    value = 0.25 + 0.65 * (int(label) / max(1, int(n_classes) - 1))
    return colormaps["viridis"](value)


plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    }
)


def stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["esci_reranking_us", "mslr", "esci", "scidocs", "wands", "alloprof"])
    parser.add_argument("--bases", nargs="+", default=["raw", "ts", "diag", "spline", "hcal", "smart"], choices=BASES)
    parser.add_argument("--max-seeds", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cal-fraction", type=float, default=0.5)
    parser.add_argument("--projection-fit-fraction", type=float, default=0.5)
    parser.add_argument("--projection-fit-cap", type=int, default=8000)
    parser.add_argument("--projection-selection-cap", type=int, default=8000)
    parser.add_argument("--n-conf-bins", type=int, default=10)
    parser.add_argument("--min-cell", type=int, default=30)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--c-knots", type=int, default=8)
    parser.add_argument("--g-knots", type=int, default=6)
    parser.add_argument("--curve-datasets", nargs="+", default=["esci", "mslr", "wands"])
    parser.add_argument("--curve-base", default="smart", choices=BASES)
    parser.add_argument("--anchor-dataset", default="mslr")
    parser.add_argument("--anchor-base", default="smart", choices=BASES)
    parser.add_argument(
        "--anchor-lambdas",
        type=float,
        nargs="+",
        default=[0.0, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
    )
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--smooth", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default="results/analysis/mrp_structure")
    parser.add_argument("--paper-figure-dir", default="paper/figure")
    parser.add_argument("--paper-table-dir", default="paper/table")
    parser.add_argument("--make-anchor-path", action="store_true", help="Also generate the optional anchor-path diagnostic.")
    return parser.parse_args()


def confidence_bins(conf: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.quantile(conf, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    edges = np.maximum.accumulate(edges)
    # Degenerate duplicate edges are okay for digitize; empty bins are skipped.
    return np.digitize(conf, edges[1:-1], right=True)


def label_gap_stat(conf: np.ndarray, top: np.ndarray, corr: np.ndarray, *, n_bins: int, min_cell: int) -> tuple[float, float, int]:
    bins = confidence_bins(conf, n_bins)
    weighted_gap = 0.0
    weighted_var = 0.0
    total_weight = 0
    used_bins = 0
    residual = corr - conf
    for b in range(n_bins):
        in_bin = bins == b
        if int(in_bin.sum()) < min_cell * 2:
            continue
        vals = []
        counts = []
        for label in np.unique(top[in_bin]):
            mask = in_bin & (top == label)
            count = int(mask.sum())
            if count >= min_cell:
                vals.append(float(residual[mask].mean()))
                counts.append(count)
        if len(vals) < 2:
            continue
        vals_arr = np.asarray(vals, dtype=np.float64)
        counts_arr = np.asarray(counts, dtype=np.float64)
        w = int(counts_arr.sum())
        weighted_gap += w * float(vals_arr.max() - vals_arr.min())
        mean = float(np.average(vals_arr, weights=counts_arr))
        weighted_var += w * float(np.average((vals_arr - mean) ** 2, weights=counts_arr))
        total_weight += w
        used_bins += 1
    if total_weight == 0:
        return 0.0, 0.0, 0
    return weighted_gap / total_weight, weighted_var / total_weight, used_bins


def permutation_test(conf, top, corr, *, n_bins, min_cell, permutations, seed):
    obs_gap, obs_var, used_bins = label_gap_stat(conf, top, corr, n_bins=n_bins, min_cell=min_cell)
    rng = np.random.default_rng(seed)
    perm_gap = np.zeros(permutations, dtype=np.float64)
    perm_var = np.zeros(permutations, dtype=np.float64)
    bins = confidence_bins(conf, n_bins)
    for p in range(permutations):
        shuffled = top.copy()
        for b in range(n_bins):
            idx = np.flatnonzero(bins == b)
            if len(idx) > 1:
                shuffled[idx] = rng.permutation(shuffled[idx])
        perm_gap[p], perm_var[p], _ = label_gap_stat(conf, shuffled, corr, n_bins=n_bins, min_cell=min_cell)
    p_gap = (1.0 + float(np.sum(perm_gap >= obs_gap))) / (permutations + 1.0)
    p_var = (1.0 + float(np.sum(perm_var >= obs_var))) / (permutations + 1.0)
    return {
        "ObsGap": obs_gap,
        "PermGapMean": float(perm_gap.mean()),
        "PermGapP95": float(np.quantile(perm_gap, 0.95)),
        "GapPValue": p_gap,
        "ObsVar": obs_var,
        "PermVarMean": float(perm_var.mean()),
        "PermVarP95": float(np.quantile(perm_var, 0.95)),
        "VarPValue": p_var,
        "UsedBins": used_bins,
    }


def load_predictions(dataset_key: str, base: str, run_idx: int, args: argparse.Namespace):
    seed_everything(args.seed + run_idx * 1009)
    config = dataset_config(dataset_key)
    run_dirs = sorted(Path(".").glob(config["glob"]))[: args.max_seeds]
    run_dir = run_dirs[run_idx]
    data = np.load(run_dir / "logits_and_labels.npz")
    meta = load_meta_args(run_dir, args.bases, args.device)
    meta.methods = [m for m in BASES if m in set(args.bases)]
    val_logits = data["val_logits"]
    val_labels = data["val_labels"]
    test_logits = data[config["test_logits_key"]]
    test_labels = data[config["test_labels_key"]]
    temperatures = np.asarray(data["temperatures"], dtype=np.float64)
    cal_idx, fit_idx, val_idx = split_indices(len(val_labels), args.cal_fraction, args.projection_fit_fraction, args.seed + run_idx * 1009)
    fit_idx = cap_indices(fit_idx, args.projection_fit_cap, args.seed + run_idx * 2029 + 11)
    val_idx = cap_indices(val_idx, args.projection_selection_cap, args.seed + run_idx * 2029 + 23)
    calibrators = fit_calibrators(config, subset_logits(val_logits, cal_idx), val_labels[cal_idx], temperatures, meta)
    _logits, fit_probs, _members = config["view"](base, subset_logits(val_logits, fit_idx), temperatures, calibrators)
    _logits, selection_probs, _members = config["view"](base, subset_logits(val_logits, val_idx), temperatures, calibrators)
    _logits, test_probs, _members = config["view"](base, test_logits, temperatures, calibrators)
    return {
        "config": config,
        "run_dir": run_dir,
        "fit_info": top_info(fit_probs, val_labels[fit_idx]),
        "selection_info": top_info(selection_probs, val_labels[val_idx]),
        "test_info": top_info(test_probs, test_labels),
        "fit_gap": top_runner_gap(fit_probs),
        "selection_gap": top_runner_gap(selection_probs),
        "test_gap": top_runner_gap(test_probs),
        "n_classes": int(test_probs.shape[1]),
    }


def run_label_conditional_tests(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    rows = []
    for dataset_key in args.datasets:
        config = dataset_config(dataset_key)
        run_dirs = sorted(Path(".").glob(config["glob"]))[: args.max_seeds]
        for run_idx in range(len(run_dirs)):
            for base in args.bases:
                pred = load_predictions(dataset_key, base, run_idx, args)
                info = pred["test_info"]
                print(f"[label-test] {config['dataset']} run={run_idx} base={BASE_LABELS[base]}", flush=True)
                stat = permutation_test(
                    clip01(info.confidence),
                    info.top_label,
                    info.correctness,
                    n_bins=args.n_conf_bins,
                    min_cell=args.min_cell,
                    permutations=args.permutations,
                    seed=args.seed + 7919 * run_idx + stable_seed(dataset_key, base) % 10000,
                )
                rows.append(
                    {
                        "Dataset": config["dataset"],
                        "Run": run_idx,
                        "Base": BASE_LABELS[base],
                        **stat,
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "label_conditional_residual_tests.csv", index=False)
    summary_rows = []
    for keys, group in df.groupby(["Dataset", "Base"]):
        out = {"Dataset": keys[0], "Base": keys[1]}
        for col in ["ObsGap", "PermGapP95", "GapPValue", "ObsVar", "PermVarP95", "VarPValue"]:
            out[f"{col}_mean"] = float(group[col].mean())
            out[f"{col}_std"] = float(group[col].std(ddof=1)) if len(group) > 1 else 0.0
        out["SignificantGapRuns"] = int((group["GapPValue"] <= 0.05).sum())
        out["Runs"] = int(len(group))
        summary_rows.append(out)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "label_conditional_residual_summary.csv", index=False)
    return summary


def fit_label1d(pred, args: argparse.Namespace, lambdas: list[float], device: torch.device):
    return fit_lattice(
        pred["fit_info"],
        pred["fit_gap"],
        pred["selection_info"],
        pred["selection_gap"],
        variant="Label1D",
        labelwise=True,
        use_gap=False,
        n_classes=pred["n_classes"],
        c_knots=args.c_knots,
        g_knots=args.g_knots,
        lambda_grid=lambdas,
        smooth=args.smooth,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )


def curve_values(fit, n_classes: int, device: torch.device):
    grid = np.linspace(0.02, 0.98, 100, dtype=np.float64)
    all_rows = []
    for label in range(n_classes):
        dummy_info = type("Dummy", (), {})()
        dummy_info.confidence = grid
        dummy_info.top_label = np.full_like(grid, label, dtype=np.int64)
        q = predict_lattice(fit, dummy_info, np.zeros_like(grid), labelwise=True, device=device)
        for c, val in zip(grid, q):
            all_rows.append({"Label": label, "Confidence": float(c), "Reliability": float(val)})
    return pd.DataFrame(all_rows)


def resolve_device(requested: str) -> torch.device:
    requested = str(requested)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for MRP analysis, but torch.cuda.is_available() is False.")
    return torch.device(requested)


def make_curve_figures(args: argparse.Namespace, out_dir: Path, fig_dir: Path) -> pd.DataFrame:
    device = resolve_device(args.device)
    frames = []
    selected = args.curve_datasets
    fig, axes = plt.subplots(1, len(selected), figsize=(4.05 * len(selected), 3.1), squeeze=False)
    axes = axes[0]
    for ax, dataset_key in zip(axes, selected):
        pred = load_predictions(dataset_key, args.curve_base, 0, args)
        fit = fit_label1d(pred, args, [0.0], device)
        curves = curve_values(fit, pred["n_classes"], device)
        dataset_name = pred["config"]["dataset"]
        curves["Dataset"] = dataset_name
        curves["Base"] = BASE_LABELS[args.curve_base]
        curves["Lambda"] = fit.lambda_anchor
        frames.append(curves)
        label_names = LABEL_DISPLAY.get(dataset_name, {})
        for label, group in curves.groupby("Label"):
            label = int(label)
            ax.plot(
                group["Confidence"],
                group["Reliability"],
                linewidth=1.8,
                color=relevance_color(label, pred["n_classes"]),
                label=label_names.get(label, f"label {label}"),
            )
        ax.plot([0, 1], [0, 1], color="black", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_title(dataset_name)
        ax.set_xlabel("calibrated confidence $c$")
        if ax is axes[0]:
            ax.set_ylabel("projected reliability $T_k(c)$")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.legend(frameon=False, fontsize=8, loc="best", title="relevance", title_fontsize=8)
    fig.tight_layout()
    fig_path = fig_dir / "mrp_labelwise_curves.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(out_dir / "labelwise_curve_values.csv", index=False)
    return out


def make_anchor_path(args: argparse.Namespace, out_dir: Path, fig_dir: Path) -> pd.DataFrame:
    device = resolve_device(args.device)
    rows = []
    for run_idx in range(args.max_seeds):
        pred = load_predictions(args.anchor_dataset, args.anchor_base, run_idx, args)
        info = pred["test_info"]
        for lam in args.anchor_lambdas:
            fit = fit_label1d(pred, args, [float(lam)], device)
            q = predict_lattice(fit, info, pred["test_gap"], labelwise=True, device=device)
            metrics = eval_q(info.correctness, q)
            q_t = torch.tensor(clip01(q), dtype=torch.float32, device=device)
            c_t = torch.tensor(clip01(info.confidence), dtype=torch.float32, device=device)
            deviation = float(bern_kl(q_t, c_t).mean().detach().cpu().numpy())
            rows.append(
                {
                    "Dataset": pred["config"]["dataset"],
                    "Run": run_idx,
                    "Base": BASE_LABELS[args.anchor_base],
                    "Lambda": float(lam),
                    "BernKLToConfidence": deviation,
                    **metrics,
                }
            )
    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "anchor_path_rows.csv", index=False)
    summary = raw.groupby(["Dataset", "Base", "Lambda"], as_index=False).agg(
        Binary_NLL_mean=("Binary_NLL", "mean"),
        Wrong_AUPRC_mean=("Wrong_AUPRC", "mean"),
        AURC_mean=("AURC", "mean"),
        SelAcc80_mean=("SelAcc80", "mean"),
        BernKLToConfidence_mean=("BernKLToConfidence", "mean"),
    )
    summary.to_csv(out_dir / "anchor_path_summary.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(5.0, 3.2))
    plot_df = summary.sort_values("BernKLToConfidence_mean").reset_index(drop=True)
    x = plot_df["BernKLToConfidence_mean"].to_numpy()
    ax1.plot(x, plot_df["Wrong_AUPRC_mean"], marker="o", markersize=4.2, label="AUPR-Error")
    ax1.set_xlabel(r"mean $D_{\rm Bern}(T_k(c)\Vert c)$")
    ax1.set_ylabel("AUPR-Error")
    ax1.grid(alpha=0.25, linewidth=0.5)
    ax2 = ax1.twinx()
    ax2.plot(x, plot_df["Binary_NLL_mean"], marker="s", markersize=4.2, color="#cc5a43", label=r"$\mathrm{NLL}_{\mathrm{correct}}$")
    ax2.set_ylabel(r"$\mathrm{NLL}_{\mathrm{correct}}$")
    if len(plot_df) > 1:
        ax1.annotate(
            "strong anchor\n($T_k(c)\\approx c$)",
            xy=(x[0], plot_df["Wrong_AUPRC_mean"].iloc[0]),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=8,
            ha="left",
        )
        ax1.annotate(
            "flexible projection",
            xy=(x[-1], plot_df["Wrong_AUPRC_mean"].iloc[-1]),
            xytext=(-4, -24),
            textcoords="offset points",
            fontsize=8,
            ha="right",
        )
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best", frameon=False)
    fig.tight_layout()
    fig_path = fig_dir / "mrp_anchor_path.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    return summary


def write_label_residual_tex(summary: pd.DataFrame, path: Path):
    overall = summary.groupby("Dataset", as_index=False).agg(
        ObsGap=("ObsGap_mean", "mean"),
        PermP95=("PermGapP95_mean", "mean"),
        SigRuns=("SignificantGapRuns", "sum"),
        Runs=("Runs", "sum"),
    )
    dataset_order = ["ESCI-Rerank-US", "MSLR-WEB10K", "Amazon ESCI", "SciDocs", "WANDS", "Alloprof-Rerank"]
    overall["Dataset"] = pd.Categorical(overall["Dataset"], categories=dataset_order, ordered=True)
    overall = overall.sort_values("Dataset")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Label-conditioned reliability spread. Within each calibrated-confidence group, we measure how much \(Z-\hat c^{\mathcal{A}}\) changes across predicted labels. A large spread indicates that equal-confidence predictions can have label-dependent correctness reliability. The random column is the 95th percentile after shuffling labels within each group.}",
        r"\label{tab:label_conditional_residual}",
        r"\setlength{\tabcolsep}{3.8pt}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Dataset & Label spread & Random & Ratio & Seeds \\",
        r"\midrule",
    ]
    for _, row in overall.iterrows():
        ratio = row["ObsGap"] / max(row["PermP95"], 1e-12)
        lines.append(
            f"{row['Dataset']} & {100 * row['ObsGap']:.1f}pp & {100 * row['PermP95']:.1f}pp & {ratio:.1f}$\\times$ & {int(row['SigRuns'])}/{int(row['Runs'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    out_dir = Path(args.output_dir)
    fig_dir = Path(args.paper_figure_dir)
    table_dir = Path(args.paper_table_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    summary = run_label_conditional_tests(args, out_dir)
    write_label_residual_tex(summary, table_dir / "label_conditional_residual.tex")
    make_curve_figures(args, out_dir, fig_dir)
    if args.make_anchor_path:
        make_anchor_path(args, out_dir, fig_dir)
    print(out_dir)
    print(fig_dir / "mrp_labelwise_curves.pdf")
    if args.make_anchor_path:
        print(fig_dir / "mrp_anchor_path.pdf")
    print(table_dir / "label_conditional_residual.tex")


if __name__ == "__main__":
    main()
