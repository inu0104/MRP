#!/usr/bin/env python3
"""Evaluate whether MRP can induce a sample-wise simplex projection.

MRP produces a scalar correctness reliability

    q_i = T_{d_i}(c_i^{\mathcal{A}})

for the fixed decision label d_i.  This script asks whether that scalar can be
realized as a top-label probability by moving the base calibrated probability
vector along a temperature-like path on the simplex.

For a base probability vector p_i and alpha >= 0, the path

    p_i(alpha)_k = p_{i,k}^alpha / sum_j p_{i,j}^alpha

is equivalent to applying temperature scaling to log p_i.  If d_i is the top
class and q_i >= 1/K, there is an alpha that realizes p_i(alpha)_{d_i}=q_i.
If q_i < 1/K, preserving the top decision is impossible inside the simplex, so
we clamp the target to 1/K and report the infeasible rate.
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

from experiments.reproducibility import seed_everything  # noqa: E402
from experiments.table.evaluate_monotone_reliability_projection import (  # noqa: E402
    EPS,
    clip01,
    fit_lattice,
    fixed_decision_info,
    fixed_runner_gap,
    predict_lattice,
    resolve_device,
    select_member_logits,
    select_member_temperatures,
)
from experiments.table.mrp_final_utils import (  # noqa: E402
    BASE_LABELS,
    BASES,
    brier,
    dataset_config,
    ece_from_conf,
    fit_calibrators,
    load_meta_args,
    nll,
    split_indices,
    subset_logits,
    top_info,
)


def fixed_ece(probs: np.ndarray, labels: np.ndarray, raw_info) -> float:
    info = fixed_decision_info(probs, labels, raw_info)
    return ece_from_conf(info.confidence, info.correctness)


def fixed_acc(labels: np.ndarray, raw_info) -> float:
    return float((np.asarray(labels) == raw_info.top_label).mean())


def project_power_path(probs: np.ndarray, fixed_label: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    probs = clip01(probs)
    n, k = probs.shape
    fixed_label = np.asarray(fixed_label, dtype=np.int64)
    q = clip01(q)
    floor = 1.0 / float(k)
    q_target = np.clip(q, floor + 1e-7, 1.0 - 1e-7)
    projected = np.empty_like(probs, dtype=np.float64)
    rows = np.arange(n)
    base_top = np.argmax(probs, axis=1)
    fixed_is_base_top = base_top == fixed_label
    below_floor = q < floor

    logp = np.log(probs)
    alphas = np.zeros(n, dtype=np.float64)
    solved = np.zeros(n, dtype=bool)
    for i in range(n):
        d = fixed_label[i]
        if not fixed_is_base_top[i]:
            # This should be rare for order-preserving calibrators.  Fall back
            # to the original vector because an order-preserving temperature
            # path cannot make a non-top class the fixed top decision.
            projected[i] = probs[i]
            continue

        target = float(q_target[i])
        if target <= floor + 2e-7:
            alpha = 0.0
        else:
            lo, hi = 0.0, 1.0

            def top_prob(alpha_value: float) -> float:
                z = alpha_value * logp[i]
                z = z - z.max()
                p = np.exp(z)
                p = p / p.sum()
                return float(p[d])

            while top_prob(hi) < target and hi < 512.0:
                hi *= 2.0
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if top_prob(mid) < target:
                    lo = mid
                else:
                    hi = mid
            alpha = 0.5 * (lo + hi)
        z = alpha * logp[i]
        z = z - z.max()
        p = np.exp(z)
        projected[i] = p / p.sum()
        alphas[i] = alpha
        solved[i] = True

    achieved = projected[rows, fixed_label]
    stats = {
        "q_below_uniform_rate": float(below_floor.mean()),
        "fixed_not_base_top_rate": float((~fixed_is_base_top).mean()),
        "solved_rate": float(solved.mean()),
        "mean_abs_target_error": float(np.abs(achieved - q_target).mean()),
        "mean_alpha": float(alphas[solved].mean()) if np.any(solved) else float("nan"),
    }
    return projected, stats


def project_top_mass(probs: np.ndarray, fixed_label: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Direct simplex projection with top mass q and original non-top ratios.

    This is not a temperature path.  It is useful as an upper sanity check for
    what happens if MRP's scalar q is forced into a class-probability vector.
    """

    probs = clip01(probs)
    n, k = probs.shape
    fixed_label = np.asarray(fixed_label, dtype=np.int64)
    floor = 1.0 / float(k)
    q_target = np.clip(clip01(q), floor + 1e-7, 1.0 - 1e-7)
    out = np.zeros_like(probs, dtype=np.float64)
    rows = np.arange(n)
    out[rows, fixed_label] = q_target
    non_top_mass = 1.0 - q_target
    for i in range(n):
        mask = np.ones(k, dtype=bool)
        mask[fixed_label[i]] = False
        weights = probs[i, mask]
        weights = weights / weights.sum()
        out[i, mask] = non_top_mass[i] * weights
    return out, {"q_below_uniform_rate": float((clip01(q) < floor).mean())}


def metric_row(dataset: str, run_idx: int, base_label: str, variant: str, probs: np.ndarray, labels: np.ndarray, raw_info, extra: dict[str, float]):
    probs = clip01(probs)
    info = fixed_decision_info(probs, labels, raw_info)
    row = {
        "Dataset": dataset,
        "Run": run_idx,
        "Base": base_label,
        "Variant": variant,
        "FixedAcc": fixed_acc(labels, raw_info),
        "ArgmaxAcc": float((np.argmax(probs, axis=1) == labels).mean()),
        "FixedECE": fixed_ece(probs, labels, raw_info),
        "TopECE": ece_from_conf(np.max(probs, axis=1), (np.argmax(probs, axis=1) == labels).astype(np.float64)),
        "NLL": nll(probs, labels),
        "Brier": brier(probs, labels),
        "MeanFixedConf": float(info.confidence.mean()),
    }
    row.update(extra)
    return row


def run_dataset(name: str, args: argparse.Namespace) -> list[dict[str, object]]:
    config = dataset_config(name)
    run_dirs = sorted(path for path in Path(".").glob(config["glob"]) if path.name.rsplit("seed", 1)[-1].isdigit())
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found for {name}: {config['glob']}")
    if args.model_run_index < 0 or args.model_run_index >= len(run_dirs):
        raise IndexError(f"model_run_index={args.model_run_index} out of range for {name}")
    run_dir = run_dirs[args.model_run_index]
    device = resolve_device(args.device)
    rows: list[dict[str, object]] = []
    requested = set(args.methods)
    for run_idx, calibration_seed in enumerate(args.calibration_seeds):
        seed_everything(args.seed + int(calibration_seed))
        print(f"[simplex-proj] {config['dataset']} seed={calibration_seed} {run_dir}", flush=True)
        data = np.load(run_dir / "logits_and_labels.npz")
        meta = load_meta_args(run_dir, args.methods, args.device)
        meta.methods = [m for m in BASES if m in requested]
        val_logits = select_member_logits(data["val_logits"], args.member_index)
        val_labels = data["val_labels"]
        test_logits = select_member_logits(data[config["test_logits_key"]], args.member_index)
        test_labels = data[config["test_labels_key"]]
        temperatures = select_member_temperatures(data["temperatures"], args.member_index)
        split_seed = args.seed + int(calibration_seed) * 1009
        cal_idx, fit_idx, val_idx = split_indices(len(val_labels), args.cal_fraction, args.projection_fit_fraction, split_seed)
        if len(fit_idx) > args.projection_fit_cap:
            rng = np.random.default_rng(split_seed + 11)
            fit_idx = np.sort(rng.choice(fit_idx, size=args.projection_fit_cap, replace=False))
        if len(val_idx) > args.projection_selection_cap:
            rng = np.random.default_rng(split_seed + 23)
            val_idx = np.sort(rng.choice(val_idx, size=args.projection_selection_cap, replace=False))

        calibrators = fit_calibrators(config, subset_logits(val_logits, cal_idx), val_labels[cal_idx], temperatures, meta)
        _raw_logits_fit, raw_fit_probs, _ = config["view"]("raw", subset_logits(val_logits, fit_idx), temperatures, calibrators)
        _raw_logits_val, raw_val_probs, _ = config["view"]("raw", subset_logits(val_logits, val_idx), temperatures, calibrators)
        _raw_logits_test, raw_test_probs, _ = config["view"]("raw", test_logits, temperatures, calibrators)
        raw_fit_info = top_info(raw_fit_probs, val_labels[fit_idx])
        raw_val_info = top_info(raw_val_probs, val_labels[val_idx])
        raw_test_info = top_info(raw_test_probs, test_labels)

        for base in [m for m in BASES if m in set(meta.methods)]:
            base_label = BASE_LABELS[base]
            _logits, fit_probs, _ = config["view"](base, subset_logits(val_logits, fit_idx), temperatures, calibrators)
            _logits, val_probs, _ = config["view"](base, subset_logits(val_logits, val_idx), temperatures, calibrators)
            _logits, test_probs, _ = config["view"](base, test_logits, temperatures, calibrators)
            n_classes = test_probs.shape[1]
            fit_info = fixed_decision_info(fit_probs, val_labels[fit_idx], raw_fit_info)
            val_info = fixed_decision_info(val_probs, val_labels[val_idx], raw_val_info)
            test_info = fixed_decision_info(test_probs, test_labels, raw_test_info)
            fit_gap = fixed_runner_gap(fit_probs, raw_fit_info)
            val_gap = fixed_runner_gap(val_probs, raw_val_info)
            test_gap = fixed_runner_gap(test_probs, raw_test_info)
            lattice = fit_lattice(
                fit_info,
                fit_gap,
                val_info,
                val_gap,
                variant="Label1D",
                labelwise=True,
                use_gap=False,
                n_classes=n_classes,
                c_knots=args.c_knots,
                g_knots=args.g_knots,
                lambda_grid=args.lambda_anchor,
                smooth=args.smooth,
                epochs=args.epochs,
                lr=args.lr,
                device=device,
            )
            q = predict_lattice(lattice, test_info, test_gap, labelwise=True, device=device)
            rows.append(metric_row(config["dataset"], run_idx, base_label, "Base", test_probs, test_labels, raw_test_info, {"Lambda": lattice.lambda_anchor}))

            power_probs, power_stats = project_power_path(test_probs, raw_test_info.top_label, q)
            power_stats = {f"Power_{key}": value for key, value in power_stats.items()}
            power_stats["Lambda"] = lattice.lambda_anchor
            rows.append(metric_row(config["dataset"], run_idx, base_label, "MRP-PowerTS", power_probs, test_labels, raw_test_info, power_stats))

            mass_probs, mass_stats = project_top_mass(test_probs, raw_test_info.top_label, q)
            mass_stats = {f"Mass_{key}": value for key, value in mass_stats.items()}
            mass_stats["Lambda"] = lattice.lambda_anchor
            rows.append(metric_row(config["dataset"], run_idx, base_label, "MRP-TopMass", mass_probs, test_labels, raw_test_info, mass_stats))
    return rows


def summarize(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        "FixedAcc",
        "ArgmaxAcc",
        "FixedECE",
        "TopECE",
        "NLL",
        "Brier",
        "MeanFixedConf",
        "Power_q_below_uniform_rate",
        "Power_fixed_not_base_top_rate",
        "Power_solved_rate",
        "Power_mean_abs_target_error",
        "Power_mean_alpha",
        "Mass_q_below_uniform_rate",
    ]
    rows = []
    for key, group in raw.groupby(["Dataset", "Base", "Variant"], dropna=False):
        out = dict(zip(["Dataset", "Base", "Variant"], key))
        for metric in metrics:
            if metric not in group:
                continue
            vals = group[metric].dropna().astype(float)
            if len(vals) == 0:
                continue
            out[f"{metric}_mean"] = float(vals.mean())
            out[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(out)
    summary = pd.DataFrame(rows)
    base_ref = summary[summary["Variant"] == "Base"].set_index(["Dataset", "Base"])
    delta_rows = []
    for _, row in summary[summary["Variant"] != "Base"].iterrows():
        ref = base_ref.loc[(row["Dataset"], row["Base"])]
        out = row.to_dict()
        for metric in ["FixedECE", "TopECE", "NLL", "Brier", "ArgmaxAcc"]:
            col = f"{metric}_mean"
            if col in row and col in ref:
                out[f"Delta_{metric}_vs_Base"] = row[col] - ref[col]
        delta_rows.append(out)
    return summary, pd.DataFrame(delta_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["esci_reranking_us", "mslr", "esci", "scidocs", "wands", "alloprof"])
    parser.add_argument("--methods", nargs="+", default=["raw", "ts", "diag", "spline", "hcal", "smart"])
    parser.add_argument("--model-run-index", type=int, default=0)
    parser.add_argument("--member-index", type=int, default=0)
    parser.add_argument("--calibration-seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cal-fraction", type=float, default=0.5)
    parser.add_argument("--projection-fit-fraction", type=float, default=0.5)
    parser.add_argument("--projection-fit-cap", type=int, default=8000)
    parser.add_argument("--projection-selection-cap", type=int, default=8000)
    parser.add_argument("--c-knots", type=int, default=8)
    parser.add_argument("--g-knots", type=int, default=6)
    parser.add_argument("--lambda-anchor", type=float, nargs="+", default=[0.0])
    parser.add_argument("--smooth", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default="results/tables/mrp_simplex_projection_gpu0")
    args = parser.parse_args()
    seed_everything(args.seed)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    rows = []
    for dataset in args.datasets:
        rows.extend(run_dataset(dataset, args))
    raw = pd.DataFrame(rows)
    summary, deltas = summarize(raw)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "simplex_projection_rows.csv", index=False)
    summary.to_csv(out / "simplex_projection_summary.csv", index=False)
    deltas.to_csv(out / "simplex_projection_deltas.csv", index=False)

    overall = summary.groupby("Variant", as_index=False)[
        ["FixedECE_mean", "TopECE_mean", "NLL_mean", "Brier_mean", "ArgmaxAcc_mean"]
    ].mean()
    overall.to_csv(out / "simplex_projection_overall.csv", index=False)
    print(overall.round(5).to_string(index=False))
    print(out / "simplex_projection_summary.csv")
    print(out / "simplex_projection_deltas.csv")


if __name__ == "__main__":
    main()
