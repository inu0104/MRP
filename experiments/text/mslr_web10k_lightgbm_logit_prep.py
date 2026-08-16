#!/usr/bin/env python3
"""LightGBM relevance logits for MSLR-WEB10K.

This provides a stronger tabular relevance predictor than the linear MSLR
baseline while preserving the multiclass relevance-label output expected by the
shared calibration/MRP pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.reproducibility import seed_everything  # noqa: E402
from experiments.text.amazon_esci_logit_prep import fit_temperature  # noqa: E402
from experiments.text.mslr_web10k_logit_prep import LABEL_NAMES, flatten_split, label_counts  # noqa: E402


def train_member(seed: int, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, args):
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=5,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        subsample_freq=1,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        class_weight="balanced" if args.class_weight_balanced else None,
        random_state=seed,
        n_jobs=args.n_jobs,
        verbosity=-1,
    )
    model.fit(
        train_x,
        train_y,
        eval_set=[(val_x, val_y)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False)],
    )
    return model


def model_logits(model, x: np.ndarray, n_classes: int) -> np.ndarray:
    logits = np.asarray(model.predict(x, raw_score=True), dtype=np.float32)
    if logits.ndim != 2 or logits.shape[1] != n_classes:
        raise ValueError(f"Expected logits shape (*, {n_classes}), got {logits.shape}")
    return logits


def run_one(ds, args, run_idx: int):
    seed_everything(args.sample_seed + run_idx * 1009)
    out = Path(args.output_root) / f"{args.output_prefix}_seed{run_idx}"
    out.mkdir(parents=True, exist_ok=True)
    sample_seed = args.sample_seed + run_idx * 1009
    train_x, train_y, train_q = flatten_split(ds["train"], args.max_train_docs, sample_seed)
    val_x, val_y, val_q = flatten_split(ds["validation"], args.max_val_docs, sample_seed + 1)
    test_x, test_y, test_q = flatten_split(ds["test"], args.max_test_docs, sample_seed + 2)
    n_classes = 5
    split_counts = {"train": int(len(train_y)), "val": int(len(val_y)), "test": int(len(test_y))}
    print(f"[mslr-lightgbm] run={run_idx} counts={split_counts}", flush=True)
    print(
        f"[mslr-lightgbm] run={run_idx} labels={{'train': {label_counts(train_y)}, 'val': {label_counts(val_y)}, 'test': {label_counts(test_y)}}}",
        flush=True,
    )

    member_seeds = [args.member_seed_base + run_idx * 100 + idx for idx in range(args.members)]
    val_logits = []
    test_logits = []
    temperatures = []
    best_iterations = []
    for seed in member_seeds:
        print(f"[mslr-lightgbm] run={run_idx} train member seed={seed}", flush=True)
        model = train_member(seed, train_x, train_y, val_x, val_y, args)
        v_logits = model_logits(model, val_x, n_classes)
        t_logits = model_logits(model, test_x, n_classes)
        temp = fit_temperature(v_logits, val_y)
        best_iter = int(getattr(model, "best_iteration_", 0) or args.n_estimators)
        print(f"[mslr-lightgbm] run={run_idx} seed={seed} best_iter={best_iter} T={temp:.4f}", flush=True)
        val_logits.append(v_logits)
        test_logits.append(t_logits)
        temperatures.append(temp)
        best_iterations.append(best_iter)

    np.savez_compressed(
        out / "logits_and_labels.npz",
        val_logits=np.stack(val_logits).astype(np.float32),
        val_labels=val_y.astype(np.int64),
        val_query_ids=val_q.astype(np.int64),
        test_logits=np.stack(test_logits).astype(np.float32),
        test_labels=test_y.astype(np.int64),
        test_query_ids=test_q.astype(np.int64),
        temperatures=np.asarray(temperatures, dtype=np.float32),
    )
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset_name,
                "base_model": "lightgbm_multiclass",
                "run_idx": run_idx,
                "member_seeds": member_seeds,
                "methods": args.methods,
                "n_classes": n_classes,
                "label_names": LABEL_NAMES,
                "split_counts": split_counts,
                "label_counts": {
                    "train": label_counts(train_y),
                    "val": label_counts(val_y),
                    "test": label_counts(test_y),
                },
                "max_train_docs": args.max_train_docs,
                "max_val_docs": args.max_val_docs,
                "max_test_docs": args.max_test_docs,
                "n_estimators": args.n_estimators,
                "best_iterations": best_iterations,
                "learning_rate": args.learning_rate,
                "num_leaves": args.num_leaves,
                "max_depth": args.max_depth,
                "min_child_samples": args.min_child_samples,
                "subsample": args.subsample,
                "colsample_bytree": args.colsample_bytree,
                "reg_alpha": args.reg_alpha,
                "reg_lambda": args.reg_lambda,
                "class_weight_balanced": args.class_weight_balanced,
                "hcal_segments": args.hcal_segments,
                "hcal_epochs": args.hcal_epochs,
                "hcal_patience": args.hcal_patience,
                "hcal_lr": args.hcal_lr,
                "hcal_batch_size": args.hcal_batch_size,
                "hcal_window": args.hcal_window,
                "hcal_loss_weight": args.hcal_loss_weight,
                "smart_hidden_dim": args.smart_hidden_dim,
                "smart_layers": args.smart_layers,
                "smart_epochs": args.smart_epochs,
                "smart_patience": args.smart_patience,
                "smart_lr": args.smart_lr,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(out / "logits_and_labels.npz", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="philipphager/MSLR-WEB10k")
    parser.add_argument("--output-root", default=".local/runs/relevance_projection")
    parser.add_argument("--output-prefix", default="mslr_web10k_lightgbm")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--members", type=int, default=1)
    parser.add_argument("--sample-seed", type=int, default=20260620)
    parser.add_argument("--member-seed-base", type=int, default=9101)
    parser.add_argument("--max-train-docs", type=int, default=250000)
    parser.add_argument("--max-val-docs", type=int, default=80000)
    parser.add_argument("--max-test-docs", type=int, default=100000)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-child-samples", type=int, default=80)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--class-weight-balanced", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--methods", nargs="*", default=["raw", "ts", "diag", "spline", "hcal", "smart"])
    parser.add_argument("--hcal-segments", type=int, default=50)
    parser.add_argument("--hcal-epochs", type=int, default=80)
    parser.add_argument("--hcal-patience", type=int, default=18)
    parser.add_argument("--hcal-lr", type=float, default=0.005)
    parser.add_argument("--hcal-batch-size", type=int, default=6000)
    parser.add_argument("--hcal-window", type=int, default=200)
    parser.add_argument("--hcal-loss-weight", type=float, default=1e5)
    parser.add_argument("--smart-hidden-dim", type=int, default=16)
    parser.add_argument("--smart-layers", type=int, default=2)
    parser.add_argument("--smart-epochs", type=int, default=120)
    parser.add_argument("--smart-patience", type=int, default=28)
    parser.add_argument("--smart-lr", type=float, default=0.005)
    args = parser.parse_args()

    ds = load_dataset(args.dataset_name)
    for run_idx in range(args.runs):
        run_one(ds, args, run_idx)


if __name__ == "__main__":
    main()
