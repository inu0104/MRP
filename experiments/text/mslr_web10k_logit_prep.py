#!/usr/bin/env python3
"""MSLR-WEB10K relevance-classification reliability projection pilot.

MSLR-WEB10K is a standard learning-to-rank benchmark. Each example is a
query-document pair represented by ranking features and a graded relevance
label. We treat the graded label as a multiclass relevance-classification
target, then evaluate calibrated relevance decisions with the same fixed-
probability protocol used for ESCI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.text.amazon_esci_logit_prep import (  # noqa: E402
    decision_logits,
    fit_temperature,
)


LABEL_NAMES = {0: "Bad", 1: "Fair", 2: "Good", 3: "Excellent", 4: "Perfect"}


def flatten_split(dataset, max_docs: int, seed: int):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(dataset))
    xs = []
    ys = []
    qids = []
    total = 0
    for row_idx in order:
        row = dataset[int(row_idx)]
        features = np.asarray(row["features"], dtype=np.float32)
        labels = np.asarray(row["labels"], dtype=np.int64)
        query = int(row["query"])
        if features.ndim != 2 or len(features) == 0:
            continue
        remaining = int(max_docs) - total
        if remaining <= 0:
            break
        if len(labels) > remaining:
            doc_idx = rng.choice(len(labels), size=remaining, replace=False)
            features = features[doc_idx]
            labels = labels[doc_idx]
        xs.append(features)
        ys.append(labels)
        qids.append(np.full(len(labels), query, dtype=np.int64))
        total += len(labels)
    if not xs:
        raise RuntimeError("No examples collected from MSLR split")
    x = np.nan_to_num(np.vstack(xs), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    y = np.concatenate(ys).astype(np.int64)
    q = np.concatenate(qids).astype(np.int64)
    return x, y, q


def train_member(seed: int, train_x: np.ndarray, train_y: np.ndarray):
    model = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            alpha=2e-5,
            max_iter=24,
            tol=1e-4,
            random_state=seed,
            class_weight="balanced",
            n_jobs=1,
        ),
    )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_y))
    model.fit(train_x[order], train_y[order])
    return model


def model_logits(model, x: np.ndarray, n_classes: int):
    logits = model.decision_function(x)
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim == 1:
        logits = np.stack([-logits, logits], axis=1)
    if logits.shape[1] != n_classes:
        raise ValueError(f"Expected {n_classes} classes, got {logits.shape}")
    return logits


def label_counts(labels: np.ndarray):
    return {LABEL_NAMES[int(k)]: int(v) for k, v in zip(*np.unique(labels, return_counts=True))}


def run_one(ds, args, run_idx: int):
    out = Path(args.output_root) / f"mslr_web10k_sgd_seed{run_idx}"
    out.mkdir(parents=True, exist_ok=True)
    sample_seed = args.sample_seed + run_idx * 1009
    train_x, train_y, train_q = flatten_split(ds["train"], args.max_train_docs, sample_seed)
    val_x, val_y, val_q = flatten_split(ds["validation"], args.max_val_docs, sample_seed + 1)
    test_x, test_y, test_q = flatten_split(ds["test"], args.max_test_docs, sample_seed + 2)
    n_classes = 5
    split_counts = {"train": int(len(train_y)), "val": int(len(val_y)), "test": int(len(test_y))}
    print(f"[mslr] run={run_idx} counts={split_counts}", flush=True)
    print(
        f"[mslr] run={run_idx} labels={{'train': {label_counts(train_y)}, 'val': {label_counts(val_y)}, 'test': {label_counts(test_y)}}}",
        flush=True,
    )

    member_seeds = [args.member_seed_base + run_idx * 100 + idx for idx in range(args.members)]
    val_logits = []
    test_logits = []
    temperatures = []
    for seed in member_seeds:
        print(f"[mslr] run={run_idx} train member seed={seed}", flush=True)
        model = train_member(seed, train_x, train_y)
        v_logits = model_logits(model, val_x, n_classes)
        t_logits = model_logits(model, test_x, n_classes)
        temp = fit_temperature(v_logits, val_y)
        print(f"[mslr] run={run_idx} seed={seed} T={temp:.4f}", flush=True)
        val_logits.append(v_logits)
        test_logits.append(t_logits)
        temperatures.append(temp)

    val_logits = np.stack(val_logits).astype(np.float32)
    test_logits = np.stack(test_logits).astype(np.float32)
    temperatures = np.asarray(temperatures, dtype=np.float32)
    np.savez_compressed(
        out / "logits_and_labels.npz",
        val_logits=val_logits,
        val_labels=val_y.astype(np.int64),
        val_query_ids=val_q.astype(np.int64),
        test_logits=test_logits,
        test_labels=test_y.astype(np.int64),
        test_query_ids=test_q.astype(np.int64),
        temperatures=temperatures,
    )
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset_name,
                "base_model": "mslr_features_sgd_logistic",
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
                "device": args.device,
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
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--sample-seed", type=int, default=20260620)
    parser.add_argument("--member-seed-base", type=int, default=7101)
    parser.add_argument("--max-train-docs", type=int, default=250000)
    parser.add_argument("--max-val-docs", type=int, default=80000)
    parser.add_argument("--max-test-docs", type=int, default=100000)
    parser.add_argument("--methods", nargs="*", default=["raw", "ts", "hcal", "smart"])
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
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
