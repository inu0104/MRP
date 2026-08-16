#!/usr/bin/env python3
"""WANDS product-search relevance reliability projection pilot.

WANDS provides query-product pairs with three relevance labels:
Irrelevant, Partial, and Exact. This script trains a lightweight TF-IDF + SGD
ensemble, stores validation/test logits, and reuses the common calibrator and
reliability projection evaluation code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.text.amazon_esci_logit_prep import (  # noqa: E402
    decision_logits,
    fit_temperature,
    train_member,
)


LABEL_NAMES = {0: "Irrelevant", 1: "Partial", 2: "Exact"}


def make_text(row) -> str:
    query = str(row.get("query") or "").strip()
    name = str(row.get("product_name") or "").strip()
    product_class = str(row.get("product_class") or "").strip()
    category = str(row.get("category hierarchy") or "").strip()
    description = str(row.get("product_description") or "").strip()
    features = str(row.get("product_features") or "").strip()
    return (
        f"query: {query}\n"
        f"product_name: {name}\n"
        f"product_class: {product_class}\n"
        f"category: {category}\n"
        f"description: {description}\n"
        f"features: {features}"
    ).strip()


def sample_split(dataset, max_examples: int | None, seed: int):
    n = len(dataset) if max_examples is None else min(int(max_examples), len(dataset))
    sampled = dataset.shuffle(seed=seed).select(range(n))
    texts = [make_text(row) for row in sampled]
    labels = np.asarray(sampled["label"], dtype=np.int64)
    query_ids = np.asarray(sampled["query_id"], dtype=np.int64)
    return texts, labels, query_ids


def label_counts(labels: np.ndarray):
    return {LABEL_NAMES[int(k)]: int(v) for k, v in zip(*np.unique(labels, return_counts=True))}


def run_one(ds, args, run_idx: int):
    out = Path(args.output_root) / f"wands_sgd_seed{run_idx}"
    out.mkdir(parents=True, exist_ok=True)
    sample_seed = args.sample_seed + run_idx * 1009
    train_texts, train_labels, train_qids = sample_split(ds["train"], args.max_train, sample_seed)
    val_texts, val_labels, val_qids = sample_split(ds["dev"], args.max_val, sample_seed + 1)
    test_texts, test_labels, test_qids = sample_split(ds["test"], args.max_test, sample_seed + 2)
    n_classes = 3
    split_counts = {
        "train": int(len(train_labels)),
        "val": int(len(val_labels)),
        "test": int(len(test_labels)),
    }
    print(f"[wands] run={run_idx} counts={split_counts}", flush=True)
    print(
        f"[wands] run={run_idx} labels={{'train': {label_counts(train_labels)}, 'val': {label_counts(val_labels)}, 'test': {label_counts(test_labels)}}}",
        flush=True,
    )

    member_seeds = [args.member_seed_base + run_idx * 100 + idx for idx in range(args.members)]
    val_logits = []
    test_logits = []
    temperatures = []
    for seed in member_seeds:
        print(f"[wands] run={run_idx} train member seed={seed}", flush=True)
        model = train_member(seed, train_texts, train_labels, args.max_features)
        v_logits = decision_logits(model, val_texts, n_classes)
        t_logits = decision_logits(model, test_texts, n_classes)
        temp = fit_temperature(v_logits, val_labels)
        print(f"[wands] run={run_idx} seed={seed} T={temp:.4f}", flush=True)
        val_logits.append(v_logits)
        test_logits.append(t_logits)
        temperatures.append(temp)

    val_logits = np.stack(val_logits).astype(np.float32)
    test_logits = np.stack(test_logits).astype(np.float32)
    temperatures = np.asarray(temperatures, dtype=np.float32)
    np.savez_compressed(
        out / "logits_and_labels.npz",
        val_logits=val_logits,
        val_labels=val_labels.astype(np.int64),
        val_query_ids=val_qids.astype(np.int64),
        test_logits=test_logits,
        test_labels=test_labels.astype(np.int64),
        test_query_ids=test_qids.astype(np.int64),
        temperatures=temperatures,
    )
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset_name,
                "base_model": "tfidf_sgd_logistic",
                "run_idx": run_idx,
                "member_seeds": member_seeds,
                "methods": args.methods,
                "n_classes": n_classes,
                "label_names": LABEL_NAMES,
                "split_counts": split_counts,
                "label_counts": {
                    "train": label_counts(train_labels),
                    "val": label_counts(val_labels),
                    "test": label_counts(test_labels),
                },
                "query_counts": {
                    "train": int(len(set(train_qids.tolist()))),
                    "val": int(len(set(val_qids.tolist()))),
                    "test": int(len(set(test_qids.tolist()))),
                },
                "max_features": args.max_features,
                "device": args.device,
                "hcal_segments": args.hcal_segments,
                "hcal_epochs": args.hcal_epochs,
                "hcal_patience": args.hcal_patience,
                "hcal_lr": args.hcal_lr,
                "hcal_batch_size": args.hcal_batch_size,
                "hcal_window": args.hcal_window,
                "hcal_loss_weight": args.hcal_loss_weight,
                "dirichlet_max_iter": args.dirichlet_max_iter,
                "dirichlet_lr": args.dirichlet_lr,
                "dirichlet_weight_decay": args.dirichlet_weight_decay,
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
    parser.add_argument("--dataset-name", default="napsternxg/wands")
    parser.add_argument("--output-root", default=".local/runs/relevance_projection")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--sample-seed", type=int, default=20260623)
    parser.add_argument("--member-seed-base", type=int, default=10101)
    parser.add_argument("--max-train", type=int, default=90000)
    parser.add_argument("--max-val", type=int, default=25000)
    parser.add_argument("--max-test", type=int, default=40000)
    parser.add_argument("--max-features", type=int, default=100000)
    parser.add_argument("--methods", nargs="*", default=["raw", "ts", "isotonic", "hcal", "dirichlet", "smart"])
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--hcal-segments", type=int, default=50)
    parser.add_argument("--hcal-epochs", type=int, default=80)
    parser.add_argument("--hcal-patience", type=int, default=18)
    parser.add_argument("--hcal-lr", type=float, default=0.005)
    parser.add_argument("--hcal-batch-size", type=int, default=6000)
    parser.add_argument("--hcal-window", type=int, default=200)
    parser.add_argument("--hcal-loss-weight", type=float, default=1e5)
    parser.add_argument("--dirichlet-max-iter", type=int, default=160)
    parser.add_argument("--dirichlet-lr", type=float, default=0.1)
    parser.add_argument("--dirichlet-weight-decay", type=float, default=1e-3)
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
