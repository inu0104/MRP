#!/usr/bin/env python3
"""MTEB qrels-style reranking relevance-classification pilot.

Some newer MTEB reranking/retrieval datasets expose qrels, corpus, and queries
as separate configs instead of per-query positive/negative lists.  This script
joins those pieces into query-document relevance-classification examples,
creates query-heldout splits, and stores ensemble logits for the shared
calibration/MRP evaluators.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.text.amazon_esci_logit_prep import (  # noqa: E402
    decision_logits,
    fit_temperature,
)
from experiments.reproducibility import seed_everything  # noqa: E402


def require_device(requested: str) -> None:
    requested = str(requested)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")


def row_id(row) -> str:
    return str(row.get("_id", row.get("id")))


def row_text(row) -> str:
    title = str(row.get("title") or "").strip()
    text = str(row.get("text") or "").strip()
    if title and text:
        return f"{title}\n{text}"
    return title or text


def load_text_maps(args):
    corpus = load_dataset(args.dataset_name, "corpus", split=args.data_split)
    queries = load_dataset(args.dataset_name, "queries", split=args.data_split)
    docs = {row_id(row): row_text(row) for row in corpus}
    qs = {row_id(row): row_text(row) for row in queries}
    return docs, qs


def label_names(labels: np.ndarray) -> dict[int, str]:
    return {int(label): f"rel={int(label)}" for label in sorted(np.unique(labels).tolist())}


def label_counts(labels: np.ndarray, names: dict[int, str]):
    return {names[int(k)]: int(v) for k, v in zip(*np.unique(labels, return_counts=True))}


def split_query_ids(qrels, seed: int):
    qids = np.asarray(sorted({str(row["query-id"]) for row in qrels}), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(qids)
    n = len(qids)
    n_train = max(1, int(round(n * 0.6)))
    n_val = max(1, int(round(n * 0.2)))
    return set(qids[:n_train]), set(qids[n_train : n_train + n_val]), set(qids[n_train + n_val :])


def build_examples(qrels, docs, queries, qid_set, max_examples: int | None, seed: int):
    rows = []
    for row in qrels:
        qid = str(row["query-id"])
        if qid not in qid_set:
            continue
        doc_id = str(row["corpus-id"])
        if qid not in queries or doc_id not in docs:
            continue
        label = int(row["score"])
        if label < 0:
            continue
        text = f"query: {queries[qid]}\ndocument: {docs[doc_id]}"
        rows.append((text, label, qid))

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    if max_examples is not None:
        order = order[: min(int(max_examples), len(order))]
    texts = [rows[int(i)][0] for i in order]
    labels = np.asarray([rows[int(i)][1] for i in order], dtype=np.int64)
    qids = np.asarray([rows[int(i)][2] for i in order], dtype=object)
    return texts, labels, qids


def train_member(seed: int, train_texts: list[str], train_labels: np.ndarray, args):
    model = make_pipeline(
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=args.min_df,
            max_features=args.max_features,
            sublinear_tf=True,
        ),
        SGDClassifier(
            loss="log_loss",
            alpha=args.sgd_alpha,
            max_iter=args.sgd_max_iter,
            tol=args.sgd_tol,
            random_state=seed,
            class_weight="balanced",
            n_jobs=1,
        ),
    )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_texts))
    model.fit([train_texts[int(i)] for i in order], train_labels[order])
    return model


def run_one(qrels, docs, queries, args, run_idx: int):
    seed_everything(args.sample_seed + run_idx * 1009)
    out = Path(args.output_root) / f"{args.output_prefix}_seed{run_idx}"
    out.mkdir(parents=True, exist_ok=True)
    split_seed = args.sample_seed + run_idx * 1009
    train_q, val_q, test_q = split_query_ids(qrels, split_seed)
    train_texts, train_labels, train_qids = build_examples(qrels, docs, queries, train_q, args.max_train, split_seed + 1)
    val_texts, val_labels, val_qids = build_examples(qrels, docs, queries, val_q, args.max_val, split_seed + 2)
    test_texts, test_labels, test_qids = build_examples(qrels, docs, queries, test_q, args.max_test, split_seed + 3)

    all_labels = np.concatenate([train_labels, val_labels, test_labels])
    names = label_names(all_labels)
    n_classes = int(all_labels.max()) + 1
    split_counts = {"train": int(len(train_labels)), "val": int(len(val_labels)), "test": int(len(test_labels))}
    print(f"[{args.output_prefix}] run={run_idx} counts={split_counts}", flush=True)
    print(
        f"[{args.output_prefix}] run={run_idx} labels={{'train': {label_counts(train_labels, names)}, 'val': {label_counts(val_labels, names)}, 'test': {label_counts(test_labels, names)}}}",
        flush=True,
    )
    if min(split_counts.values()) == 0:
        raise RuntimeError(f"Empty split after joining qrels/corpus/queries: {split_counts}")

    member_seeds = [args.member_seed_base + run_idx * 100 + idx for idx in range(args.members)]
    val_logits = []
    test_logits = []
    temperatures = []
    for seed in member_seeds:
        print(f"[{args.output_prefix}] run={run_idx} train member seed={seed}", flush=True)
        model = train_member(seed, train_texts, train_labels, args)
        v_logits = decision_logits(model, val_texts, n_classes)
        t_logits = decision_logits(model, test_texts, n_classes)
        temp = fit_temperature(v_logits, val_labels)
        print(f"[{args.output_prefix}] run={run_idx} seed={seed} T={temp:.4f}", flush=True)
        val_logits.append(v_logits)
        test_logits.append(t_logits)
        temperatures.append(temp)

    val_logits = np.stack(val_logits).astype(np.float32)
    test_logits = np.stack(test_logits).astype(np.float32)
    temperatures = np.asarray(temperatures, dtype=np.float32)
    qid_to_int = {qid: idx for idx, qid in enumerate(sorted(set(val_qids.tolist()) | set(test_qids.tolist())))}
    np.savez_compressed(
        out / "logits_and_labels.npz",
        val_logits=val_logits,
        val_labels=val_labels.astype(np.int64),
        val_query_ids=np.asarray([qid_to_int[str(qid)] for qid in val_qids], dtype=np.int64),
        test_logits=test_logits,
        test_labels=test_labels.astype(np.int64),
        test_query_ids=np.asarray([qid_to_int[str(qid)] for qid in test_qids], dtype=np.int64),
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
                "label_names": names,
                "split_counts": split_counts,
                "label_counts": {
                    "train": label_counts(train_labels, names),
                    "val": label_counts(val_labels, names),
                    "test": label_counts(test_labels, names),
                },
                "query_counts": {"train": len(train_q), "val": len(val_q), "test": len(test_q)},
                "data_split": args.data_split,
                "max_features": args.max_features,
                "min_df": args.min_df,
                "sgd_alpha": args.sgd_alpha,
                "sgd_max_iter": args.sgd_max_iter,
                "sgd_tol": args.sgd_tol,
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
    parser.add_argument("--dataset-name", default="mteb/AlloprofReranking")
    parser.add_argument("--data-split", default="test")
    parser.add_argument("--output-root", default=".local/runs/relevance_projection")
    parser.add_argument("--output-prefix", default="alloprof_reranking_sgd")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--sample-seed", type=int, default=20260624)
    parser.add_argument("--member-seed-base", type=int, default=10101)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--max-features", type=int, default=80000)
    parser.add_argument("--min-df", type=int, default=3)
    parser.add_argument("--sgd-alpha", type=float, default=1e-5)
    parser.add_argument("--sgd-max-iter", type=int, default=60)
    parser.add_argument("--sgd-tol", type=float, default=1e-4)
    parser.add_argument("--methods", nargs="*", default=["raw", "ts", "isotonic", "hcal", "dirichlet", "smart"])
    parser.add_argument("--device", default="cuda")
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
    require_device(args.device)
    seed_everything(args.sample_seed)

    docs, queries = load_text_maps(args)
    qrels = list(load_dataset(args.dataset_name, split=args.data_split))
    for run_idx in range(args.runs):
        run_one(qrels, docs, queries, args, run_idx)


if __name__ == "__main__":
    main()
