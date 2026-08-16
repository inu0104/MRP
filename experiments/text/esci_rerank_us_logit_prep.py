#!/usr/bin/env python3
"""MTEB ESCI reranking reliability projection pilot.

The MTEB ESCIReranking release stores corpus, queries, qrels, and top-ranked
candidate ids in separate configs. This script builds binary query-product
relevance examples from one locale, using qrels positives and top-ranked
non-qrels negatives.
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


LABEL_NAMES = {0: "Non-relevant", 1: "Relevant"}


def load_maps(args):
    prefix = args.locale
    corpus = load_dataset(args.dataset_name, f"{prefix}-corpus", split=args.split)
    queries = load_dataset(args.dataset_name, f"{prefix}-queries", split=args.split)
    qrels = load_dataset(args.dataset_name, f"{prefix}-qrels", split=args.split)
    top_ranked = load_dataset(args.dataset_name, f"{prefix}-top_ranked", split=args.split)
    docs = {
        str(row["_id"]): "\n".join(
            part for part in [str(row.get("title") or "").strip(), str(row.get("text") or "").strip()] if part
        )
        for row in corpus
    }
    qs = {str(row["_id"]): str(row.get("text") or "").strip() for row in queries}
    positive = {(str(row["query-id"]), str(row["corpus-id"])) for row in qrels if int(row["score"]) > 0}
    ranked = {str(row["query-id"]): [str(cid) for cid in row["corpus-ids"]] for row in top_ranked}
    return docs, qs, positive, ranked


def split_query_ids(query_ids, seed: int):
    qids = np.asarray(sorted(query_ids), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(qids)
    n = len(qids)
    n_train = max(1, int(round(n * 0.6)))
    n_val = max(1, int(round(n * 0.2)))
    return set(qids[:n_train]), set(qids[n_train : n_train + n_val]), set(qids[n_train + n_val :])


def build_examples(docs, queries, positive, ranked, qid_set, args, seed: int):
    rng = np.random.default_rng(seed)
    texts: list[str] = []
    labels: list[int] = []
    qids: list[int] = []
    qid_to_int = {qid: idx for idx, qid in enumerate(sorted(ranked))}
    for qid in sorted(qid_set):
        query = queries.get(qid)
        candidates = ranked.get(qid, [])
        if not query or not candidates:
            continue
        positives = [cid for cid in candidates if (qid, cid) in positive and cid in docs]
        negatives = [cid for cid in candidates if (qid, cid) not in positive and cid in docs]
        if not positives or not negatives:
            continue
        rng.shuffle(positives)
        rng.shuffle(negatives)
        positives = positives[: args.max_positives_per_query]
        neg_n = min(len(negatives), max(1, int(round(len(positives) * args.negative_ratio))))
        negatives = negatives[: min(neg_n, args.max_negatives_per_query)]
        for cid in positives:
            texts.append(f"query: {query}\nproduct: {docs[cid]}")
            labels.append(1)
            qids.append(qid_to_int[qid])
        for cid in negatives:
            texts.append(f"query: {query}\nproduct: {docs[cid]}")
            labels.append(0)
            qids.append(qid_to_int[qid])
    order = rng.permutation(len(labels))
    if args.max_examples is not None:
        order = order[: min(int(args.max_examples), len(order))]
    return (
        [texts[int(i)] for i in order],
        np.asarray([labels[int(i)] for i in order], dtype=np.int64),
        np.asarray([qids[int(i)] for i in order], dtype=np.int64),
    )


def label_counts(labels: np.ndarray):
    return {LABEL_NAMES[int(k)]: int(v) for k, v in zip(*np.unique(labels, return_counts=True))}


def run_one(resources, args, run_idx: int):
    docs, queries, positive, ranked = resources
    out = Path(args.output_root) / f"esci_reranking_{args.locale}_sgd_seed{run_idx}"
    out.mkdir(parents=True, exist_ok=True)
    split_seed = args.sample_seed + run_idx * 1009
    train_q, val_q, test_q = split_query_ids(set(ranked), split_seed)
    train_texts, train_labels, train_qids = build_examples(docs, queries, positive, ranked, train_q, args, split_seed + 1)
    val_texts, val_labels, val_qids = build_examples(docs, queries, positive, ranked, val_q, args, split_seed + 2)
    test_texts, test_labels, test_qids = build_examples(docs, queries, positive, ranked, test_q, args, split_seed + 3)
    n_classes = 2
    split_counts = {
        "train": int(len(train_labels)),
        "val": int(len(val_labels)),
        "test": int(len(test_labels)),
    }
    print(f"[esci-reranking-{args.locale}] run={run_idx} counts={split_counts}", flush=True)
    print(
        f"[esci-reranking-{args.locale}] run={run_idx} labels={{'train': {label_counts(train_labels)}, 'val': {label_counts(val_labels)}, 'test': {label_counts(test_labels)}}}",
        flush=True,
    )

    member_seeds = [args.member_seed_base + run_idx * 100 + idx for idx in range(args.members)]
    val_logits = []
    test_logits = []
    temperatures = []
    for seed in member_seeds:
        print(f"[esci-reranking-{args.locale}] run={run_idx} train member seed={seed}", flush=True)
        model = train_member(seed, train_texts, train_labels, args.max_features)
        v_logits = decision_logits(model, val_texts, n_classes)
        t_logits = decision_logits(model, test_texts, n_classes)
        temp = fit_temperature(v_logits, val_labels)
        print(f"[esci-reranking-{args.locale}] run={run_idx} seed={seed} T={temp:.4f}", flush=True)
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
                "locale": args.locale,
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
                "query_counts": {"train": len(train_q), "val": len(val_q), "test": len(test_q)},
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
    parser.add_argument("--dataset-name", default="mteb/ESCIReranking")
    parser.add_argument("--locale", default="us", choices=["us", "es", "jp"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-root", default=".local/runs/relevance_projection")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--sample-seed", type=int, default=20260624)
    parser.add_argument("--member-seed-base", type=int, default=11101)
    parser.add_argument("--max-positives-per-query", type=int, default=5)
    parser.add_argument("--max-negatives-per-query", type=int, default=5)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--max-examples", type=int, default=None)
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

    resources = load_maps(args)
    for run_idx in range(args.runs):
        run_one(resources, args, run_idx)


if __name__ == "__main__":
    main()
