#!/usr/bin/env python3
"""Amazon ESCI post-calibration reliability ordering pilot.

This is a fast text-system pilot for MRP on query-product relevance
classification. It uses a TF-IDF + SGD ensemble to prepare saved logits for
calibration and reliability-ordering experiments.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from calibration.dirichlet import DirichletCalibrator  # noqa: E402
from calibration.diag import DiagonalOrderPreservingCalibrator  # noqa: E402
from calibration.h_calibration import HCalibrator  # noqa: E402
from calibration.isotonic import IsotonicCalibrator  # noqa: E402
from calibration.smart import SMARTCalibrator  # noqa: E402
from calibration.spline import TopLabelSplineCalibrator  # noqa: E402
from experiments.text.wilds_amazon_logit_prep import (  # noqa: E402
    FEATURE_SETS,
    auroc,
    average_precision,
    base_metrics,
    brier,
    ece_from_conf,
    entropy,
    fit_ridge,
    fit_temperature,
    matrix,
    nll,
    predict_ridge,
    prediction_arrays,
    quantile_ids,
    softmax,
)


LABEL_NAMES = {0: "Irrelevant", 1: "Complement", 2: "Substitute", 3: "Exact"}
BASE_LABELS = {
    "raw": "Raw ensemble",
    "ts": "TS",
    "isotonic": "Iso",
    "diag": "DIAG",
    "hcal": "h-cal ensemble mean",
    "dirichlet": "Dirichlet",
    "spline": "Spline",
    "smart": "SMART",
}
PROJECTION_LABELS = {
    "confidence": "Confidence",
    "controls": "Controls",
    "controls_mi": "Controls+MI",
    "controls_l2": "Controls+L2",
    "controls_mi_l2": "Controls+MI+L2",
    "controls_shuffled_mi": "Controls+shuffled MI",
}


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(xs, ys):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if len(xs) < 2 or np.std(xs) == 0 or np.std(ys) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(xs), rankdata(ys))[0, 1])


def ndcg_at_k(relevance, scores, k=10):
    relevance = np.asarray(relevance, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(relevance) == 0 or np.all(relevance <= 0):
        return float("nan")
    k = min(int(k), len(relevance))
    order = np.argsort(-scores, kind="mergesort")[:k]
    ideal = np.argsort(-relevance, kind="mergesort")[:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(relevance[order] * discounts))
    idcg = float(np.sum(relevance[ideal] * discounts))
    return dcg / idcg if idcg > 0 else float("nan")


def pairwise_accuracy(relevance, scores):
    relevance = np.asarray(relevance, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    total = 0.0
    correct = 0.0
    for i in range(len(relevance)):
        rel_diff = relevance[i] - relevance[i + 1 :]
        score_diff = scores[i] - scores[i + 1 :]
        keep = rel_diff != 0
        if not np.any(keep):
            continue
        products = rel_diff[keep] * score_diff[keep]
        correct += float((products > 0).sum())
        correct += 0.5 * float((products == 0).sum())
        total += float(keep.sum())
    return correct / total if total > 0 else float("nan")


def macro_f1(labels, preds, num_classes):
    scores = []
    for cls in range(num_classes):
        tp = np.sum((labels == cls) & (preds == cls))
        fp = np.sum((labels != cls) & (preds == cls))
        fn = np.sum((labels == cls) & (preds != cls))
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else float(2 * tp / denom))
    return float(np.mean(scores))


def balanced_accuracy(labels, preds, num_classes):
    recalls = []
    for cls in range(num_classes):
        mask = labels == cls
        recalls.append(0.0 if not np.any(mask) else float((preds[mask] == cls).mean()))
    return float(np.mean(recalls))


def adaptive_ece(conf, correct, bins=15):
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    order = np.argsort(conf, kind="mergesort")
    chunks = np.array_split(order, bins)
    score = 0.0
    for idx in chunks:
        if len(idx) == 0:
            continue
        score += len(idx) / len(conf) * abs(float(correct[idx].mean() - conf[idx].mean()))
    return float(score)


def classwise_ece(probs, labels, bins=15):
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    errors = []
    for cls in range(probs.shape[1]):
        cls_probs = probs[:, cls]
        cls_true = (labels == cls).astype(np.float64)
        errors.append(ece_from_conf(cls_probs, cls_true, bins=bins))
    return float(np.mean(errors))


def esci_base_metrics(base, split, arrays, probs, labels):
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    preds = probs.argmax(axis=1)
    row = base_metrics(base, split, arrays, probs, labels)
    row.update(
        {
            "macro_f1": macro_f1(labels, preds, probs.shape[1]),
            "balanced_accuracy": balanced_accuracy(labels, preds, probs.shape[1]),
            "normalized_nll": row["nll"] / math.log(probs.shape[1]),
            "adaptive_ece": adaptive_ece(arrays["confidence"], arrays["correct"]),
            "classwise_ece": classwise_ece(probs, labels),
        }
    )
    return row


def mean_log_probability(member_logits):
    probs = softmax(member_logits).mean(axis=0)
    return np.log(probs + 1e-12).astype(np.float32)


def make_text(row):
    query = str(row.get("query") or "").strip()
    title = str(row.get("product_title") or row.get("title") or "").strip()
    body = str(row.get("text") or "").strip()
    return f"query: {query}\nproduct_title: {title}\n{body}".strip()


def sample_dataset(dataset, n, seed):
    n = min(int(n), len(dataset))
    return dataset.shuffle(seed=seed).select(range(n))


def split_train_val(train_ds, max_train, max_val, seed):
    total = min(len(train_ds), int(max_train) + int(max_val))
    sampled = train_ds.shuffle(seed=seed).select(range(total))
    val_n = min(int(max_val), total)
    train_n = total - val_n
    return sampled.select(range(train_n)), sampled.select(range(train_n, total))


def dataset_to_xy(dataset):
    texts = [make_text(row) for row in dataset]
    labels = np.asarray(dataset["label"], dtype=np.int64)
    return texts, labels


def train_member(seed, train_texts, train_labels, max_features):
    model = make_pipeline(
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=3,
            max_features=max_features,
            sublinear_tf=True,
        ),
        SGDClassifier(
            loss="log_loss",
            alpha=1e-5,
            max_iter=14,
            tol=1e-4,
            random_state=seed,
            class_weight="balanced",
            n_jobs=1,
        ),
    )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_texts))
    model.fit([train_texts[i] for i in order], train_labels[order])
    return model


def decision_logits(model, texts, n_classes):
    logits = model.decision_function(texts)
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim == 1:
        logits = np.stack([-logits, logits], axis=1)
    if logits.shape[1] != n_classes:
        raise ValueError(f"Expected {n_classes} classes, got {logits.shape}")
    return logits


def risk_bins(arrays, min_n):
    specs = [("confidence", 5), ("entropy", 3), ("probability_margin", 3), ("mutual_information", 2)]
    ids = np.stack([quantile_ids(arrays[name], bins) for name, bins in specs], axis=1)
    groups = {}
    for idx, key in enumerate(map(tuple, ids)):
        groups.setdefault(key, []).append(idx)
    return [np.asarray(v, dtype=np.int64) for v in groups.values() if len(v) >= min_n]


def high_risk_ranking_eval(arrays, pred_residual, top_frac, min_bin_n):
    bins = risk_bins(arrays, min_bin_n)
    actual = []
    scores = []
    for idx in bins:
        actual.append(abs(float(arrays["residual"][idx].mean())))
        scores.append(abs(float(pred_residual[idx].mean())))
    actual = np.asarray(actual, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(actual) == 0:
        return {
            "n_bins": 0,
            "auroc": float("nan"),
            "auprc": float("nan"),
            "ndcg_at_10": float("nan"),
            "ndcg_all": float("nan"),
            "top_capture": float("nan"),
            "spearman": float("nan"),
            "pairwise_accuracy": float("nan"),
            "actual_worst_score": float("nan"),
            "predicted_top_actual_score": float("nan"),
        }
    k = max(1, int(math.ceil(len(actual) * top_frac)))
    actual_order = np.argsort(-actual, kind="mergesort")
    pred_order = np.argsort(-scores, kind="mergesort")
    labels = np.zeros(len(actual), dtype=np.int64)
    labels[actual_order[:k]] = 1
    capture = len(set(actual_order[:k].tolist()) & set(pred_order[:k].tolist())) / k
    return {
        "n_bins": int(len(actual)),
        "auroc": auroc(labels, scores),
        "auprc": average_precision(labels, scores),
        "ndcg_at_10": ndcg_at_k(actual, scores, k=10),
        "ndcg_all": ndcg_at_k(actual, scores, k=len(actual)),
        "top_capture": float(capture),
        "spearman": spearman(actual, scores),
        "pairwise_accuracy": pairwise_accuracy(actual, scores),
        "actual_worst_score": float(actual[actual_order[0]]),
        "predicted_top_actual_score": float(actual[pred_order[0]]),
    }


def fit_base_calibrators(val_logits, val_labels, temperatures, args):
    ensemble_logp = mean_log_probability(val_logits)
    calibrators = {}
    if "hcal" in args.methods:
        calibrators["hcal"] = HCalibrator(
            segments=args.hcal_segments,
            epochs=args.hcal_epochs,
            patience=args.hcal_patience,
            lr=args.hcal_lr,
            batch_size=args.hcal_batch_size,
            window=args.hcal_window,
            loss_weight=args.hcal_loss_weight,
            device=args.device,
        ).fit(ensemble_logp, val_labels)
    if "isotonic" in args.methods:
        calibrators["isotonic"] = IsotonicCalibrator().fit(ensemble_logp, val_labels)
    if "dirichlet" in args.methods:
        calibrators["dirichlet"] = DirichletCalibrator(
            max_iter=args.dirichlet_max_iter,
            lr=args.dirichlet_lr,
            weight_decay=args.dirichlet_weight_decay,
            device=args.device,
        ).fit(ensemble_logp, val_labels)
    if "diag" in args.methods:
        calibrators["diag"] = DiagonalOrderPreservingCalibrator(
            max_iter=args.diag_max_iter,
            lr=args.diag_lr,
            weight_decay=args.diag_weight_decay,
            device=args.device,
        ).fit(ensemble_logp, val_labels)
    if "spline" in args.methods:
        calibrators["spline"] = TopLabelSplineCalibrator(
            n_knots=args.spline_knots,
            degree=args.spline_degree,
            c=args.spline_c,
            max_iter=args.spline_max_iter,
        ).fit(ensemble_logp, val_labels)
    if "smart" in args.methods:
        calibrators["smart"] = SMARTCalibrator(
            hidden_dim=args.smart_hidden_dim,
            num_layers=args.smart_layers,
            epochs=args.smart_epochs,
            patience=args.smart_patience,
            lr=args.smart_lr,
            device=args.device,
        ).fit(ensemble_logp, val_labels)
    return calibrators


def calibrated_view(method, member_logits, temperatures, calibrators):
    if method == "raw":
        member_probs = softmax(member_logits)
        probs = member_probs.mean(axis=0)
        logits = np.log(probs + 1e-12).astype(np.float32)
        return logits, probs, member_probs
    if method == "ts":
        ts_logits = member_logits / temperatures[:, None, None]
        member_probs = softmax(ts_logits)
        probs = member_probs.mean(axis=0)
        logits = np.log(probs + 1e-12).astype(np.float32)
        return logits, probs, member_probs
    ts_member_probs = softmax(member_logits / temperatures[:, None, None])
    logits = calibrators[method].transform_logits(mean_log_probability(member_logits))
    probs = softmax(logits)
    return logits, probs, ts_member_probs


def fmt(value):
    if value == "" or pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def markdown_table(rows, numeric_cols):
    if not rows:
        return ""
    df = pd.DataFrame(rows)
    headers = list(df.columns)
    aligns = ["---:" if h in numeric_cols else "---" for h in headers]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(aligns) + " |"]
    for _, row in df.iterrows():
        values = []
        for header in headers:
            value = row[header]
            values.append(fmt(value) if header in numeric_cols else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main_table(metric_rows, risk_rows):
    metrics = pd.DataFrame(metric_rows)
    risk = pd.DataFrame(risk_rows)
    rows = []
    for base_name in [m for m in ["raw", "ts", "isotonic", "hcal", "dirichlet", "smart"] if m in set(metrics["base"])]:
        metric = metrics[(metrics["base"] == base_name) & (metrics["split"] == "test")].iloc[0]
        for reliability_projection in ["confidence", "controls", "controls_mi"]:
            r = risk[
                (risk["base"] == base_name)
                & (risk["split"] == "test")
                & (risk["reliability_projection"] == reliability_projection)
                & (risk["top_fraction"] == 0.1)
            ].iloc[0]
            rows.append(
                {
                    "Dataset": "Amazon ESCI",
                    "Split": "Query-product test",
                    "Base": BASE_LABELS[base_name],
                    "Reliability Projection": PROJECTION_LABELS[reliability_projection],
                    "Full Acc": metric["accuracy"],
                    "Macro-F1": metric["macro_f1"],
                    "Balanced Acc": metric["balanced_accuracy"],
                    "NLL/logK": metric["normalized_nll"],
                    "Full NLL": metric["nll"],
                    "Full ECE": metric["ece"],
                    "Adaptive ECE": metric["adaptive_ece"],
                    "Classwise ECE": metric["classwise_ece"],
                    "Full Brier": metric["brier"],
                    "PairAcc": r["pairwise_accuracy"],
                    "Spearman": r["spearman"],
                    "NDCG-all": r["ndcg_all"],
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="shuttie/esci-us-small")
    parser.add_argument("--output-dir", default=".local/runs/relevance_projection/esci_sgd")
    parser.add_argument("--seeds", nargs="*", type=int, default=[3001, 3002, 3003])
    parser.add_argument("--sample-seed", type=int, default=20260515)
    parser.add_argument("--max-train", type=int, default=60000)
    parser.add_argument("--max-val", type=int, default=15000)
    parser.add_argument("--max-test", type=int, default=40000)
    parser.add_argument("--max-features", type=int, default=80000)
    parser.add_argument("--min-bin-n", type=int, default=60)
    parser.add_argument("--methods", nargs="*", default=["raw", "ts", "diag", "spline", "hcal", "smart"])
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--hcal-segments", type=int, default=50)
    parser.add_argument("--hcal-epochs", type=int, default=90)
    parser.add_argument("--hcal-patience", type=int, default=20)
    parser.add_argument("--hcal-lr", type=float, default=0.005)
    parser.add_argument("--hcal-batch-size", type=int, default=6000)
    parser.add_argument("--hcal-window", type=int, default=200)
    parser.add_argument("--hcal-loss-weight", type=float, default=1e5)
    parser.add_argument("--dirichlet-max-iter", type=int, default=180)
    parser.add_argument("--dirichlet-lr", type=float, default=0.1)
    parser.add_argument("--dirichlet-weight-decay", type=float, default=1e-3)
    parser.add_argument("--diag-max-iter", type=int, default=120)
    parser.add_argument("--diag-lr", type=float, default=0.1)
    parser.add_argument("--diag-weight-decay", type=float, default=1e-3)
    parser.add_argument("--spline-knots", type=int, default=8)
    parser.add_argument("--spline-degree", type=int, default=3)
    parser.add_argument("--spline-c", type=float, default=1.0)
    parser.add_argument("--spline-max-iter", type=int, default=1000)
    parser.add_argument("--smart-hidden-dim", type=int, default=16)
    parser.add_argument("--smart-layers", type=int, default=2)
    parser.add_argument("--smart-epochs", type=int, default=180)
    parser.add_argument("--smart-patience", type=int, default=35)
    parser.add_argument("--smart-lr", type=float, default=0.005)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset_name)
    train_ds, val_ds = split_train_val(ds["train"], args.max_train, args.max_val, args.sample_seed)
    test_ds = sample_dataset(ds["test"], args.max_test, args.sample_seed + 1)
    train_texts, train_labels = dataset_to_xy(train_ds)
    val_texts, val_labels = dataset_to_xy(val_ds)
    test_texts, test_labels = dataset_to_xy(test_ds)
    n_classes = 4
    split_counts = {
        "train": int(len(train_labels)),
        "val": int(len(val_labels)),
        "test": int(len(test_labels)),
    }
    label_counts = {
        split: {LABEL_NAMES[int(k)]: int(v) for k, v in pd.Series(labels).value_counts().sort_index().items()}
        for split, labels in [("train", train_labels), ("val", val_labels), ("test", test_labels)]
    }
    print(f"sample counts: {split_counts}", flush=True)
    print(f"label counts: {label_counts}", flush=True)

    val_logits = []
    test_logits = []
    temperatures = []
    for seed in args.seeds:
        print(f"training ESCI member seed={seed}", flush=True)
        model = train_member(seed, train_texts, train_labels, args.max_features)
        v_logits = decision_logits(model, val_texts, n_classes)
        t_logits = decision_logits(model, test_texts, n_classes)
        temp = fit_temperature(v_logits, val_labels)
        print(f"  T={temp:.4f}", flush=True)
        val_logits.append(v_logits)
        test_logits.append(t_logits)
        temperatures.append(temp)
    val_logits = np.stack(val_logits).astype(np.float32)
    test_logits = np.stack(test_logits).astype(np.float32)
    temperatures = np.asarray(temperatures, dtype=np.float64)
    calibrators = fit_base_calibrators(val_logits, val_labels, temperatures, args)
    rng = np.random.default_rng(args.sample_seed + 17)

    arrays = {}
    metric_rows = []
    risk_rows = []
    base_methods = [m for m in ["raw", "ts", "diag", "spline", "hcal", "smart"] if m in args.methods]
    for method in base_methods:
        arrays[method] = {}
        for split, logits, labels in [("val", val_logits, val_labels), ("test", test_logits, test_labels)]:
            _cal_logits, probs, member_probs = calibrated_view(method, logits, temperatures, calibrators)
            arr = prediction_arrays(probs, labels, member_probs, rng)
            arrays[method][split] = arr
            metric_rows.append(esci_base_metrics(method, split, arr, probs, labels))

        for reliability_projection, features in FEATURE_SETS.items():
            model = fit_ridge(matrix(arrays[method]["val"], features), arrays[method]["val"]["residual"].astype(np.float64))
            pred = predict_ridge(model, matrix(arrays[method]["test"], features))
            for frac in [0.1, 0.2]:
                risk_rows.append(
                    {
                        "base": method,
                        "split": "test",
                        "reliability_projection": reliability_projection,
                        "top_fraction": frac,
                        **high_risk_ranking_eval(arrays[method]["test"], pred, frac, args.min_bin_n),
                    }
                )

    main_rows = main_table(metric_rows, risk_rows)
    pd.DataFrame(metric_rows).to_csv(out / "base_metrics.csv", index=False)
    pd.DataFrame(risk_rows).to_csv(out / "risk_ranking.csv", index=False)
    pd.DataFrame(main_rows).to_csv(out / "main_table.csv", index=False)
    np.savez_compressed(
        out / "logits_and_labels.npz",
        val_logits=val_logits.astype(np.float32),
        val_labels=val_labels.astype(np.int64),
        test_logits=test_logits.astype(np.float32),
        test_labels=test_labels.astype(np.int64),
        temperatures=temperatures.astype(np.float32),
    )

    numeric = {
        "Full Acc",
        "Macro-F1",
        "Balanced Acc",
        "NLL/logK",
        "Full NLL",
        "Full ECE",
        "Adaptive ECE",
        "Classwise ECE",
        "Full Brier",
        "PairAcc",
        "Spearman",
        "NDCG-all",
    }
    report = [
        "# Amazon ESCI Reliability Projection Pilot",
        "",
        "Scope: lightweight TF-IDF + SGD ensemble on Amazon Shopping Queries ESCI. Reliability projections preserve base calibrated probabilities and order calibrated relevance predictions by predicted residual risk.",
        "",
        "## Sample Counts",
        "",
        markdown_table([{"split": k, "n": v} for k, v in split_counts.items()], {"n"}),
        "",
        "## Label Counts",
        "",
        markdown_table(
            [{"split": split, **counts} for split, counts in label_counts.items()],
            set(LABEL_NAMES.values()),
        ),
        "",
        "## Main Table",
        "",
        markdown_table(main_rows, numeric),
        "",
    ]
    (out / "report.md").write_text("\n".join(report))
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset_name,
                "base_model": "tfidf_sgd_logistic",
                "seeds": args.seeds,
                "n_classes": n_classes,
                "label_names": LABEL_NAMES,
                "split_counts": split_counts,
                "label_counts": label_counts,
                "note": "Fast pilot only; final benchmark should use a transformer or e-commerce encoder ensemble.",
            },
            indent=2,
        )
        + "\n"
    )
    print(out / "report.md")


if __name__ == "__main__":
    main()
