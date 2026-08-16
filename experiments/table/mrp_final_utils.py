#!/usr/bin/env python3
"""Shared utilities for the final MRP relevance experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.text.amazon_esci_logit_prep import (  # noqa: E402
    calibrated_view as relevance_calibrated_view,
    fit_base_calibrators as relevance_fit_base_calibrators,
)


EPS = 1e-6

BASES = ["raw", "ts", "diag", "spline", "isotonic", "hcal", "dirichlet", "smart"]
FINAL_BASES = ["raw", "ts", "diag", "spline", "hcal", "smart"]
BASE_LABELS = {
    "raw": "Uncal.",
    "ts": "TS",
    "diag": "DIAG",
    "spline": "Spline",
    "isotonic": "Iso",
    "hcal": "h-cal",
    "hcal_ensemble_mean": "h-cal",
    "dirichlet": "Dirichlet",
    "smart": "SMART",
}


@dataclass
class TopInfo:
    confidence: np.ndarray
    top_label: np.ndarray
    second_label: np.ndarray
    correctness: np.ndarray


def clip_probs(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), EPS, 1.0 - EPS)


def ece_from_conf(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    score = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (conf >= left) & (conf <= right) if right == 1.0 else (conf >= left) & (conf < right)
        if np.any(mask):
            score += float(mask.mean() * abs(correct[mask].mean() - conf[mask].mean()))
    return score


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = clip_probs(probs)
    labels = np.asarray(labels, dtype=np.int64)
    return float(-np.log(probs[np.arange(len(labels)), labels]).mean())


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(np.sum((probs - onehot) ** 2, axis=1).mean())


def load_meta_args(path: Path, methods: list[str], device: str) -> SimpleNamespace:
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    defaults = {
        "methods": FINAL_BASES,
        "device": device,
        "hcal_segments": 50,
        "hcal_epochs": 80,
        "hcal_patience": 18,
        "hcal_lr": 0.005,
        "hcal_batch_size": 6000,
        "hcal_window": 200,
        "hcal_loss_weight": 1e5,
        "dirichlet_max_iter": 160,
        "dirichlet_lr": 0.1,
        "dirichlet_weight_decay": 1e-3,
        "diag_max_iter": 120,
        "diag_lr": 0.1,
        "diag_weight_decay": 1e-3,
        "spline_knots": 8,
        "spline_degree": 3,
        "spline_c": 1.0,
        "spline_max_iter": 1000,
        "smart_hidden_dim": 16,
        "smart_layers": 2,
        "smart_epochs": 160,
        "smart_patience": 35,
        "smart_lr": 0.005,
    }
    defaults.update(meta)
    defaults["device"] = device
    requested = set(methods)
    defaults["methods"] = [method for method in BASES if method in requested]
    return SimpleNamespace(**defaults)


def dataset_config(name: str) -> dict[str, object]:
    configs = {
        "esci_reranking_us": ("ESCI-Rerank-US", ".local/runs/relevance_projection/esci_reranking_us_sgd_seed*"),
        "esci_reranking_us_distilbert": (
            "ESCI-Rerank-US (DistilBERT)",
            ".local/runs/relevance_projection/esci_reranking_us_distilbert_seed*",
        ),
        "mslr": ("MSLR-WEB10K", ".local/runs/relevance_projection/mslr_web10k_sgd_seed*"),
        "mslr_lightgbm": ("MSLR-WEB10K (LightGBM)", ".local/runs/relevance_projection/mslr_web10k_lightgbm_seed*"),
        "mslr_lightgbm_balanced": (
            "MSLR-WEB10K (LightGBM balanced)",
            ".local/runs/relevance_projection/mslr_web10k_lightgbm_balanced_seed*",
        ),
        "esci": ("Amazon ESCI", ".local/runs/relevance_projection/esci_sgd_seed*"),
        "scidocs": ("SciDocs", ".local/runs/relevance_projection/scidocs_reranking_sgd_seed*"),
        "scidocs_msmarco_minilm": (
            "SciDocs (MS MARCO MiniLM)",
            ".local/runs/relevance_projection/scidocs_reranking_msmarco_minilm_seed*",
        ),
        "wands": ("WANDS", ".local/runs/relevance_projection/wands_sgd_seed*"),
        "alloprof": ("Alloprof-Rerank", ".local/runs/relevance_projection/alloprof_reranking_sgd_seed*"),
        "alloprof_distilbert": (
            "Alloprof-Rerank (DistilBERT)",
            ".local/runs/relevance_projection/alloprof_reranking_distilbert_seed*",
        ),
        "alloprof_msmarco_minilm": (
            "Alloprof-Rerank (MS MARCO MiniLM)",
            ".local/runs/relevance_projection/alloprof_reranking_msmarco_minilm_seed*",
        ),
        "stackoverflow_msmarco_minilm": (
            "StackOverflowDupQuestions (MS MARCO MiniLM)",
            ".local/runs/relevance_projection/stackoverflowdupquestions_msmarco_minilm_seed*",
        ),
    }
    if name not in configs:
        raise ValueError(f"Unknown final MRP dataset {name}")
    dataset, glob = configs[name]
    return {
        "dataset": dataset,
        "glob": glob,
        "test_logits_key": "test_logits",
        "test_labels_key": "test_labels",
        "fit": relevance_fit_base_calibrators,
        "view": relevance_calibrated_view,
        "fit_needs_temperatures": True,
    }


def fit_calibrators(config, val_logits, val_labels, temperatures, args):
    if config["fit_needs_temperatures"]:
        return config["fit"](val_logits, val_labels, temperatures, args)
    return config["fit"](val_logits, val_labels, args)


def split_indices(n: int, cal_fraction: float, projection_fit_fraction: float, seed: int):
    if not 0.0 < cal_fraction < 1.0:
        raise ValueError("--cal-fraction must be in (0, 1)")
    if not 0.0 < projection_fit_fraction < 1.0:
        raise ValueError("--projection-fit-fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_cal = max(1, int(round(n * cal_fraction)))
    remaining = n - n_cal
    n_projection_fit = max(1, int(round(remaining * projection_fit_fraction)))
    n_projection_fit = min(n_projection_fit, remaining - 1) if remaining > 1 else remaining
    cal_idx = perm[:n_cal]
    projection_fit_idx = perm[n_cal : n_cal + n_projection_fit]
    projection_selection_idx = perm[n_cal + n_projection_fit :]
    if len(projection_selection_idx) == 0:
        projection_selection_idx = projection_fit_idx
    return cal_idx, projection_fit_idx, projection_selection_idx


def cap_indices(idx: np.ndarray, cap: int, seed: int) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    if cap <= 0 or len(idx) <= cap:
        return idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=int(cap), replace=False))


def subset_logits(logits: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.asarray(logits)[:, idx, :]


def top_info(probs: np.ndarray, labels: np.ndarray) -> TopInfo:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    order = np.argsort(-probs, axis=1, kind="stable")
    top = order[:, 0].astype(np.int64)
    second = order[:, 1].astype(np.int64)
    conf = probs[np.arange(len(probs)), top]
    correct = (top == labels).astype(np.float64)
    return TopInfo(confidence=conf, top_label=top, second_label=second, correctness=correct)


def selective_metrics(arrays: dict[str, np.ndarray], risk: np.ndarray, coverage: float) -> dict[str, float]:
    confidence = arrays["confidence"].astype(np.float64)
    correct = arrays["correct"].astype(np.float64)
    order = np.argsort(np.asarray(risk, dtype=np.float64), kind="mergesort")
    keep_n = max(1, int(np.ceil(len(order) * coverage)))
    keep = order[:keep_n]
    return {
        "coverage": float(coverage),
        "selective_acc": float(correct[keep].mean()),
        "selective_ece": ece_from_conf(confidence[keep], correct[keep]),
        "mean_confidence": float(confidence[keep].mean()),
    }
