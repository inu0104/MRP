"""Small probability, metric, and feature utilities."""

from __future__ import annotations

import math

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


def entropy(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    return -(probs * np.log(probs + 1e-12)).sum(axis=-1)


def ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    score = 0.0
    for left, right in zip(np.linspace(0, 1, bins, endpoint=False), np.linspace(1 / bins, 1, bins)):
        mask = (confidence >= left) & (confidence <= right if right == 1 else confidence < right)
        if np.any(mask):
            score += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(score)


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probs = np.asarray(probs, dtype=np.float64)
    return float(-np.log(probs[np.arange(len(labels)), labels] + 1e-12).mean())


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probs = np.asarray(probs, dtype=np.float64)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(np.sum((probs - onehot) ** 2, axis=1).mean())


def top_label_arrays(probs: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probs = np.asarray(probs, dtype=np.float64)
    pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = (pred == labels).astype(np.float64)
    return {
        "prediction": pred,
        "confidence": confidence,
        "correct": correct,
        "residual": correct - confidence,
    }


def probability_features(probs: np.ndarray) -> dict[str, np.ndarray]:
    probs = np.asarray(probs, dtype=np.float64)
    sorted_probs = np.sort(probs, axis=1)
    log_probs = np.log(probs + 1e-12)
    sorted_logits = np.sort(log_probs, axis=1)
    return {
        "confidence": sorted_probs[:, -1],
        "entropy": entropy(probs) / math.log(probs.shape[1]),
        "probability_margin": sorted_probs[:, -1] - sorted_probs[:, -2],
        "logit_gap": sorted_logits[:, -1] - sorted_logits[:, -2],
    }


def ensemble_mutual_information(member_probs: np.ndarray) -> np.ndarray:
    member_probs = np.asarray(member_probs, dtype=np.float64)
    mean_probs = member_probs.mean(axis=0)
    total = entropy(mean_probs)
    aleatoric = entropy(member_probs).mean(axis=0)
    return ((total - aleatoric) / math.log(member_probs.shape[-1])).astype(np.float64)


def ensemble_l2_spread(member_probs: np.ndarray) -> np.ndarray:
    member_probs = np.asarray(member_probs, dtype=np.float64)
    mean_probs = member_probs.mean(axis=0)
    return ((member_probs - mean_probs[None, :, :]) ** 2).sum(axis=-1).mean(axis=0)


def mean_log_probability(member_logits: np.ndarray) -> np.ndarray:
    probs = softmax(member_logits).mean(axis=0)
    return np.log(probs + 1e-12).astype(np.float32)

