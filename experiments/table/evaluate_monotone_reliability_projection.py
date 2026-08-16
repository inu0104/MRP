#!/usr/bin/env python3
"""Evaluate label-wise monotone reliability projection.

This script implements a non-additive post-calibration reliability projection:

    q_i = T_{k_i}(c_i, g_i)

where c_i is the calibrated confidence assigned to a fixed raw top-label
decision, k_i is that fixed top label, and g_i is the top/runner-up gap for
that fixed decision.  T_k is represented as a monotone lattice surface and is
trained with correctness BCE plus a small second-difference penalty:

    BCE(O_i, q_i) + rho * R_Delta2(T).

The additive M1/M1+gap variants are included only as low-capacity special-case
baselines.

Final experiments intentionally separate model and calibration randomness.  A
single saved model member is selected with ``--member-index`` from one saved
model run, and ``--calibration-seeds`` only changes the calibrator/MRP
validation split and optimizer seeds.  The raw top-label decision from that
single member is kept fixed for every calibrator so that accuracy is a property
of the base predictor, not of the post-hoc calibration method.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.table.mrp_final_utils import (  # noqa: E402
    BASE_LABELS,
    BASES,
    cap_indices,
    dataset_config,
    fit_calibrators,
    load_meta_args,
    selective_metrics,
    split_indices,
    subset_logits,
    top_info,
)
from experiments.reproducibility import seed_everything  # noqa: E402


EPS = 1e-6


@dataclass
class LabelConstantModel:
    lambda_l2: float
    alpha: np.ndarray


@dataclass
class PerLabelPlattModel:
    lambda_l2: float
    slope: np.ndarray
    bias: np.ndarray


@dataclass
class ReliabilityIsotonicModel:
    labelwise: bool
    models: list[IsotonicRegression | None]


def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float64), EPS, 1.0 - EPS)


def logit_np(x: np.ndarray) -> np.ndarray:
    x = clip01(x)
    return np.log(x) - np.log1p(-x)


def ece_from_conf(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    conf = clip01(conf)
    correct = np.asarray(correct, dtype=np.float64)
    score = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (conf >= left) & (conf <= right) if right == 1.0 else (conf >= left) & (conf < right)
        if np.any(mask):
            score += float(mask.mean() * abs(correct[mask].mean() - conf[mask].mean()))
    return score


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=np.float64), -60.0, 60.0)))


def rankdata(values: np.ndarray) -> np.ndarray:
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


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[pos].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    n_pos = int(sorted_labels.sum())
    if n_pos == 0:
        return float("nan")
    tp = np.cumsum(sorted_labels)
    precision = tp / (np.arange(len(sorted_labels)) + 1)
    return float((precision * sorted_labels).sum() / n_pos)


def binary_nll(correct: np.ndarray, q: np.ndarray) -> float:
    correct = np.asarray(correct, dtype=np.float64)
    q = clip01(q)
    return float(-(correct * np.log(q) + (1.0 - correct) * np.log1p(-q)).mean())


def binary_brier(correct: np.ndarray, q: np.ndarray) -> float:
    return float(np.mean((np.asarray(correct, dtype=np.float64) - np.asarray(q, dtype=np.float64)) ** 2))


def aurc(correct: np.ndarray, risk: np.ndarray) -> float:
    correct = np.asarray(correct, dtype=np.float64)
    order = np.argsort(np.asarray(risk, dtype=np.float64), kind="mergesort")
    error = 1.0 - correct[order]
    return float(np.mean(np.cumsum(error) / (np.arange(len(error)) + 1)))


def top_risk_error(correct: np.ndarray, risk: np.ndarray, fraction: float) -> float:
    n = max(1, int(math.ceil(len(correct) * fraction)))
    idx = np.argsort(-np.asarray(risk, dtype=np.float64), kind="mergesort")[:n]
    return float((1.0 - np.asarray(correct, dtype=np.float64)[idx]).mean())


def eval_q(correct: np.ndarray, q: np.ndarray) -> dict[str, float]:
    risk = 1.0 - clip01(q)
    wrong = (1.0 - np.asarray(correct, dtype=np.float64)).astype(np.int64)
    arrays = {"confidence": clip01(q), "correct": np.asarray(correct, dtype=np.float64)}
    return {
        "Binary_NLL": binary_nll(correct, q),
        "Binary_Brier": binary_brier(correct, q),
        "Wrong_AUROC": auroc(wrong, risk),
        "Wrong_AUPRC": average_precision(wrong, risk),
        "AURC": aurc(correct, risk),
        "Top20RiskErr": top_risk_error(correct, risk, 0.20),
        "SelAcc80": selective_metrics(arrays, risk, 0.80)["selective_acc"],
    }


def fit_label_constant_for_lambda(info, n_classes: int, lambda_l2: float, max_iter: int) -> LabelConstantModel:
    labels = torch.tensor(info.top_label, dtype=torch.long)
    base = torch.tensor(logit_np(info.confidence), dtype=torch.float32)
    target = torch.tensor(info.correctness, dtype=torch.float32)
    alpha = torch.zeros(int(n_classes), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.LBFGS([alpha], lr=0.8, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = base + alpha[labels]
        loss = F.binary_cross_entropy_with_logits(logits, target) + float(lambda_l2) * alpha.pow(2).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return LabelConstantModel(lambda_l2=float(lambda_l2), alpha=alpha.detach().cpu().numpy().astype(np.float64))


def predict_label_constant(info, model: LabelConstantModel) -> np.ndarray:
    return clip01(sigmoid_np(logit_np(info.confidence) + model.alpha[info.top_label]))


def choose_label_constant(info_fit, info_val, n_classes: int, lambdas: list[float], max_iter: int) -> LabelConstantModel:
    best = None
    best_score = float("inf")
    for lam in lambdas:
        model = fit_label_constant_for_lambda(info_fit, n_classes, lam, max_iter)
        score = binary_nll(info_val.correctness, predict_label_constant(info_val, model))
        if score < best_score - 1e-12 or (abs(score - best_score) <= 1e-12 and (best is None or lam > best.lambda_l2)):
            best = model
            best_score = score
    assert best is not None
    return best


def fit_per_label_platt_for_lambda(info, n_classes: int, lambda_l2: float, max_iter: int) -> PerLabelPlattModel:
    labels = torch.tensor(info.top_label, dtype=torch.long)
    base = torch.tensor(logit_np(info.confidence), dtype=torch.float32)
    target = torch.tensor(info.correctness, dtype=torch.float32)
    raw_slope = torch.full((int(n_classes),), 0.54132485, dtype=torch.float32, requires_grad=True)
    bias = torch.zeros(int(n_classes), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.LBFGS([raw_slope, bias], lr=0.8, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        slope = F.softplus(raw_slope)
        logits = slope[labels] * base + bias[labels]
        reg = (slope - 1.0).pow(2).mean() + bias.pow(2).mean()
        loss = F.binary_cross_entropy_with_logits(logits, target) + float(lambda_l2) * reg
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        slope = F.softplus(raw_slope).detach().cpu().numpy().astype(np.float64)
    return PerLabelPlattModel(lambda_l2=float(lambda_l2), slope=slope, bias=bias.detach().cpu().numpy().astype(np.float64))


def predict_per_label_platt(info, model: PerLabelPlattModel) -> np.ndarray:
    labels = info.top_label.astype(np.int64)
    logits = model.slope[labels] * logit_np(info.confidence) + model.bias[labels]
    return clip01(sigmoid_np(logits))


def choose_per_label_platt(info_fit, info_val, n_classes: int, lambdas: list[float], max_iter: int) -> PerLabelPlattModel:
    best = None
    best_score = float("inf")
    for lam in lambdas:
        model = fit_per_label_platt_for_lambda(info_fit, n_classes, lam, max_iter)
        score = binary_nll(info_val.correctness, predict_per_label_platt(info_val, model))
        if score < best_score - 1e-12 or (abs(score - best_score) <= 1e-12 and (best is None or lam > best.lambda_l2)):
            best = model
            best_score = score
    assert best is not None
    return best


def fit_reliability_isotonic(info, n_classes: int, *, labelwise: bool) -> ReliabilityIsotonicModel:
    confidence = clip01(info.confidence)
    correctness = np.asarray(info.correctness, dtype=np.float64)
    labels = np.asarray(info.top_label, dtype=np.int64)
    models: list[IsotonicRegression | None] = []
    groups = int(n_classes) if labelwise else 1
    for group in range(groups):
        mask = labels == group if labelwise else np.ones_like(labels, dtype=bool)
        if int(mask.sum()) == 0:
            models.append(None)
            continue
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(confidence[mask], correctness[mask])
        models.append(model)
    return ReliabilityIsotonicModel(labelwise=bool(labelwise), models=models)


def predict_reliability_isotonic(info, model: ReliabilityIsotonicModel) -> np.ndarray:
    confidence = clip01(info.confidence)
    if not model.labelwise:
        fitted = model.models[0]
        return clip01(fitted.predict(confidence) if fitted is not None else confidence)

    labels = np.asarray(info.top_label, dtype=np.int64)
    q = confidence.copy()
    for label, fitted in enumerate(model.models):
        mask = labels == label
        if not np.any(mask) or fitted is None:
            continue
        q[mask] = fitted.predict(confidence[mask])
    return clip01(q)


def inv_softplus_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.log(np.expm1(np.maximum(x, 1e-8)))


def top_runner_gap(probs: np.ndarray) -> np.ndarray:
    probs = clip01(probs)
    order = np.argsort(probs, axis=1)
    top = order[:, -1]
    second = order[:, -2]
    rows = np.arange(len(probs))
    return (logit_np(probs[rows, top]) - logit_np(probs[rows, second])).astype(np.float64)


def select_member_logits(logits: np.ndarray, member_index: int) -> np.ndarray:
    logits = np.asarray(logits)
    if logits.ndim != 3:
        raise ValueError(f"Expected logits with shape (members, n, classes), got {logits.shape}")
    member_index = int(member_index)
    if member_index < 0 or member_index >= logits.shape[0]:
        raise IndexError(f"member_index={member_index} is out of range for {logits.shape[0]} members")
    return logits[member_index : member_index + 1]


def select_member_temperatures(temperatures: np.ndarray, member_index: int) -> np.ndarray:
    temperatures = np.asarray(temperatures, dtype=np.float64).reshape(-1)
    if len(temperatures) == 0:
        return np.asarray([1.0], dtype=np.float64)
    member_index = int(member_index)
    if member_index < 0 or member_index >= len(temperatures):
        raise IndexError(f"member_index={member_index} is out of range for {len(temperatures)} temperatures")
    return temperatures[member_index : member_index + 1]


def fixed_decision_info(probs: np.ndarray, labels: np.ndarray, raw_info) -> object:
    """Use calibrated confidence for the raw top-label decision.

    The post-hoc calibrator may reshape the probability vector, but the
    correctness event being evaluated remains the raw model's selected label.
    """

    probs = clip01(probs)
    rows = np.arange(len(probs))
    info = type(raw_info)(
        confidence=probs[rows, raw_info.top_label].astype(np.float64),
        top_label=raw_info.top_label.astype(np.int64),
        second_label=raw_info.second_label.astype(np.int64),
        correctness=raw_info.correctness.astype(np.float64),
    )
    return info


def fixed_runner_gap(probs: np.ndarray, raw_info) -> np.ndarray:
    probs = clip01(probs)
    rows = np.arange(len(probs))
    top = raw_info.top_label
    second = raw_info.second_label
    return (logit_np(probs[rows, top]) - logit_np(probs[rows, second])).astype(np.float64)


def robust_gap_bounds(gap: np.ndarray):
    gap = np.asarray(gap, dtype=np.float64)
    lo, hi = np.quantile(gap, [0.01, 0.99])
    if hi - lo < 1e-8:
        lo, hi = float(gap.min()), float(gap.max())
    if hi - lo < 1e-8:
        hi = lo + 1.0
    return float(lo), float(hi)


def scale_gap(gap: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((np.asarray(gap, dtype=np.float64) - lo) / max(hi - lo, 1e-8), 0.0, 1.0)


def deterministic_cumsum(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Small-loop cumulative sum that works with strict CUDA determinism."""

    pieces = []
    running = None
    for piece in torch.unbind(values, dim=dim):
        running = piece if running is None else running + piece
        pieces.append(running)
    return torch.stack(pieces, dim=dim)


@dataclass
class LatticeFit:
    variant: str
    lambda_anchor: float
    gap_lo: float
    gap_hi: float
    model: "MonotoneLattice"


class MonotoneLattice(nn.Module):
    """Hard-monotone lattice in confidence and optionally gap.

    The lattice is parameterized on the logit scale.  For 2D surfaces, values
    are monotone because row/column/interaction increments are softplus-positive.
    """

    def __init__(self, groups: int, c_knots: int, g_knots: int, *, use_gap: bool, device: torch.device):
        super().__init__()
        self.groups = int(groups)
        self.c_knots = int(c_knots)
        self.g_knots = int(g_knots if use_gap else 1)
        self.use_gap = bool(use_gap)
        if self.c_knots < 2:
            raise ValueError("c_knots must be >= 2")
        if self.use_gap and self.g_knots < 2:
            raise ValueError("g_knots must be >= 2 for 2D lattice")

        c_nodes = np.linspace(0.03, 0.97, self.c_knots)
        c_logits = logit_np(c_nodes)
        base_init = float(c_logits[0])
        c_diffs = np.diff(c_logits)

        self.base = nn.Parameter(torch.full((self.groups,), base_init, dtype=torch.float32, device=device))
        self.raw_c = nn.Parameter(
            torch.tensor(np.tile(inv_softplus_np(c_diffs)[None, :], (self.groups, 1)), dtype=torch.float32, device=device)
        )
        if self.use_gap:
            # Do not initialize these increments too close to zero.  With a
            # softplus-positive parameterization, a near-zero increment also has
            # a near-zero gradient and the gap axis effectively cannot wake up.
            tiny = inv_softplus_np(np.full((self.groups, self.g_knots - 1), 1e-2))
            inter = inv_softplus_np(np.full((self.groups, self.c_knots - 1, self.g_knots - 1), 1e-3))
            self.raw_g = nn.Parameter(torch.tensor(tiny, dtype=torch.float32, device=device))
            self.raw_inter = nn.Parameter(torch.tensor(inter, dtype=torch.float32, device=device))
        else:
            self.raw_g = None
            self.raw_inter = None

    def table(self) -> torch.Tensor:
        c_inc = F.softplus(self.raw_c)
        c_part = torch.cat([torch.zeros((self.groups, 1), device=c_inc.device), deterministic_cumsum(c_inc, dim=1)], dim=1)
        if not self.use_gap:
            return self.base[:, None] + c_part

        g_inc = F.softplus(self.raw_g)
        g_part = torch.cat([torch.zeros((self.groups, 1), device=g_inc.device), deterministic_cumsum(g_inc, dim=1)], dim=1)
        inter_inc = F.softplus(self.raw_inter)
        inter_part = torch.zeros((self.groups, self.c_knots, self.g_knots), device=inter_inc.device)
        inter_part[:, 1:, 1:] = deterministic_cumsum(deterministic_cumsum(inter_inc, dim=1), dim=2)
        return self.base[:, None, None] + c_part[:, :, None] + g_part[:, None, :] + inter_part

    def forward(self, c01: torch.Tensor, g01: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        table = self.table()
        group = labels.clamp(0, self.groups - 1)
        c = torch.clamp(c01, 0.0, 1.0) * (self.c_knots - 1)
        c0 = torch.floor(c).long().clamp(0, self.c_knots - 2)
        cf = c - c0.float()
        if not self.use_gap:
            values = table[group]
            logits = values.gather(1, c0[:, None]).squeeze(1) * (1.0 - cf) + values.gather(1, (c0 + 1)[:, None]).squeeze(1) * cf
            return torch.sigmoid(logits).clamp(EPS, 1.0 - EPS)

        g = torch.clamp(g01, 0.0, 1.0) * (self.g_knots - 1)
        g0 = torch.floor(g).long().clamp(0, self.g_knots - 2)
        gf = g - g0.float()
        values = table[group]
        v00 = values[torch.arange(len(values), device=values.device), c0, g0]
        v10 = values[torch.arange(len(values), device=values.device), c0 + 1, g0]
        v01 = values[torch.arange(len(values), device=values.device), c0, g0 + 1]
        v11 = values[torch.arange(len(values), device=values.device), c0 + 1, g0 + 1]
        logits = (
            v00 * (1.0 - cf) * (1.0 - gf)
            + v10 * cf * (1.0 - gf)
            + v01 * (1.0 - cf) * gf
            + v11 * cf * gf
        )
        return torch.sigmoid(logits).clamp(EPS, 1.0 - EPS)

    def smoothness(self) -> torch.Tensor:
        table = self.table()
        if not self.use_gap:
            if self.c_knots < 3:
                return torch.tensor(0.0, device=table.device)
            second = table[:, :-2] - 2.0 * table[:, 1:-1] + table[:, 2:]
            return second.pow(2).mean()
        loss = torch.tensor(0.0, device=table.device)
        if self.c_knots >= 3:
            second_c = table[:, :-2, :] - 2.0 * table[:, 1:-1, :] + table[:, 2:, :]
            loss = loss + second_c.pow(2).mean()
        if self.g_knots >= 3:
            second_g = table[:, :, :-2] - 2.0 * table[:, :, 1:-1] + table[:, :, 2:]
            loss = loss + second_g.pow(2).mean()
        return loss


def bern_kl(q: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    q = torch.clamp(q, EPS, 1.0 - EPS)
    c = torch.clamp(c, EPS, 1.0 - EPS)
    return q * (torch.log(q) - torch.log(c)) + (1.0 - q) * (torch.log1p(-q) - torch.log1p(-c))


def fit_lattice(
    info_fit,
    gap_fit: np.ndarray,
    info_val,
    gap_val: np.ndarray,
    *,
    variant: str,
    labelwise: bool,
    use_gap: bool,
    n_classes: int,
    c_knots: int,
    g_knots: int,
    lambda_grid: list[float],
    smooth: float,
    epochs: int,
    lr: float,
    device: torch.device,
) -> LatticeFit:
    gap_lo, gap_hi = robust_gap_bounds(gap_fit)
    labels_fit_np = info_fit.top_label if labelwise else np.zeros_like(info_fit.top_label)
    labels_val_np = info_val.top_label if labelwise else np.zeros_like(info_val.top_label)
    groups = n_classes if labelwise else 1

    fit_tensors = {
        "c": torch.tensor(clip01(info_fit.confidence), dtype=torch.float32, device=device),
        "g": torch.tensor(scale_gap(gap_fit, gap_lo, gap_hi), dtype=torch.float32, device=device),
        "labels": torch.tensor(labels_fit_np, dtype=torch.long, device=device),
        "y": torch.tensor(info_fit.correctness.astype(np.float64), dtype=torch.float32, device=device),
    }
    val_tensors = {
        "c": torch.tensor(clip01(info_val.confidence), dtype=torch.float32, device=device),
        "g": torch.tensor(scale_gap(gap_val, gap_lo, gap_hi), dtype=torch.float32, device=device),
        "labels": torch.tensor(labels_val_np, dtype=torch.long, device=device),
    }

    best_state = None
    best_lambda = None
    best_score = float("inf")
    for lambda_anchor in lambda_grid:
        model = MonotoneLattice(groups, c_knots, g_knots, use_gap=use_gap, device=device).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=float(lr))
        for _ in range(int(epochs)):
            opt.zero_grad()
            q = model(fit_tensors["c"], fit_tensors["g"], fit_tensors["labels"])
            loss = F.binary_cross_entropy(q, fit_tensors["y"])
            if lambda_anchor > 0:
                loss = loss + float(lambda_anchor) * bern_kl(q, fit_tensors["c"]).mean()
            if smooth > 0:
                loss = loss + float(smooth) * model.smoothness()
            loss.backward()
            opt.step()
        with torch.no_grad():
            q_val = model(val_tensors["c"], val_tensors["g"], val_tensors["labels"]).detach().cpu().numpy()
        score = float(eval_q(info_val.correctness, q_val)["Binary_NLL"])
        if score < best_score - 1e-12 or (abs(score - best_score) <= 1e-12 and (best_lambda is None or lambda_anchor > best_lambda)):
            best_score = score
            best_lambda = float(lambda_anchor)
            best_state = {key: val.detach().cpu().clone() for key, val in model.state_dict().items()}

    assert best_state is not None and best_lambda is not None
    model = MonotoneLattice(groups, c_knots, g_knots, use_gap=use_gap, device=device).to(device)
    model.load_state_dict(best_state)
    model.eval()
    return LatticeFit(variant=variant, lambda_anchor=best_lambda, gap_lo=gap_lo, gap_hi=gap_hi, model=model)


def predict_lattice(fit: LatticeFit, info, gap: np.ndarray, *, labelwise: bool, device: torch.device) -> np.ndarray:
    labels_np = info.top_label if labelwise else np.zeros_like(info.top_label)
    with torch.no_grad():
        q = fit.model(
            torch.tensor(clip01(info.confidence), dtype=torch.float32, device=device),
            torch.tensor(scale_gap(gap, fit.gap_lo, fit.gap_hi), dtype=torch.float32, device=device),
            torch.tensor(labels_np, dtype=torch.long, device=device),
        )
    return q.detach().cpu().numpy().astype(np.float64)


def resolve_device(requested: str) -> torch.device:
    requested = str(requested)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for MRP evaluation, but torch.cuda.is_available() is False.")
    return torch.device(requested)


def add_eval_row(rows, dataset, run_idx, base_label, variant, q, info, lambda_value):
    row = {
        "Dataset": dataset,
        "Run": run_idx,
        "Base": base_label,
        "Variant": variant,
        "Lambda": lambda_value,
    }
    row.update(eval_q(info.correctness, q))
    row["Binary_ECE"] = ece_from_conf(q, info.correctness)
    risk = 1.0 - clip01(q)
    arrays = {"confidence": clip01(q), "correct": info.correctness.astype(np.float64)}
    for coverage in [0.10, 0.50, 0.70, 0.90]:
        row[f"SelAcc{int(coverage * 100)}"] = selective_metrics(arrays, risk, coverage)["selective_acc"]
    rows.append(row)


def run_dataset(name: str, args: argparse.Namespace):
    config = dataset_config(name)
    all_run_dirs = sorted(Path(".").glob(config["glob"]))
    if not all_run_dirs:
        raise FileNotFoundError(f"No run directories found for {name}: {config['glob']}")
    if args.model_run_index < 0 or args.model_run_index >= len(all_run_dirs):
        raise IndexError(
            f"model_run_index={args.model_run_index} is out of range for {name}; found {len(all_run_dirs)} run dirs"
        )
    run_dir = all_run_dirs[args.model_run_index]
    rows = []
    device = resolve_device(args.device)
    for run_idx, calibration_seed in enumerate(args.calibration_seeds):
        seed_everything(args.seed + int(calibration_seed))
        print(
            f"[mrp] {config['dataset']} model_run={args.model_run_index} member={args.member_index} "
            f"cal_seed={calibration_seed} {run_dir}",
            flush=True,
        )
        data = np.load(run_dir / "logits_and_labels.npz")
        meta = load_meta_args(run_dir, args.methods, args.device)
        meta.methods = [m for m in BASES if m in set(args.methods)]
        val_logits = select_member_logits(data["val_logits"], args.member_index)
        val_labels = data["val_labels"]
        test_logits = select_member_logits(data[config["test_logits_key"]], args.member_index)
        test_labels = data[config["test_labels_key"]]
        temperatures = select_member_temperatures(data["temperatures"], args.member_index)
        split_seed = args.seed + int(calibration_seed) * 1009
        cal_idx, fit_idx, val_idx = split_indices(len(val_labels), args.cal_fraction, args.projection_fit_fraction, split_seed)
        fit_idx = cap_indices(fit_idx, args.projection_fit_cap, split_seed + 11)
        val_idx = cap_indices(val_idx, args.projection_selection_cap, split_seed + 23)
        calibrators = fit_calibrators(config, subset_logits(val_logits, cal_idx), val_labels[cal_idx], temperatures, meta)
        _raw_logits_fit, raw_fit_probs, _raw_fit_members = config["view"]("raw", subset_logits(val_logits, fit_idx), temperatures, calibrators)
        _raw_logits_selection, raw_selection_probs, _raw_selection_members = config["view"]("raw", subset_logits(val_logits, val_idx), temperatures, calibrators)
        _raw_logits_test, raw_test_probs, _raw_test_members = config["view"]("raw", test_logits, temperatures, calibrators)
        raw_fit_info = top_info(raw_fit_probs, val_labels[fit_idx])
        raw_selection_info = top_info(raw_selection_probs, val_labels[val_idx])
        raw_test_info = top_info(raw_test_probs, test_labels)
        for base in [m for m in BASES if m in set(meta.methods)]:
            base_label = BASE_LABELS[base]
            _logits, fit_probs, _fit_members = config["view"](base, subset_logits(val_logits, fit_idx), temperatures, calibrators)
            _logits, selection_probs, _selection_members = config["view"](base, subset_logits(val_logits, val_idx), temperatures, calibrators)
            _logits, test_probs, _test_members = config["view"](base, test_logits, temperatures, calibrators)
            n_classes = test_probs.shape[1]
            fit_info = fixed_decision_info(fit_probs, val_labels[fit_idx], raw_fit_info)
            selection_info = fixed_decision_info(selection_probs, val_labels[val_idx], raw_selection_info)
            test_info = fixed_decision_info(test_probs, test_labels, raw_test_info)
            fit_gap = fixed_runner_gap(fit_probs, raw_fit_info)
            selection_gap = fixed_runner_gap(selection_probs, raw_selection_info)
            test_gap = fixed_runner_gap(test_probs, raw_test_info)
            requested_variants = set(args.variants)

            if "M0_confidence" in requested_variants:
                add_eval_row(rows, config["dataset"], run_idx, base_label, "M0_confidence", test_info.confidence, test_info, float("nan"))

            if "GlobalIsotonic" in requested_variants:
                iso_model = fit_reliability_isotonic(fit_info, n_classes, labelwise=False)
                q_iso_test = predict_reliability_isotonic(test_info, iso_model)
                add_eval_row(rows, config["dataset"], run_idx, base_label, "GlobalIsotonic", q_iso_test, test_info, float("nan"))

            if "PerLabelIsotonic" in requested_variants:
                iso_model = fit_reliability_isotonic(fit_info, n_classes, labelwise=True)
                q_iso_test = predict_reliability_isotonic(test_info, iso_model)
                add_eval_row(rows, config["dataset"], run_idx, base_label, "PerLabelIsotonic", q_iso_test, test_info, float("nan"))

            if "PerLabelPlatt" in requested_variants:
                platt_model = choose_per_label_platt(fit_info, selection_info, n_classes, args.offset_lambda_l2, args.offset_max_iter)
                q_platt_test = predict_per_label_platt(test_info, platt_model)
                add_eval_row(rows, config["dataset"], run_idx, base_label, "PerLabelPlatt", q_platt_test, test_info, platt_model.lambda_l2)

            if "LabelConstant" in requested_variants:
                top_model = choose_label_constant(fit_info, selection_info, n_classes, args.offset_lambda_l2, args.offset_max_iter)
                q_top_test = predict_label_constant(test_info, top_model)
                add_eval_row(rows, config["dataset"], run_idx, base_label, "LabelConstant", q_top_test, test_info, top_model.lambda_l2)

            specs = [
                ("Shared1D", False, False, args.lambda_anchor),
                ("Label1D", True, False, args.lambda_anchor),
                ("Shared2D", False, True, args.lambda_anchor),
                ("Label2D", True, True, args.lambda_anchor),
                ("Label2D_NoAnchor", True, True, [0.0]),
            ]
            for variant, labelwise, use_gap, lambdas in specs:
                if variant not in requested_variants:
                    continue
                lattice = fit_lattice(
                    fit_info,
                    fit_gap,
                    selection_info,
                    selection_gap,
                    variant=variant,
                    labelwise=labelwise,
                    use_gap=use_gap,
                    n_classes=n_classes,
                    c_knots=args.c_knots,
                    g_knots=args.g_knots,
                    lambda_grid=lambdas,
                    smooth=args.smooth,
                    epochs=args.epochs,
                    lr=args.lr,
                    device=device,
                )
                q = predict_lattice(lattice, test_info, test_gap, labelwise=labelwise, device=device)
                add_eval_row(rows, config["dataset"], run_idx, base_label, variant, q, test_info, lattice.lambda_anchor)
    return rows


def summarize(raw: pd.DataFrame):
    metrics = [
        "Binary_NLL",
        "Binary_ECE",
        "Binary_Brier",
        "Wrong_AUROC",
        "Wrong_AUPRC",
        "AURC",
        "Top20RiskErr",
        "SelAcc10",
        "SelAcc50",
        "SelAcc70",
        "SelAcc90",
    ]
    rows = []
    for key, group in raw.groupby(["Dataset", "Base", "Variant"], dropna=False):
        out = dict(zip(["Dataset", "Base", "Variant"], key))
        for metric in metrics:
            vals = group[metric].astype(float)
            out[f"{metric}_mean"] = float(vals.mean())
            out[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(out)
    summary = pd.DataFrame(rows)
    ref = summary[summary["Variant"] == "M0_confidence"].set_index(["Dataset", "Base"])
    delta_rows = []
    for _, row in summary[summary["Variant"] != "M0_confidence"].iterrows():
        base = ref.loc[(row["Dataset"], row["Base"])]
        out = row.to_dict()
        for metric in metrics:
            out[f"Delta_{metric}_vs_M0"] = row[f"{metric}_mean"] - base[f"{metric}_mean"]
        delta_rows.append(out)
    deltas = pd.DataFrame(delta_rows)
    overall = summary.groupby("Variant")[[f"{m}_mean" for m in metrics]].mean().reset_index()
    return summary, deltas, overall


def main():
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
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["M0_confidence", "LabelConstant", "Shared1D", "Label1D", "Shared2D", "Label2D", "Label2D_NoAnchor"],
        choices=[
            "M0_confidence",
            "GlobalIsotonic",
            "PerLabelIsotonic",
            "PerLabelPlatt",
            "LabelConstant",
            "Shared1D",
            "Label1D",
            "Shared2D",
            "Label2D",
            "Label2D_NoAnchor",
        ],
    )
    parser.add_argument("--smooth", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--offset-lambda-l2", type=float, nargs="+", default=[0.0, 0.001, 0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--offset-max-iter", type=int, default=80)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default="results/tables/monotone_reliability_projection")
    args = parser.parse_args()
    seed_everything(args.seed)

    rows = []
    for dataset in args.datasets:
        rows.extend(run_dataset(dataset, args))
    raw = pd.DataFrame(rows)
    summary, deltas, overall = summarize(raw)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "mrp_rows.csv", index=False)
    summary.to_csv(out / "mrp_summary.csv", index=False)
    deltas.to_csv(out / "mrp_deltas.csv", index=False)
    overall.to_csv(out / "mrp_overall.csv", index=False)

    print("\nVariant averages")
    print(
        overall.sort_values("Wrong_AUPRC_mean", ascending=False)[
            [
                "Variant",
                "Binary_NLL_mean",
                "Wrong_AUPRC_mean",
                "AURC_mean",
                "SelAcc10_mean",
                "SelAcc50_mean",
                "SelAcc70_mean",
                "SelAcc90_mean",
            ]
        ]
        .round(5)
        .to_string(index=False)
    )
    print(out / "mrp_summary.csv")
    print(out / "mrp_deltas.csv")


if __name__ == "__main__":
    main()
