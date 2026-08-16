"""Self-contained h-calibration-style calibrator.

The public h-Calibration release is kept in `.local/external/` for reference and
reproduction. This module provides a local implementation so experiments do
not have to import from that external source tree.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from calibration.base import BaseCalibrator


class PiecewiseLinearMonotonic(nn.Module):
    """Order-preserving piecewise-linear transform over log probabilities."""

    def __init__(self, segments: int = 100, logit_range: float = 100.0):
        super().__init__()
        self.logit_range = float(logit_range)
        self.delta = nn.Parameter(torch.ones(1) / segments * self.logit_range, requires_grad=False)
        steps = torch.linspace(start=0, end=self.logit_range, steps=segments + 1)
        self.steps = nn.Parameter(steps, requires_grad=False)
        self.slopes = nn.Parameter(torch.ones(segments + 1), requires_grad=True)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        v1 = (((-values >= self.steps[None, :]) * torch.abs(self.slopes[None, :])).sum(1) - 1) * self.delta
        v2 = (((-values + self.delta >= self.steps[None, :]) * torch.abs(self.slopes[None, :])).sum(1) - 1) * self.delta
        out = v1 + (v2 - v1) / self.delta * (-values.flatten() - v1)
        return -out


class HCalMonotonicModel(nn.Module):
    """Monotonic transformation applied elementwise to log-softmax logits."""

    def __init__(self, segments: int = 100):
        super().__init__()
        self.model_monotonic = PiecewiseLinearMonotonic(segments)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        flat = logits.contiguous().view(-1).unsqueeze(1)
        return self.model_monotonic(flat).contiguous().view(logits.shape)


def hcalibration_loss(
    probs: torch.Tensor,
    labels: torch.Tensor,
    *,
    epsilon: float = 1e-20,
    window: int = 200,
    num_classes: int | None = None,
    loss_weight: float = 1e5,
) -> torch.Tensor:
    """Event-wise h-calibration loss with convolution smoothing.

    This captures the core probability-occurrence residual used by h-cal:
    sort event probabilities, smooth local occurrence/non-occurrence terms,
    and penalize local calibration deviations above epsilon.
    """

    if num_classes is None:
        num_classes = probs.shape[1]
    targets = F.one_hot(labels.long(), num_classes=num_classes).float()
    losses = []
    for event_id in range(num_classes):
        keep = labels.long() != event_id
        if keep.sum() < 2:
            continue
        prob_mat = probs[keep]
        target_mat = targets[keep]
        prob_evt = prob_mat.flatten()
        occur = target_mat.flatten()
        order = torch.argsort(torch.log(prob_evt + 1e-20))
        prob_evt = prob_evt[order]
        occur = occur[order]
        inv_prob_evt = 1.0 - prob_evt
        prob_not_occ = prob_evt * (1.0 - occur)
        inv_prob_occ = inv_prob_evt * occur
        pad = int(window / 2)
        kernel = torch.ones((1, 1, window + 1), device=probs.device) / (window + 1)
        prob_not_occ = F.pad(prob_not_occ, pad=(pad, pad), mode="constant", value=0)
        inv_prob_occ = F.pad(inv_prob_occ, pad=(pad, pad), mode="constant", value=0)
        lhs = F.conv1d(prob_not_occ[None, None, :].float(), kernel).flatten()
        rhs = F.conv1d(inv_prob_occ[None, None, :].float(), kernel).flatten()
        local_error = torch.abs(lhs - rhs)
        losses.append(F.relu(local_error - epsilon).mean())
    if not losses:
        return probs.new_tensor(0.0)
    return torch.stack(losses).mean() * loss_weight


class HCalibrator(BaseCalibrator):
    """Piecewise-monotonic h-calibration-style calibrator."""

    name = "hcal"

    def __init__(
        self,
        segments: int = 100,
        epochs: int = 250,
        patience: int = 50,
        lr: float = 0.005,
        batch_size: int = 5000,
        epsilon: float = 1e-20,
        window: int = 200,
        loss_weight: float = 1e5,
        device: str | None = None,
    ):
        self.segments = segments
        self.epochs = epochs
        self.patience = patience
        self.lr = lr
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.window = window
        self.loss_weight = loss_weight
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_: HCalMonotonicModel | None = None
        self.history_: list[float] = []

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "HCalibrator":
        logits_np = np.asarray(logits, dtype=np.float32)
        labels_np = np.asarray(labels).reshape(-1).astype(np.int64)
        num_classes = logits_np.shape[1]
        dataset = TensorDataset(torch.tensor(logits_np), torch.tensor(labels_np, dtype=torch.long))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        model = HCalMonotonicModel(self.segments).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        eval_logits = torch.tensor(logits_np, dtype=torch.float32, device=self.device)
        eval_labels = torch.tensor(labels_np, dtype=torch.long, device=self.device)
        best_loss = float("inf")
        best_state = None
        stale = 0
        self.history_ = []
        for _epoch in range(1, self.epochs + 1):
            model.train()
            for batch_logits, batch_labels in loader:
                batch_logits = batch_logits.to(self.device)
                batch_labels = batch_labels.to(self.device)
                cal_probs = F.softmax(model(batch_logits), dim=1)
                loss = hcalibration_loss(
                    cal_probs,
                    batch_labels,
                    epsilon=self.epsilon,
                    window=self.window,
                    num_classes=num_classes,
                    loss_weight=self.loss_weight,
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                eval_probs = F.softmax(model(eval_logits), dim=1)
                eval_loss = hcalibration_loss(
                    eval_probs,
                    eval_labels,
                    epsilon=self.epsilon,
                    window=self.window,
                    num_classes=num_classes,
                    loss_weight=self.loss_weight,
                )
            value = float(eval_loss.detach().cpu())
            self.history_.append(value)
            if value < best_loss:
                best_loss = value
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if self.patience > 0 and stale >= self.patience:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        self.model_ = model
        return self

    @torch.no_grad()
    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("HCalibrator must be fit before transform_logits")
        logits_t = torch.tensor(np.asarray(logits), dtype=torch.float32, device=self.device)
        self.model_.eval()
        return self.model_(logits_t).detach().cpu().numpy().astype(np.float32)

