"""SMART-style sample-adaptive temperature scaling."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from calibration.base import BaseCalibrator


class _GapToTemperature(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        layers = [nn.Linear(1, hidden_dim)]
        for _ in range(max(num_layers - 1, 0)):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.hidden = nn.ModuleList(layers)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, gaps: torch.Tensor) -> torch.Tensor:
        x = gaps.view(-1, 1)
        for layer in self.hidden:
            x = F.relu(layer(x))
        return F.softplus(self.out(x)).view(-1) + 0.1


def _logit_gap(logits: torch.Tensor) -> torch.Tensor:
    top2 = torch.topk(logits, k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


class SMARTCalibrator(BaseCalibrator):
    """Logit-gap-conditioned temperature scaler.

    This is a self-contained implementation of the SMART-style idea used in
    our pilots: predict a per-sample temperature from the top-2 logit gap.
    """

    name = "smart"

    def __init__(
        self,
        hidden_dim: int = 16,
        num_layers: int = 2,
        epochs: int = 300,
        patience: int = 50,
        lr: float = 0.005,
        min_delta: float = 1e-7,
        device: str | None = None,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.patience = patience
        self.lr = lr
        self.min_delta = min_delta
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_: _GapToTemperature | None = None
        self.gap_mean_: torch.Tensor | None = None
        self.gap_std_: torch.Tensor | None = None
        self.history_: list[float] = []

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "SMARTCalibrator":
        logits_t = torch.tensor(np.asarray(logits), dtype=torch.float32, device=self.device)
        labels_t = torch.tensor(np.asarray(labels).reshape(-1), dtype=torch.long, device=self.device)
        gaps = _logit_gap(logits_t)
        self.gap_mean_ = gaps.mean().detach().cpu()
        self.gap_std_ = gaps.std().clamp_min(1e-8).detach().cpu()
        norm_gaps = (gaps - self.gap_mean_.to(self.device)) / self.gap_std_.to(self.device)
        model = _GapToTemperature(self.hidden_dim, self.num_layers).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        best_loss = float("inf")
        best_state = None
        stale = 0
        self.history_ = []
        for _epoch in range(1, self.epochs + 1):
            model.train()
            temps = model(norm_gaps)
            loss = F.cross_entropy(logits_t / temps.view(-1, 1), labels_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            value = float(loss.detach().cpu())
            self.history_.append(value)
            if value < best_loss - self.min_delta:
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
        if self.model_ is None or self.gap_mean_ is None or self.gap_std_ is None:
            raise RuntimeError("SMARTCalibrator must be fit before transform_logits")
        logits_t = torch.tensor(np.asarray(logits), dtype=torch.float32, device=self.device)
        gaps = _logit_gap(logits_t)
        norm_gaps = (gaps - self.gap_mean_.to(self.device)) / self.gap_std_.to(self.device).clamp_min(1e-8)
        self.model_.eval()
        temps = self.model_(norm_gaps)
        return (logits_t / temps.view(-1, 1)).detach().cpu().numpy().astype(np.float32)

