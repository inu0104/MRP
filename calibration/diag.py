"""Decision-preserving diagonal order calibration."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from calibration.base import BaseCalibrator


def _deterministic_cumsum(values: torch.Tensor, dim: int) -> torch.Tensor:
    pieces = []
    running = None
    for piece in torch.unbind(values, dim=dim):
        running = piece if running is None else running + piece
        pieces.append(running)
    return torch.stack(pieces, dim=dim)


class DiagonalOrderPreservingCalibrator(BaseCalibrator):
    """Rank-gap diagonal calibrator that preserves each sample's class order.

    The calibrator sorts each logit vector, rescales adjacent rank gaps by
    positive learnable factors, and then unsorts the result.  Positive gap
    scales guarantee that the transformed logits keep the original within-sample
    ranking.  This gives a lightweight DIAG-style order-preserving baseline for
    fixed-decision calibration experiments.
    """

    name = "diag"

    def __init__(
        self,
        max_iter: int = 120,
        lr: float = 0.1,
        weight_decay: float = 1e-3,
        device: str | None = None,
    ):
        self.max_iter = int(max_iter)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gap_scales_: np.ndarray | None = None

    @staticmethod
    def _sorted_gap_transform(logits: torch.Tensor, raw_gap_scales: torch.Tensor) -> torch.Tensor:
        sorted_logits, order = torch.sort(logits, dim=1, descending=True)
        gaps = sorted_logits[:, :-1] - sorted_logits[:, 1:]
        scales = F.softplus(raw_gap_scales).clamp_min(1e-6)
        calibrated_gaps = gaps * scales[None, :]
        sorted_out = torch.zeros_like(sorted_logits)
        sorted_out[:, 1:] = -_deterministic_cumsum(calibrated_gaps, dim=1)
        # Centering removes an irrelevant additive degree of freedom.
        sorted_out = sorted_out - sorted_out.mean(dim=1, keepdim=True)
        out = torch.empty_like(sorted_out)
        out.scatter_(1, order, sorted_out)
        return out

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "DiagonalOrderPreservingCalibrator":
        logits_np = np.asarray(logits, dtype=np.float32)
        labels_np = np.asarray(labels).reshape(-1).astype(np.int64)
        if logits_np.ndim != 2:
            raise ValueError(f"Expected logits with shape (n, classes), got {logits_np.shape}")
        if logits_np.shape[1] < 2:
            raise ValueError("Need at least two classes for DIAG calibration")

        logits_t = torch.tensor(logits_np, dtype=torch.float32, device=self.device)
        labels_t = torch.tensor(labels_np, dtype=torch.long, device=self.device)
        raw_scales = torch.zeros(logits_np.shape[1] - 1, dtype=torch.float32, device=self.device, requires_grad=True)
        opt = torch.optim.LBFGS([raw_scales], lr=self.lr, max_iter=self.max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            out = self._sorted_gap_transform(logits_t, raw_scales)
            scales = F.softplus(raw_scales)
            loss = F.cross_entropy(out, labels_t) + self.weight_decay * (torch.log(scales).pow(2).mean())
            loss.backward()
            return loss

        opt.step(closure)
        self.gap_scales_ = F.softplus(raw_scales).detach().cpu().numpy().astype(np.float32)
        return self

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        if self.gap_scales_ is None:
            raise RuntimeError("DiagonalOrderPreservingCalibrator must be fit before transform_logits")
        logits_np = np.asarray(logits, dtype=np.float32)
        if logits_np.ndim != 2:
            raise ValueError(f"Expected logits with shape (n, classes), got {logits_np.shape}")
        sorted_logits = np.sort(logits_np, axis=1)[:, ::-1]
        order = np.argsort(-logits_np, axis=1, kind="stable")
        gaps = sorted_logits[:, :-1] - sorted_logits[:, 1:]
        calibrated_gaps = gaps * self.gap_scales_[None, :]
        sorted_out = np.zeros_like(sorted_logits, dtype=np.float32)
        sorted_out[:, 1:] = -np.cumsum(calibrated_gaps, axis=1)
        sorted_out = sorted_out - sorted_out.mean(axis=1, keepdims=True)
        out = np.empty_like(sorted_out)
        rows = np.arange(len(logits_np))[:, None]
        out[rows, order] = sorted_out
        return out.astype(np.float32)
