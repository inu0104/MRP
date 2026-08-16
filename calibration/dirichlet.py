"""Dirichlet-style multiclass calibration."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from calibration.base import BaseCalibrator


class DirichletCalibrator(BaseCalibrator):
    """Affine calibration over log-probabilities.

    This lightweight implementation follows the practical Dirichlet-calibration
    form used for multiclass post-hoc calibration: first map logits to
    log-probabilities, then learn an affine transform before softmax.
    """

    name = "dirichlet"

    def __init__(
        self,
        max_iter: int = 200,
        lr: float = 0.1,
        weight_decay: float = 1e-3,
        device: str | None = None,
    ):
        self.max_iter = max_iter
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.weight_: np.ndarray | None = None
        self.bias_: np.ndarray | None = None
        self.history_: list[float] = []

    @staticmethod
    def _log_probs(logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float32)
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_norm = np.log(np.exp(shifted).sum(axis=1, keepdims=True) + 1e-12)
        return (shifted - log_norm).astype(np.float32)

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "DirichletCalibrator":
        log_probs_np = self._log_probs(logits)
        labels_np = np.asarray(labels).reshape(-1).astype(np.int64)
        num_classes = log_probs_np.shape[1]

        log_probs = torch.tensor(log_probs_np, dtype=torch.float32, device=self.device)
        labels_t = torch.tensor(labels_np, dtype=torch.long, device=self.device)
        identity = torch.eye(num_classes, dtype=torch.float32, device=self.device)
        weight = identity.clone().requires_grad_(True)
        bias = torch.zeros(num_classes, dtype=torch.float32, device=self.device, requires_grad=True)
        opt = torch.optim.LBFGS([weight, bias], lr=self.lr, max_iter=self.max_iter, line_search_fn="strong_wolfe")
        self.history_ = []

        def closure():
            opt.zero_grad()
            out = log_probs @ weight + bias
            reg = ((weight - identity) ** 2).mean() + (bias**2).mean()
            loss = F.cross_entropy(out, labels_t) + self.weight_decay * reg
            loss.backward()
            self.history_.append(float(loss.detach().cpu()))
            return loss

        opt.step(closure)
        self.weight_ = weight.detach().cpu().numpy().astype(np.float32)
        self.bias_ = bias.detach().cpu().numpy().astype(np.float32)
        return self

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        if self.weight_ is None or self.bias_ is None:
            raise RuntimeError("DirichletCalibrator must be fit before transform_logits")
        log_probs = self._log_probs(logits)
        return (log_probs @ self.weight_ + self.bias_[None, :]).astype(np.float32)
