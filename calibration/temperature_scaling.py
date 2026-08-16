"""Temperature scaling."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from calibration.base import BaseCalibrator


class TemperatureScaling(BaseCalibrator):
    """Scalar temperature scaling fitted by validation NLL."""

    name = "temperature_scaling"

    def __init__(self, max_iter: int = 80, lr: float = 0.2, device: str | None = None):
        self.max_iter = max_iter
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature_: float = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaling":
        logits_t = torch.tensor(np.asarray(logits), dtype=torch.float32, device=self.device)
        labels_t = torch.tensor(np.asarray(labels).reshape(-1), dtype=torch.long, device=self.device)
        log_t = torch.zeros((), device=self.device, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=self.lr, max_iter=self.max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(logits_t / torch.exp(log_t), labels_t)
            loss.backward()
            return loss

        opt.step(closure)
        self.temperature_ = float(torch.exp(log_t).detach().cpu().item())
        return self

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        return (np.asarray(logits, dtype=np.float32) / max(self.temperature_, 1e-8)).astype(np.float32)

