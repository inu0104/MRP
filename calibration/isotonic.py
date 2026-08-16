"""Classwise isotonic calibration."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

from calibration.base import BaseCalibrator


class IsotonicCalibrator(BaseCalibrator):
    """One-vs-rest isotonic calibration with probability renormalization.

    For multiclass predictions, each class probability is calibrated by an
    independent isotonic regression model trained against the one-vs-rest label.
    The calibrated class scores are then clipped and renormalized back onto the
    probability simplex. ``transform_logits`` returns log-probabilities so it
    can be consumed by the same softmax-based experiment code as other local
    calibrators.
    """

    name = "isotonic"

    def __init__(self, eps: float = 1e-12):
        self.eps = eps
        self.models_: list[IsotonicRegression] | None = None
        self.priors_: np.ndarray | None = None

    @staticmethod
    def _probs(logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float32)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return (exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        probs = self._probs(logits)
        labels_np = np.asarray(labels).reshape(-1).astype(np.int64)
        num_classes = probs.shape[1]
        self.priors_ = np.asarray([(labels_np == cls).mean() for cls in range(num_classes)], dtype=np.float32)
        models = []
        for cls in range(num_classes):
            y = (labels_np == cls).astype(np.float32)
            model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            model.fit(probs[:, cls], y)
            models.append(model)
        self.models_ = models
        return self

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        if self.models_ is None or self.priors_ is None:
            raise RuntimeError("IsotonicCalibrator must be fit before transform_logits")
        probs = self._probs(logits)
        calibrated = np.stack([model.predict(probs[:, cls]) for cls, model in enumerate(self.models_)], axis=1)
        calibrated = np.asarray(calibrated, dtype=np.float32)
        calibrated = np.nan_to_num(calibrated, nan=0.0, posinf=1.0, neginf=0.0)
        calibrated = np.clip(calibrated, self.eps, 1.0)
        denom = calibrated.sum(axis=1, keepdims=True)
        bad = denom[:, 0] <= self.eps
        if np.any(bad):
            calibrated[bad] = self.priors_[None, :]
            denom = calibrated.sum(axis=1, keepdims=True)
        calibrated = calibrated / np.maximum(denom, self.eps)
        return np.log(np.clip(calibrated, self.eps, 1.0)).astype(np.float32)
