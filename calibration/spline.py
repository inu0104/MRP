"""Top-label spline confidence calibration."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from calibration.base import BaseCalibrator
from calibration.ops import softmax


class TopLabelSplineCalibrator(BaseCalibrator):
    """Spline calibrator for the selected top-label confidence.

    The fitted map is scalar: raw top confidence -> probability that the raw
    top-label decision is correct.  At transform time, the mapped confidence is
    assigned to the raw top class and the remaining probability mass is
    distributed over non-top classes in proportion to their original
    probabilities.
    """

    name = "spline"

    def __init__(
        self,
        n_knots: int = 8,
        degree: int = 3,
        c: float = 1.0,
        eps: float = 1e-8,
        max_iter: int = 1000,
    ):
        self.n_knots = int(n_knots)
        self.degree = int(degree)
        self.c = float(c)
        self.eps = float(eps)
        self.max_iter = int(max_iter)
        self.model_ = None
        self.constant_: float | None = None

    @staticmethod
    def _top_arrays(logits: np.ndarray, labels: np.ndarray | None = None):
        probs = softmax(logits).astype(np.float64)
        top = probs.argmax(axis=1).astype(np.int64)
        confidence = probs[np.arange(len(probs)), top]
        if labels is None:
            return probs, top, confidence, None
        labels_np = np.asarray(labels).reshape(-1).astype(np.int64)
        correct = (top == labels_np).astype(np.int64)
        return probs, top, confidence, correct

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TopLabelSplineCalibrator":
        _probs, _top, confidence, correct = self._top_arrays(logits, labels)
        confidence = np.asarray(confidence, dtype=np.float64).reshape(-1, 1)
        correct = np.asarray(correct, dtype=np.int64)
        if len(np.unique(correct)) < 2:
            self.constant_ = float(np.clip(correct.mean(), self.eps, 1.0 - self.eps))
            self.model_ = None
            return self

        n_unique = len(np.unique(confidence[:, 0]))
        n_knots = min(self.n_knots, max(2, n_unique))
        degree = min(self.degree, max(1, n_knots - 1))
        self.model_ = make_pipeline(
            SplineTransformer(n_knots=n_knots, degree=degree, include_bias=False, extrapolation="constant"),
            StandardScaler(with_mean=False),
            LogisticRegression(C=self.c, max_iter=self.max_iter, solver="lbfgs"),
        )
        self.model_.fit(confidence, correct)
        self.constant_ = None
        return self

    def _map_confidence(self, confidence: np.ndarray) -> np.ndarray:
        confidence = np.asarray(confidence, dtype=np.float64).reshape(-1, 1)
        if self.constant_ is not None:
            return np.full(confidence.shape[0], self.constant_, dtype=np.float64)
        if self.model_ is None:
            raise RuntimeError("TopLabelSplineCalibrator must be fit before transform_logits")
        mapped = self.model_.predict_proba(confidence)[:, 1]
        return np.clip(mapped, self.eps, 1.0 - self.eps)

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        probs, top, confidence, _correct = self._top_arrays(logits)
        mapped = self._map_confidence(confidence)
        calibrated = np.array(probs, copy=True, dtype=np.float64)
        rows = np.arange(len(calibrated))
        original_top = np.clip(confidence, self.eps, 1.0 - self.eps)
        scale = (1.0 - mapped) / np.maximum(1.0 - original_top, self.eps)
        calibrated *= scale[:, None]
        calibrated[rows, top] = mapped
        calibrated = np.clip(calibrated, self.eps, 1.0)
        calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)
        return np.log(calibrated).astype(np.float32)
