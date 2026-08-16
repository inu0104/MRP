"""Common calibrator interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class CalibratorProtocol(Protocol):
    """Minimal interface shared by all calibrators."""

    name: str

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "CalibratorProtocol":
        """Fit calibrator parameters on validation logits and labels."""

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        """Return calibrated logits."""

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities."""


@dataclass
class BaseCalibrator:
    """Base class for numpy-facing calibrators."""

    name: str = "base"

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "BaseCalibrator":
        return self

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        return np.asarray(logits, dtype=np.float32)

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        from calibration.ops import softmax

        return softmax(self.transform_logits(logits))

