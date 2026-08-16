"""Method registry for stable experiment scripts."""

from __future__ import annotations

from calibration.diag import DiagonalOrderPreservingCalibrator
from calibration.dirichlet import DirichletCalibrator
from calibration.h_calibration import HCalibrator
from calibration.isotonic import IsotonicCalibrator
from calibration.smart import SMARTCalibrator
from calibration.spline import TopLabelSplineCalibrator
from calibration.temperature_scaling import TemperatureScaling


CALIBRATORS = {
    "dirichlet": DirichletCalibrator,
    "dir": DirichletCalibrator,
    "diag": DiagonalOrderPreservingCalibrator,
    "dia": DiagonalOrderPreservingCalibrator,
    "isotonic": IsotonicCalibrator,
    "iso": IsotonicCalibrator,
    "spline": TopLabelSplineCalibrator,
    "top_spline": TopLabelSplineCalibrator,
    "ts": TemperatureScaling,
    "temperature_scaling": TemperatureScaling,
    "smart": SMARTCalibrator,
    "hcal": HCalibrator,
    "h-cal": HCalibrator,
}


def make_calibrator(name: str, **kwargs):
    key = name.lower()
    if key not in CALIBRATORS:
        raise ValueError(f"Unknown calibrator {name}. Options: {sorted(CALIBRATORS)}")
    return CALIBRATORS[key](**kwargs)
