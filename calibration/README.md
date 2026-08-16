# Calibration Methods

This folder contains method implementations used by the clean experiment path.
It should not import source files from `.local/external/`.

Current methods:

| File | Method | Role |
| --- | --- | --- |
| `temperature_scaling.py` | TS | Scalar temperature scaling fitted by validation NLL. |
| `isotonic.py` | Iso | Classwise one-vs-rest isotonic calibration with simplex renormalization. |
| `dirichlet.py` | Dirichlet | Affine multiclass calibration over log-probabilities. |
| `diag.py` | DIAG | Diagonal order-preserving logit calibration. |
| `spline.py` | Spline | Top-label spline calibration. |
| `smart.py` | SMART-style | Sample-adaptive temperature from top-2 logit gap. |
| `h_calibration.py` | h-cal-style | Piecewise monotonic h-calibration-style calibrator. |

The post-calibration reliability reranking code lives under
`experiments/table/` because it is tied to the paper evaluation protocol rather
than to a probability-changing calibrator API.
