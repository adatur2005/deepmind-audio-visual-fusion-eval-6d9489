"""Choosing an operating point.

A classifier outputs a score. Turning that into a decision requires a
threshold, and 0.5 is not a natural choice — it is the midpoint of an
arbitrary output range, and nothing about it corresponds to the costs of the
two errors.

The two errors here are not remotely equal:

  FALSE POSITIVE — a real clip flagged as synthetic. Someone's genuine video
                   is labelled a deepfake. If this feeds any enforcement
                   action, a person is accused of something they did not do.

  FALSE NEGATIVE — a synthetic clip passed as real. An impersonation is not
                   caught, which is the harm the system exists to prevent.

Which is worse depends entirely on what happens downstream, and that is a
deployment decision rather than a modelling one. This module makes the choice
explicit instead of inheriting it from a default.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    threshold: float
    precision: float
    recall: float
    false_positive_rate: float

    def describe(self) -> str:
        return (
            f"threshold {self.threshold:.3f}: "
            f"precision {self.precision:.3f}, recall {self.recall:.3f}, "
            f"FPR {self.false_positive_rate:.4f}"
        )


def sweep(scores: np.ndarray, labels: np.ndarray,
          steps: int = 200) -> list[OperatingPoint]:
    """Evaluate every threshold across the score range."""
    points: list[OperatingPoint] = []

    for threshold in np.linspace(scores.min(), scores.max(), steps):
        predicted = scores >= threshold

        tp = int(np.sum(predicted & (labels == 1)))
        fp = int(np.sum(predicted & (labels == 0)))
        fn = int(np.sum(~predicted & (labels == 1)))
        tn = int(np.sum(~predicted & (labels == 0)))

        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0

        points.append(
            OperatingPoint(float(threshold), precision, recall, fpr)
        )

    return points


def select_for_max_fpr(points: list[OperatingPoint],
                       max_fpr: float) -> OperatingPoint:
    """Pick the highest-recall point whose false positive rate stays under a cap.

    This encodes a specific deployment stance: a bounded rate of falsely
    accusing real videos, and as much detection as possible within it.

    The cap is the input, not something derived from the data, because it comes
    from a policy question the model cannot answer — how many wrongly flagged
    real clips per thousand is acceptable. A team that cannot answer that has
    not decided what the system is for, and picking a threshold before then is
    choosing a policy by accident.
    """
    admissible = [p for p in points if p.false_positive_rate <= max_fpr]

    if not admissible:
        # Every threshold exceeds the cap. Returning the least-bad point with a
        # quiet warning would hide that the model cannot meet the requirement
        # at all, so this raises instead.
        raise ValueError(
            f"no threshold achieves FPR <= {max_fpr}; "
            f"minimum achievable is {min(p.false_positive_rate for p in points):.4f}"
        )

    return max(admissible, key=lambda p: p.recall)


def report_curve(points: list[OperatingPoint]) -> str:
    """Print several operating points, not just the chosen one.

    A single reported number invites the reader to treat it as THE performance
    of the model. Showing the curve makes the tradeoff visible and lets someone
    with different costs read off their own point rather than asking for a
    rerun.
    """
    lines = ["threshold  precision  recall     FPR"]
    for target_fpr in (0.001, 0.005, 0.01, 0.05, 0.10):
        candidates = [p for p in points if p.false_positive_rate <= target_fpr]
        if not candidates:
            lines.append(f"  (FPR <= {target_fpr}: unreachable)")
            continue
        best = max(candidates, key=lambda p: p.recall)
        lines.append(
            f"{best.threshold:9.3f}  {best.precision:9.3f}  "
            f"{best.recall:9.3f}  {best.false_positive_rate:.4f}"
        )
    return "\n".join(lines)
