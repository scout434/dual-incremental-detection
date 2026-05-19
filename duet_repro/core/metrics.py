from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class RaiResult:
    avg_ri: float
    avg_gi: float
    rai: float


def average(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def retention_index(current_map: float, reference_map: float, eps: float = 1e-12) -> float:
    """Old-task retention ratio. Values above 1 are clipped for readable reports."""
    if reference_map <= eps:
        return 0.0
    return min(current_map / reference_map, 1.0)


def generalization_index(current_map: float, reference_map: float, eps: float = 1e-12) -> float:
    """New-domain adaptation ratio."""
    if reference_map <= eps:
        return 0.0
    return min(current_map / reference_map, 1.0)


def rai_from_indices(retention: list[float], generalization: list[float]) -> RaiResult:
    """Compute the Retention-Adaptability Index used for DuIOD comparison.

    The supplementary material defines RAI as the arithmetic mean of Avg RI and
    Avg GI: RAI = (Avg RI + Avg GI) / 2.
    """
    avg_ri = average(retention)
    avg_gi = average(generalization)
    rai = (avg_ri + avg_gi) / 2.0
    return RaiResult(avg_ri=avg_ri, avg_gi=avg_gi, rai=rai)
