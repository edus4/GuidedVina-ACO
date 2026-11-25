"""
Paired comparison statistics for docking benchmarks.

Docking is stochastic, so a difference between two configurations on one seed
is not evidence. These helpers treat each complex as the unit of analysis,
pair the arms within complex, and report an interval rather than a bare
percentage difference.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from math import comb
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class PairedComparison:
    n_complexes: int
    baseline_success_rate: Optional[float]
    guided_success_rate: Optional[float]
    difference: Optional[float]
    ci95_low: Optional[float]
    ci95_high: Optional[float]
    guided_only_wins: int
    baseline_only_wins: int
    both: int
    neither: int
    exact_p_two_sided: Optional[float]
    median_runtime_ratio: Optional[float]
    note: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def exact_mcnemar_p(b: int, c: int) -> Optional[float]:
    """
    Two-sided exact McNemar (binomial) test on discordant pairs.

    ``b`` = complexes where only the guided arm succeeded, ``c`` = only the
    baseline. With no discordant pairs there is no evidence either way.
    """
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return float(min(1.0, 2.0 * tail))


def _bootstrap_diff_ci(
    baseline: np.ndarray,
    guided: np.ndarray,
    iterations: int = 10000,
    seed: int = 0,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(baseline)
    idx = rng.integers(0, n, size=(iterations, n))
    diffs = guided[idx].mean(axis=1) - baseline[idx].mean(axis=1)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def compare_arms(
    per_complex: Dict[str, Dict[str, Dict[str, object]]],
    success_rule: str = "majority",
    bootstrap_iterations: int = 10000,
) -> PairedComparison:
    """
    Compare a baseline and a guided arm across complexes.

    ``per_complex`` maps complex_id -> arm -> {"successes": [bool, ...],
    "runtimes": [float, ...]}, i.e. one entry per seed. ``success_rule`` is
    "majority" (succeeds on more than half the seeds) or "any".
    """
    ids = [
        cid for cid, arms in per_complex.items()
        if "baseline" in arms and "guided" in arms
        and arms["baseline"].get("successes") and arms["guided"].get("successes")
    ]
    if not ids:
        return PairedComparison(
            0, None, None, None, None, None, 0, 0, 0, 0, None, None,
            note="no complexes had usable results in both arms",
        )

    def collapse(flags: Sequence[bool]) -> bool:
        flags = [bool(f) for f in flags]
        if not flags:
            return False
        if success_rule == "any":
            return any(flags)
        return sum(flags) > len(flags) / 2.0

    base = np.array([collapse(per_complex[c]["baseline"]["successes"]) for c in ids], dtype=float)
    guided = np.array([collapse(per_complex[c]["guided"]["successes"]) for c in ids], dtype=float)

    b = int(np.sum((guided == 1) & (base == 0)))
    c_ = int(np.sum((guided == 0) & (base == 1)))
    both = int(np.sum((guided == 1) & (base == 1)))
    neither = int(np.sum((guided == 0) & (base == 0)))

    lo, hi = _bootstrap_diff_ci(base, guided, iterations=bootstrap_iterations)

    ratios = []
    for cid in ids:
        rb = [r for r in per_complex[cid]["baseline"].get("runtimes", []) if r and r > 0]
        rg = [r for r in per_complex[cid]["guided"].get("runtimes", []) if r and r > 0]
        if rb and rg:
            ratios.append(float(np.median(rg) / np.median(rb)))

    return PairedComparison(
        n_complexes=len(ids),
        baseline_success_rate=round(float(base.mean()), 4),
        guided_success_rate=round(float(guided.mean()), 4),
        difference=round(float(guided.mean() - base.mean()), 4),
        ci95_low=round(lo, 4),
        ci95_high=round(hi, 4),
        guided_only_wins=b,
        baseline_only_wins=c_,
        both=both,
        neither=neither,
        exact_p_two_sided=exact_mcnemar_p(b, c_),
        median_runtime_ratio=round(float(np.median(ratios)), 4) if ratios else None,
        note=f"success rule: {success_rule}; unit of analysis: complex",
    )
