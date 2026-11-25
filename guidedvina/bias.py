"""
Bias emission and the staged annealing schedule.

This module exists because a bias weight that is only reported in a JSON
record has no effect on docking. Everything here ends up either in a bias
parameter file handed to the engine or in a command-line argument.

Bias parameter file format (AutoDock-Bias / Vina-family ``--bias`` input)::

    x y z Vset r type
    12.500 -4.200 18.000 -1.80 4.00 map

``Vset`` is the energy offset in kcal/mol applied inside radius ``r`` of the
point; negative values favour the region. ``type`` is ``map`` for a
type-agnostic potential well, or ``don``/``acc``/``aro`` to bias only
H-bond-donor, acceptor or aromatic ligand atoms.

Annealing is implemented as a sequence of docking stages with decreasing bias,
because the Vina-family engines expose no hook to vary bias within a run. The
final stage always runs with zero bias so that the reported pose and score are
not contaminated by the bias potential.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .planner import SearchPlan
from .structures import read_models

BIAS_TYPES = ("map", "don", "acc", "aro")
#: Energy offset, in kcal/mol, applied at bias weight 1.0.
MAX_BIAS_ENERGY = 2.0


@dataclass(frozen=True)
class BiasPoint:
    x: float
    y: float
    z: float
    energy: float          # Vset, kcal/mol (negative = attractive)
    radius: float
    bias_type: str = "map"

    def to_line(self) -> str:
        if self.bias_type not in BIAS_TYPES:
            raise ValueError(f"Unknown bias type {self.bias_type!r}; expected one of {BIAS_TYPES}")
        return (
            f"{self.x:8.3f} {self.y:8.3f} {self.z:8.3f} "
            f"{self.energy:8.2f} {self.radius:6.2f} {self.bias_type}"
        )


def write_bias_file(points: Sequence[BiasPoint], path: Path) -> Path:
    """Write a bias parameter file. Returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("x y z Vset r type\n")
        for p in points:
            fh.write(p.to_line() + "\n")
    return path


def anchor_bias_points(
    anchor: Sequence[float],
    weight: float,
    radius: float,
    bias_type: str = "map",
    max_energy: float = MAX_BIAS_ENERGY,
) -> List[BiasPoint]:
    """
    One attractive well at the anchor, scaled by ``weight`` in [0, 1].

    A weight of 0 yields no points at all, which is how the unbiased control
    arm and the final annealing stage are expressed.
    """
    if weight <= 0.0:
        return []
    energy = -abs(weight) * max_energy
    return [
        BiasPoint(
            x=float(anchor[0]),
            y=float(anchor[1]),
            z=float(anchor[2]),
            energy=energy,
            radius=float(radius),
            bias_type=bias_type,
        )
    ]


# --- pose diversity --------------------------------------------------------
def pose_diversity(poses: Sequence[np.ndarray]) -> float:
    """
    Mean pairwise heavy-atom RMSD across docked poses, in Angstrom.

    This is measured from the engine's own multi-model output, not assumed.
    Returns 0.0 when fewer than two comparable poses are available, which the
    guardrail treats as full collapse.
    """
    usable = [p for p in poses if p is not None and len(p) > 0]
    if len(usable) < 2:
        return 0.0
    n_atoms = usable[0].shape[0]
    usable = [p for p in usable if p.shape[0] == n_atoms]
    if len(usable) < 2:
        return 0.0
    vals = []
    for a, b in itertools.combinations(usable, 2):
        diff = a - b
        vals.append(float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))))
    return float(np.mean(vals)) if vals else 0.0


def pose_diversity_from_file(path: Path) -> float:
    try:
        models = read_models(Path(path))
    except Exception:
        return 0.0
    return pose_diversity([m.coords(heavy_only=True) for m in models])


# --- annealing schedule ----------------------------------------------------
@dataclass
class Stage:
    index: int
    bias_weight: float
    exhaustiveness: int
    local_only: bool
    measured_diversity: Optional[float] = None
    guardrail_applied: bool = False

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def nominal_weight(stage_index: int, n_stages: int, plan: SearchPlan) -> float:
    """
    Exponential decay across stages, with the final stage forced to zero.

    ``progress`` runs 0 -> 1 across the stages, so the decay constant
    ``annealing_tau`` is expressed in units of total schedule length and is
    independent of how many stages are run.
    """
    if n_stages <= 1:
        return 0.0
    if stage_index >= n_stages - 1:
        return 0.0
    progress = stage_index / float(n_stages - 1)
    tau = max(1e-6, plan.annealing_tau)
    return float(plan.initial_bias_weight * math.exp(-progress / tau))


def apply_diversity_guardrail(
    weight: float,
    measured_diversity: float,
    floor: float,
) -> Tuple[float, bool]:
    """
    Cut the bias further when the previous stage's poses collapsed.

    ``measured_diversity`` is the mean pairwise RMSD actually observed in the
    previous stage's output. Below ``floor`` the bias is attenuated by the
    squared ratio, reaching zero at zero diversity.
    """
    if floor <= 0.0 or measured_diversity >= floor:
        return weight, False
    ratio = max(0.0, measured_diversity / floor)
    return weight * (ratio ** 2), True


def build_schedule(plan: SearchPlan) -> List[Stage]:
    """
    The planned stages before any diversity measurement.

    Stage 0 is a global search; intermediate stages are global searches with
    reduced bias; the last stage is an unbiased local refinement when the
    engine supports it, so the reported score is bias-free.
    """
    n = max(1, int(plan.stages))
    stages: List[Stage] = []
    for i in range(n):
        is_last = i == n - 1
        stages.append(
            Stage(
                index=i,
                bias_weight=nominal_weight(i, n, plan),
                # Later stages refine rather than explore, so they get half the
                # search breadth, never more than stage 0. Under ACO this is a
                # colony count, where 1 is a legitimate value.
                exhaustiveness=(
                    plan.exhaustiveness if i == 0
                    else max(1, min(plan.exhaustiveness,
                                    int(round(plan.exhaustiveness / 2))))
                ),
                local_only=bool(is_last and n > 1),
            )
        )
    return stages
