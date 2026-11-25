"""
Search-budget selection.

Two selectors are provided and both report which one produced a plan, so a
benchmark table can never present a hand-tuned default as a learned one:

* ``HeuristicSelector`` -- a documented piecewise prior over ligand
  flexibility and pocket volume. These are *priors*, not fitted values. They
  exist so the pipeline runs before any training data has been collected.
* ``TrainedSelector`` -- loads a fitted regressor exported by
  ``scripts/train_selector.py`` and predicts the cheapest configuration whose
  predicted top-1 success stays within a tolerance of the best configuration.

Both return the same ``SearchPlan``, so the engine is indifferent to which
was used.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .features import LigandFeatures, PocketFeatures


@dataclass
class SearchPlan:
    regime: str
    selector: str                  # "heuristic" | "trained:<name>"
    exhaustiveness: int
    num_modes: int
    max_evals: Optional[int]
    # Bias schedule parameters (consumed by guidedvina.bias)
    stages: int
    initial_bias_weight: float
    annealing_tau: float
    anchor_radius: float
    diversity_floor_angstrom: float
    notes: List[str]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# --- heuristic prior -------------------------------------------------------
# Thresholds follow the conventional flexibility bands used when reporting
# docking accuracy (<=3, 4-8, >8 rotatable bonds); the budgets are starting
# points that scale with box volume relative to a 20x20x20 A reference box.
_REFERENCE_BOX_VOLUME = 8000.0

_BANDS = (
    # (max_rb, regime, exhaustiveness, max_evals, stages, w0, tau, anchor_radius)
    (3, "rigid", 8, None, 2, 0.45, 0.35, 4.0),
    (8, "moderate", 16, None, 3, 0.65, 0.45, 5.0),
    (10 ** 9, "flexible", 32, None, 3, 0.80, 0.60, 6.0),
)


class HeuristicSelector:
    name = "heuristic"

    def __init__(self, volume_scaling: bool = True) -> None:
        self.volume_scaling = volume_scaling

    def plan(self, ligand: LigandFeatures, pocket: PocketFeatures) -> SearchPlan:
        rb = ligand.rotatable_bonds
        band = next(b for b in _BANDS if rb <= b[0])
        _, regime, exh, max_evals, stages, w0, tau, radius = band

        notes: List[str] = [
            "Budgets are documented priors, not fitted values; "
            "run scripts/train_selector.py to replace them."
        ]
        if self.volume_scaling and pocket.box_volume > 0:
            factor = float(
                np.clip((pocket.box_volume / _REFERENCE_BOX_VOLUME) ** 0.25, 0.85, 1.35)
            )
            exh = int(max(8, round(exh * factor)))
            notes.append(f"exhaustiveness scaled by box-volume factor {factor:.2f}")
        if ligand.is_estimated:
            notes.append(
                "rotatable_bonds was ESTIMATED, so the regime choice is unreliable."
            )
        # A tight, well-enclosed pocket needs a smaller bias radius than a
        # shallow surface site; clamp to the ligand's own size.
        radius = float(min(radius, max(2.5, ligand.max_radius)))

        return SearchPlan(
            regime=f"{regime}_prior",
            selector=self.name,
            exhaustiveness=exh,
            num_modes=9,
            max_evals=max_evals,
            stages=stages,
            initial_bias_weight=w0,
            annealing_tau=tau,
            anchor_radius=radius,
            diversity_floor_angstrom=1.2,
            notes=notes,
        )


# --- trained selector ------------------------------------------------------
FEATURE_ORDER = (
    "rotatable_bonds",
    "heavy_atoms",
    "max_radius",
    "box_volume",
    "heavy_atoms_in_box",
    "polar_atoms_in_box",
    "buriedness",
)


def feature_vector(ligand: LigandFeatures, pocket: PocketFeatures) -> np.ndarray:
    values = {
        "rotatable_bonds": ligand.rotatable_bonds,
        "heavy_atoms": ligand.heavy_atoms,
        "max_radius": ligand.max_radius,
        "box_volume": pocket.box_volume,
        "heavy_atoms_in_box": pocket.heavy_atoms_in_box,
        "polar_atoms_in_box": pocket.polar_atoms_in_box,
        "buriedness": pocket.buriedness,
    }
    return np.array([float(values[k]) for k in FEATURE_ORDER], dtype=float)


class TrainedSelector:
    """
    Wraps a model exported as JSON by scripts/train_selector.py.

    The export records the candidate configuration grid and, for each
    configuration, a linear model over FEATURE_ORDER predicting P(top-1 <= 2 A)
    and expected runtime. The selector picks the cheapest configuration whose
    predicted success is within ``tolerance`` of the best predicted success.
    """

    def __init__(self, model_path: Path, tolerance: float = 0.02) -> None:
        self.model_path = Path(model_path)
        with open(self.model_path, "r", encoding="utf-8") as fh:
            self.model = json.load(fh)
        if self.model.get("feature_order") != list(FEATURE_ORDER):
            raise ValueError(
                f"Model {model_path} was trained on a different feature order; retrain it."
            )
        self.tolerance = tolerance
        self.name = f"trained:{self.model.get('name', self.model_path.stem)}"

    def _predict(self, coefs: Sequence[float], intercept: float, x: np.ndarray) -> float:
        z = float(np.dot(np.asarray(coefs, dtype=float), x) + intercept)
        return 1.0 / (1.0 + np.exp(-z))

    def plan(self, ligand: LigandFeatures, pocket: PocketFeatures) -> SearchPlan:
        x = feature_vector(ligand, pocket)
        scored = []
        for cfg in self.model["configurations"]:
            p = self._predict(cfg["success_coefs"], cfg["success_intercept"], x)
            cost = float(np.dot(cfg["runtime_coefs"], x) + cfg["runtime_intercept"])
            scored.append((p, cost, cfg))
        best_p = max(s[0] for s in scored)
        affordable = [s for s in scored if s[0] >= best_p - self.tolerance]
        affordable.sort(key=lambda s: s[1])
        p, cost, cfg = affordable[0]
        return SearchPlan(
            regime=cfg.get("regime", "trained"),
            selector=self.name,
            exhaustiveness=int(cfg["exhaustiveness"]),
            num_modes=int(cfg.get("num_modes", 9)),
            max_evals=cfg.get("max_evals"),
            stages=int(cfg.get("stages", 3)),
            initial_bias_weight=float(cfg["initial_bias_weight"]),
            annealing_tau=float(cfg.get("annealing_tau", 0.45)),
            anchor_radius=float(cfg.get("anchor_radius", 5.0)),
            diversity_floor_angstrom=float(cfg.get("diversity_floor_angstrom", 1.2)),
            notes=[
                f"predicted P(top1<=2A)={p:.3f}, predicted runtime={cost:.1f}s",
                f"selected from {len(scored)} candidate configurations",
            ],
        )


def baseline_plan(num_modes: int = 9, exhaustiveness: int = 8) -> SearchPlan:
    """The unbiased control arm: engine defaults, no bias, single stage."""
    return SearchPlan(
        regime="baseline_default",
        selector="none",
        exhaustiveness=exhaustiveness,
        num_modes=num_modes,
        max_evals=None,
        stages=1,
        initial_bias_weight=0.0,
        annealing_tau=1.0,
        anchor_radius=0.0,
        diversity_floor_angstrom=0.0,
        notes=["control arm: engine defaults, no bias applied"],
    )


def get_selector(model_path: Optional[Path] = None) -> object:
    if model_path:
        return TrainedSelector(model_path)
    return HeuristicSelector()
