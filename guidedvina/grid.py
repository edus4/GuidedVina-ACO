"""
The configuration grid used to collect selector training data.

The selector in planner.TrainedSelector is only as good as the sweep behind
it: for each complex every configuration in this grid is run, so that the
model learns which configuration is the cheapest one that still succeeds.
"""
from __future__ import annotations

from typing import Dict, List

from .planner import SearchPlan

#: (aco_preset, colonies, stages, initial_bias_weight, annealing_tau, anchor_radius)
DEFAULT_GRID = (
    ("fast", 1, 1, 0.0, 1.0, 0.0),
    ("balanced", 1, 1, 0.0, 1.0, 0.0),
    ("balanced", 4, 1, 0.0, 1.0, 0.0),
    ("accurate", 1, 1, 0.0, 1.0, 0.0),
    ("fast", 1, 3, 0.45, 0.35, 4.0),
    ("balanced", 1, 3, 0.65, 0.45, 5.0),
    ("balanced", 1, 3, 0.80, 0.60, 6.0),
    ("accurate", 1, 3, 0.65, 0.45, 5.0),
)


def config_id(preset: str, colonies: int, stages: int, w0: float) -> str:
    return f"{preset}_col{colonies}_st{stages}_w{w0:.2f}"


def grid_plans(grid=DEFAULT_GRID) -> List[SearchPlan]:
    plans = []
    for preset, colonies, stages, w0, tau, radius in grid:
        plans.append(
            SearchPlan(
                regime=config_id(preset, colonies, stages, w0),
                selector="grid",
                exhaustiveness=colonies,
                num_modes=9,
                max_evals=None,
                stages=stages,
                initial_bias_weight=w0,
                annealing_tau=tau,
                anchor_radius=radius,
                diversity_floor_angstrom=1.2 if w0 > 0 else 0.0,
                notes=["sweep configuration for selector training"],
                search_method="aco",
                aco_preset=preset,
            )
        )
    return plans


class FixedPlanSelector:
    """Selector that always returns one given plan (used by the sweep)."""

    def __init__(self, plan: SearchPlan) -> None:
        self._plan = plan
        self.name = f"grid:{plan.regime}"

    def plan(self, ligand, pocket) -> SearchPlan:  # noqa: D401 - interface match
        import copy
        return copy.deepcopy(self._plan)
