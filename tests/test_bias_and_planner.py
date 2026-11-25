import numpy as np
import pytest

from guidedvina.bias import (
    BiasPoint,
    anchor_bias_points,
    apply_diversity_guardrail,
    build_schedule,
    pose_diversity,
    write_bias_file,
)
from guidedvina.features import LigandFeatures, PocketFeatures
from guidedvina.planner import HeuristicSelector, baseline_plan


def _lig(rb=6, heavy=30):
    return LigandFeatures("x", "pdbqt", heavy, rb, "pdbqt_remark", (0, 0, 0), 4.5)


def _pocket(vol=8000.0):
    side = vol ** (1 / 3)
    return PocketFeatures((0, 0, 0), (side, side, side), vol, 120, 40, 80, 0.3)


@pytest.mark.parametrize("rb,expected", [(1, "rigid_prior"), (6, "moderate_prior"), (12, "flexible_prior")])
def test_regime_bands(rb, expected):
    plan = HeuristicSelector().plan(_lig(rb), _pocket())
    assert plan.regime == expected


def test_estimated_rb_is_flagged_in_plan_notes():
    lig = LigandFeatures("x", "pdb", 30, 7, "estimated", (0, 0, 0), 4.5)
    plan = HeuristicSelector().plan(lig, _pocket())
    assert any("ESTIMATED" in n for n in plan.notes)


def test_final_stage_is_unbiased():
    plan = HeuristicSelector().plan(_lig(), _pocket())
    schedule = build_schedule(plan)
    assert len(schedule) == plan.stages
    assert schedule[-1].bias_weight == 0.0
    assert schedule[-1].local_only is True


def test_bias_decays_monotonically():
    plan = HeuristicSelector().plan(_lig(12), _pocket())
    weights = [s.bias_weight for s in build_schedule(plan)]
    assert weights == sorted(weights, reverse=True)
    assert weights[0] == pytest.approx(plan.initial_bias_weight)


def test_baseline_plan_has_no_bias():
    schedule = build_schedule(baseline_plan())
    assert len(schedule) == 1
    assert schedule[0].bias_weight == 0.0
    assert anchor_bias_points((1, 2, 3), 0.0, 5.0) == []


def test_guardrail_cuts_bias_on_collapse():
    full, applied = apply_diversity_guardrail(0.5, 2.0, 1.2)
    assert (full, applied) == (0.5, False)
    cut, applied = apply_diversity_guardrail(0.5, 0.6, 1.2)
    assert applied and cut == pytest.approx(0.125)
    zero, applied = apply_diversity_guardrail(0.5, 0.0, 1.2)
    assert applied and zero == 0.0


def test_pose_diversity_is_measured_not_assumed():
    a = np.zeros((4, 3))
    b = np.ones((4, 3))
    assert pose_diversity([a]) == 0.0
    assert pose_diversity([a, a]) == 0.0
    assert pose_diversity([a, b]) == pytest.approx(np.sqrt(3.0))


def test_bias_file_format(tmp_path):
    points = anchor_bias_points((1.0, -2.0, 3.0), weight=0.5, radius=4.0)
    path = write_bias_file(points, tmp_path / "b.bpf")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "x y z Vset r type"
    fields = lines[1].split()
    assert fields[-1] == "map"
    assert float(fields[3]) == pytest.approx(-1.0)  # 0.5 * 2.0 kcal/mol, attractive


def test_unknown_bias_type_rejected():
    with pytest.raises(ValueError):
        BiasPoint(0, 0, 0, -1.0, 4.0, "nonsense").to_line()
