import json
import subprocess
import sys
from pathlib import Path

import pytest

from guidedvina.features import LigandFeatures, PocketFeatures
from guidedvina.planner import FEATURE_ORDER, TrainedSelector

ROOT = Path(__file__).resolve().parent.parent


def _model(tmp_path: Path) -> Path:
    """Two configurations: a cheap weak one and an expensive strong one."""
    zeros = [0.0] * len(FEATURE_ORDER)
    rb_index = FEATURE_ORDER.index("rotatable_bonds")
    strong = list(zeros)
    strong[rb_index] = -0.1   # strong config degrades slowly with flexibility
    weak = list(zeros)
    weak[rb_index] = -1.0     # weak config degrades fast
    model = {
        "name": "unit",
        "feature_order": list(FEATURE_ORDER),
        "configurations": [
            {
                "config_id": "cheap", "regime": "cheap",
                "exhaustiveness": 8, "stages": 1, "initial_bias_weight": 0.0,
                "success_coefs": weak, "success_intercept": 6.0,
                "runtime_coefs": zeros, "runtime_intercept": 5.0,
            },
            {
                "config_id": "expensive", "regime": "expensive",
                "exhaustiveness": 32, "stages": 3, "initial_bias_weight": 0.7,
                "success_coefs": strong, "success_intercept": 3.0,
                "runtime_coefs": zeros, "runtime_intercept": 50.0,
            },
        ],
    }
    p = tmp_path / "selector.json"
    p.write_text(json.dumps(model), encoding="utf-8")
    return p


def _features(rb: int):
    lig = LigandFeatures("x", "pdbqt", 20 + rb, rb, "pdbqt_remark", (0, 0, 0), 4.0)
    pocket = PocketFeatures((0, 0, 0), (20, 20, 20), 8000.0, 100, 35, 65, 0.25)
    return lig, pocket


def test_selector_prefers_cheap_config_when_it_suffices(tmp_path):
    sel = TrainedSelector(_model(tmp_path))
    plan = sel.plan(*_features(rb=1))
    assert plan.regime == "cheap"
    assert plan.selector == "trained:unit"


def test_selector_pays_for_the_strong_config_when_needed(tmp_path):
    sel = TrainedSelector(_model(tmp_path))
    plan = sel.plan(*_features(rb=12))
    assert plan.regime == "expensive"
    assert plan.initial_bias_weight == pytest.approx(0.7)


def test_feature_order_mismatch_is_rejected(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"feature_order": ["nonsense"], "configurations": []}),
                   encoding="utf-8")
    with pytest.raises(ValueError):
        TrainedSelector(bad)


def test_training_script_runs_end_to_end(tmp_path):
    """Sweep against the stub, then fit a model from it."""
    sweep_dir = tmp_path / "sweep"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "sweep.py"),
         "--manifest", str(ROOT / "examples" / "sample_manifest.csv"),
         "--engine-bin", str(ROOT / "tests" / "stub_engine.py"),
         "--out-dir", str(sweep_dir), "--seeds", "1", "2", "3"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    assert (sweep_dir / "sweep.csv").exists()

    out = tmp_path / "selector.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train_selector.py"),
         "--sweep", str(sweep_dir / "sweep.csv"), "--out", str(out),
         "--min-rows-per-config", "6"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    model = json.loads(out.read_text(encoding="utf-8"))
    assert model["feature_order"] == list(FEATURE_ORDER)
    assert model["configurations"]
    # Too few complexes for honest validation: that must be stated, not hidden.
    assert "cross_validation" in model
    TrainedSelector(out)  # must load back
