import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from guidedvina.anchors import parse_residue_specs
from guidedvina.engine import (
    CapabilityError,
    EngineError,
    assumed_capabilities,
    probe_engine,
    run_staged,
    write_base_maps,
)
from guidedvina.features import ligand_features, pocket_features
from guidedvina.maps import find_maps, read_map
from guidedvina.planner import HeuristicSelector, baseline_plan
from guidedvina.runner import dock_once
from guidedvina.structures import read_models

ROOT = Path(__file__).resolve().parent.parent
NOMAPS = Path(__file__).with_name("stub_engine_nomaps.py")
ANCHOR = (-0.9, 1.6, 1.1)


def _plan(examples_dir):
    lig = ligand_features(examples_dir / "sample_ligand.pdbqt")
    pock = pocket_features(examples_dir / "sample_receptor.pdbqt", (0, 0, 0), (20, 20, 20))
    return HeuristicSelector().plan(lig, pock)


def _run(caps, examples_dir, tmp_path, plan, anchor, arm="guided", seed=3):
    return run_staged(
        caps=caps,
        receptor=examples_dir / "sample_receptor.pdbqt",
        ligand=examples_dir / "sample_ligand.pdbqt",
        center=(0, 0, 0), size=(20, 20, 20),
        out_path=tmp_path / "out.pdbqt",
        plan=plan, anchor=anchor, arm=arm,
        work_dir=tmp_path / "work", seed=seed,
    )


def test_probe_resolves_real_vina_aco_options(stub_path):
    caps = probe_engine(str(stub_path))
    assert caps.is_aco
    assert caps.flag("search") == "search"
    assert caps.flag("aco_preset") == "aco_preset"
    assert caps.flag("exhaustiveness") == "exhaustiveness"
    # Vina-ACO has no bias option; bias must go through the maps.
    assert not caps.supports("bias")
    assert caps.bias_backend == "maps"


def test_probe_fails_loudly_on_missing_binary():
    with pytest.raises(EngineError):
        probe_engine("definitely-not-a-real-docking-binary-xyz")


def test_write_base_maps_uses_score_only_and_even_voxels(stub_path, examples_dir, tmp_path):
    caps = probe_engine(str(stub_path))
    cmd, prefix = write_base_maps(
        caps, examples_dir / "sample_receptor.pdbqt", examples_dir / "sample_ligand.pdbqt",
        (0, 0, 0), (20, 20, 20), tmp_path / "basemaps",
    )
    assert "--write_maps" in cmd
    assert "--force_even_voxels" in cmd
    assert "--score_only" in cmd
    written = find_maps(prefix)
    assert written, "no maps written"
    m = read_map(written[0])
    assert m.nelements[0] % 2 == 0, "map format requires an even voxel count"


def test_bias_reaches_the_engine_through_the_maps(stub_path, examples_dir, tmp_path):
    caps = probe_engine(str(stub_path))
    result = _run(caps, examples_dir, tmp_path, _plan(examples_dir), ANCHOR)
    assert result.error is None
    assert result.bias_backend == "maps"

    biased = [s for s in result.stages if s.biased]
    assert biased, "no stage was biased"
    for stage in biased:
        assert "--maps" in stage.command
        assert "--receptor" not in stage.command   # mutually exclusive in Vina
        assert stage.bias_detail["voxels_changed"] > 0

    # The modified maps really do hold a well at the anchor.
    prefix = Path(biased[0].bias_artifact)
    biased_map = read_map(find_maps(prefix)[0])
    base_map = read_map(find_maps(Path(result.map_prefix))[0])
    assert biased_map.values.min() < base_map.values.min()


def test_bias_pulls_poses_towards_the_anchor(stub_path, examples_dir, tmp_path):
    caps = probe_engine(str(stub_path))
    guided = _run(caps, examples_dir, tmp_path / "g", _plan(examples_dir), ANCHOR)
    baseline = _run(caps, examples_dir, tmp_path / "b", baseline_plan(), None, arm="baseline")
    assert guided.error is None and baseline.error is None

    def distance(path):
        c = read_models(Path(path))[0].coords().mean(axis=0)
        return float(np.linalg.norm(c - np.array(ANCHOR)))

    assert distance(guided.stages[0].out_path) < distance(baseline.stages[0].out_path)


def test_final_stage_is_unbiased_and_uses_the_receptor(stub_path, examples_dir, tmp_path):
    caps = probe_engine(str(stub_path))
    result = _run(caps, examples_dir, tmp_path, _plan(examples_dir), ANCHOR)
    last = result.stages[-1]
    assert last.stage.bias_weight == 0.0
    assert not last.biased
    assert "--maps" not in last.command
    assert "--receptor" in last.command
    assert "--local_only" in last.command


def test_aco_search_is_requested_explicitly(stub_path, examples_dir, tmp_path):
    """--search defaults to mc, so ACO must be asked for on every stage."""
    caps = probe_engine(str(stub_path))
    result = _run(caps, examples_dir, tmp_path, _plan(examples_dir), ANCHOR)
    for stage in result.stages:
        assert "--search" in stage.command
        assert stage.command[stage.command.index("--search") + 1] == "aco"
        assert "--aco_preset" in stage.command


def test_colonies_never_increase_after_the_first_stage(stub_path, examples_dir, tmp_path):
    caps = probe_engine(str(stub_path))
    result = _run(caps, examples_dir, tmp_path, _plan(examples_dir), ANCHOR)
    breadth = [s.stage.exhaustiveness for s in result.stages]
    assert breadth[0] == max(breadth)
    assert all(b >= 1 for b in breadth)


def test_baseline_arm_sends_no_bias(stub_path, examples_dir, tmp_path):
    caps = probe_engine(str(stub_path))
    result = _run(caps, examples_dir, tmp_path, baseline_plan(), None, arm="baseline")
    assert result.error is None
    assert len(result.stages) == 1
    assert not result.stages[0].biased
    assert "--maps" not in result.stages[0].command


def test_engine_without_maps_or_bias_is_refused(examples_dir, tmp_path):
    caps = probe_engine(str(NOMAPS))
    assert caps.bias_backend == "none"
    with pytest.raises(CapabilityError, match="bias cannot be applied"):
        _run(caps, examples_dir, tmp_path, _plan(examples_dir), ANCHOR)


def test_missing_seed_option_is_refused(examples_dir, tmp_path):
    """Without --seed, repeated seeds are indistinguishable."""
    bad = tmp_path / "noseed.py"
    bad.write_text(
        "#!/usr/bin/env python3\nimport sys\n"
        "print('  --receptor arg\\n  --ligand arg\\n  --out arg\\n  --exhaustiveness arg\\n"
        "  --maps arg\\n  --write_maps arg\\n  --search arg\\n  --aco_preset arg')\n",
        encoding="utf-8",
    )
    bad.chmod(0o755)
    caps = probe_engine(str(bad))
    assert not caps.supports("seed")
    with pytest.raises(CapabilityError, match="seed"):
        _run(caps, examples_dir, tmp_path, baseline_plan(), None, arm="baseline")


def test_diversity_is_measured_from_engine_output(stub_path, examples_dir, tmp_path):
    caps = probe_engine(str(stub_path))
    result = _run(caps, examples_dir, tmp_path, _plan(examples_dir), ANCHOR)
    measured = [s.diversity_after for s in result.stages]
    assert all(m is not None and m >= 0.0 for m in measured)
    assert any(m > 0.0 for m in measured)


def test_failing_stage_is_recorded_not_raised(examples_dir, tmp_path):
    bad = tmp_path / "always_fails.py"
    bad.write_text(
        "#!/usr/bin/env python3\nimport sys\n"
        "if '--help' in sys.argv or len(sys.argv) == 1:\n"
        "    print('  --receptor arg\\n  --ligand arg\\n  --out arg\\n  --seed arg\\n"
        "  --exhaustiveness arg\\n  --local_only\\n  --search arg\\n  --aco_preset arg')\n"
        "    sys.exit(0)\n"
        "sys.stderr.write('boom\\n')\nsys.exit(4)\n",
        encoding="utf-8",
    )
    bad.chmod(0o755)
    caps = probe_engine(str(bad))
    result = _run(caps, examples_dir, tmp_path, baseline_plan(), None, arm="baseline")
    assert result.error is not None and "exited with code 4" in result.error


def test_dry_run_needs_no_binary(examples_dir, tmp_path):
    outcome = dock_once(
        caps=assumed_capabilities("vina_aco"),
        complex_id="toy",
        receptor=examples_dir / "sample_receptor.pdbqt",
        ligand=examples_dir / "sample_ligand.pdbqt",
        center=(0, 0, 0), size=(20, 20, 20),
        out_path=tmp_path / "out.pdbqt",
        arm="guided", selector=HeuristicSelector(),
        residue_specs=parse_residue_specs(["A:12", "A:60"]),
        dry_run=True, work_dir=tmp_path / "work",
    )
    assert outcome.error is None
    assert outcome.anchor is not None
    assert outcome.run.bias_backend == "maps"
    # The map-writing step appears in the dry-run plan.
    assert any("--write_maps" in s.command for s in outcome.run.stages)


def test_guided_arm_without_anchor_warns_and_drops_bias(stub_path, examples_dir, tmp_path):
    caps = probe_engine(str(stub_path))
    outcome = dock_once(
        caps=caps, complex_id="toy",
        receptor=examples_dir / "sample_receptor.pdbqt",
        ligand=examples_dir / "sample_ligand.pdbqt",
        center=(0, 0, 0), size=(20, 20, 20),
        out_path=tmp_path / "out.pdbqt",
        arm="guided", selector=HeuristicSelector(),
        residue_specs=[], reference_ligand=None,
        work_dir=tmp_path / "work",
    )
    assert outcome.error is None
    assert any("without bias" in w for w in outcome.warnings)
    assert all(not s.biased for s in outcome.run.stages)


def test_cli_dry_run_is_valid_json(examples_dir):
    proc = subprocess.run(
        [sys.executable, "-m", "guidedvina.cli",
         "--receptor", str(examples_dir / "sample_receptor.pdbqt"),
         "--ligand", str(examples_dir / "sample_ligand.pdbqt"),
         "--center", "0", "0", "0", "--critical-residues", "A:12", "A:60",
         "--dry-run"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["plan"]["search_method"] == "aco"
    assert report["run"]["bias_backend"] == "maps"
    assert report["anchor"]["chains_matched"] == ["A"]
