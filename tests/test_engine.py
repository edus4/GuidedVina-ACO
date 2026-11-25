import json
import subprocess
import sys
from pathlib import Path

import pytest

from guidedvina.anchors import parse_residue_specs
from guidedvina.engine import (
    CapabilityError,
    EngineError,
    assumed_capabilities,
    probe_engine,
    run_staged,
)
from guidedvina.planner import HeuristicSelector, baseline_plan
from guidedvina.features import ligand_features, pocket_features
from guidedvina.runner import dock_once

NOBIAS = Path(__file__).with_name("stub_engine_nobias.py")


def test_probe_reads_flags_from_help(stub_path):
    caps = probe_engine(str(stub_path))
    assert caps.supports("bias")
    assert caps.flag("exhaustiveness") == "exhaustiveness"
    assert caps.supports("local_only")


def test_probe_fails_loudly_on_missing_binary():
    with pytest.raises(EngineError):
        probe_engine("definitely-not-a-real-docking-binary-xyz")


def test_bias_flag_reaches_the_command_line(stub_path, examples_dir, tmp_path):
    caps = probe_engine(str(stub_path))
    lig = ligand_features(examples_dir / "sample_ligand.pdbqt")
    pock = pocket_features(examples_dir / "sample_receptor.pdbqt", (0, 0, 0), (20, 20, 20))
    plan = HeuristicSelector().plan(lig, pock)
    result = run_staged(
        caps=caps,
        receptor=examples_dir / "sample_receptor.pdbqt",
        ligand=examples_dir / "sample_ligand.pdbqt",
        center=(0, 0, 0), size=(20, 20, 20),
        out_path=tmp_path / "out.pdbqt",
        plan=plan,
        anchor=(-0.9, 1.6, 1.1),
        arm="guided",
        work_dir=tmp_path / "work",
    )
    assert result.error is None
    biased = [s for s in result.stages if s.bias_file]
    assert biased, "no stage received a bias file"
    for stage in biased:
        assert "--bias" in stage.command
        assert Path(stage.bias_file).exists()
    # The anchor coordinates must be in the file the engine was handed.
    content = Path(biased[0].bias_file).read_text(encoding="utf-8")
    assert "-0.900" in content


def test_biased_stage_refused_when_binary_cannot_bias(examples_dir, tmp_path):
    caps = probe_engine(str(NOBIAS))
    assert not caps.supports("bias")
    lig = ligand_features(examples_dir / "sample_ligand.pdbqt")
    pock = pocket_features(examples_dir / "sample_receptor.pdbqt", (0, 0, 0), (20, 20, 20))
    plan = HeuristicSelector().plan(lig, pock)
    with pytest.raises(CapabilityError):
        run_staged(
            caps=caps,
            receptor=examples_dir / "sample_receptor.pdbqt",
            ligand=examples_dir / "sample_ligand.pdbqt",
            center=(0, 0, 0), size=(20, 20, 20),
            out_path=tmp_path / "out.pdbqt",
            plan=plan,
            anchor=(0.0, 0.0, 0.0),
            arm="guided",
            work_dir=tmp_path / "work",
        )


def test_baseline_arm_sends_no_bias(stub_path, examples_dir, tmp_path):
    caps = probe_engine(str(stub_path))
    result = run_staged(
        caps=caps,
        receptor=examples_dir / "sample_receptor.pdbqt",
        ligand=examples_dir / "sample_ligand.pdbqt",
        center=(0, 0, 0), size=(20, 20, 20),
        out_path=tmp_path / "base.pdbqt",
        plan=baseline_plan(),
        anchor=None,
        arm="baseline",
        work_dir=tmp_path / "work",
    )
    assert result.error is None
    assert len(result.stages) == 1
    assert result.stages[0].bias_file is None
    assert "--bias" not in result.stages[0].command


def test_diversity_is_measured_from_engine_output(stub_path, examples_dir, tmp_path):
    caps = probe_engine(str(stub_path))
    lig = ligand_features(examples_dir / "sample_ligand.pdbqt")
    pock = pocket_features(examples_dir / "sample_receptor.pdbqt", (0, 0, 0), (20, 20, 20))
    plan = HeuristicSelector().plan(lig, pock)
    result = run_staged(
        caps=caps,
        receptor=examples_dir / "sample_receptor.pdbqt",
        ligand=examples_dir / "sample_ligand.pdbqt",
        center=(0, 0, 0), size=(20, 20, 20),
        out_path=tmp_path / "out.pdbqt",
        plan=plan, anchor=(0.0, 0.0, 0.0), arm="guided",
        work_dir=tmp_path / "work", seed=7,
    )
    measured = [s.diversity_after for s in result.stages]
    assert all(m is not None and m >= 0.0 for m in measured)
    assert any(m > 0.0 for m in measured)


def test_failing_stage_is_recorded_not_raised(examples_dir, tmp_path):
    bad = tmp_path / "always_fails.py"
    bad.write_text(
        "#!/usr/bin/env python3\nimport sys\n"
        "if '--help' in sys.argv:\n"
        "    print('  --receptor arg\\n  --ligand arg\\n  --out arg\\n  --seed arg\\n"
        "  --exhaustiveness arg\\n  --bias arg\\n  --local_only')\n"
        "    sys.exit(0)\n"
        "sys.stderr.write('boom\\n')\nsys.exit(4)\n",
        encoding="utf-8",
    )
    bad.chmod(0o755)
    caps = probe_engine(str(bad))
    result = run_staged(
        caps=caps,
        receptor=examples_dir / "sample_receptor.pdbqt",
        ligand=examples_dir / "sample_ligand.pdbqt",
        center=(0, 0, 0), size=(20, 20, 20),
        out_path=tmp_path / "out.pdbqt",
        plan=baseline_plan(), anchor=None, arm="baseline",
        work_dir=tmp_path / "work",
    )
    assert result.error is not None
    assert "exited with code 4" in result.error


def test_dry_run_needs_no_binary(examples_dir, tmp_path):
    caps = assumed_capabilities("vina-aco")
    outcome = dock_once(
        caps=caps,
        complex_id="toy",
        receptor=examples_dir / "sample_receptor.pdbqt",
        ligand=examples_dir / "sample_ligand.pdbqt",
        center=(0, 0, 0), size=(20, 20, 20),
        out_path=tmp_path / "out.pdbqt",
        arm="guided",
        selector=HeuristicSelector(),
        residue_specs=parse_residue_specs(["A:12", "A:60"]),
        dry_run=True,
        work_dir=tmp_path / "work",
    )
    assert outcome.error is None
    assert outcome.anchor is not None
    assert outcome.run is not None and outcome.run.dry_run


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
    assert all(s.bias_file is None for s in outcome.run.stages)


def test_cli_dry_run_is_valid_json(examples_dir):
    proc = subprocess.run(
        [sys.executable, "-m", "guidedvina.cli",
         "--receptor", str(examples_dir / "sample_receptor.pdbqt"),
         "--ligand", str(examples_dir / "sample_ligand.pdbqt"),
         "--center", "0", "0", "0", "--critical-residues", "A:12", "A:60",
         "--dry-run"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["plan"]["selector"] == "heuristic"
    assert report["anchor"]["chains_matched"] == ["A"]
