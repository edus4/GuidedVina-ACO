import csv
import json
from pathlib import Path

import pytest

from guidedvina.benchmark import ManifestError, read_manifest, run_benchmark
from guidedvina.engine import probe_engine
from guidedvina.planner import HeuristicSelector
from guidedvina.stats import compare_arms, exact_mcnemar_p


def test_manifest_requires_columns(tmp_path):
    bad = tmp_path / "m.csv"
    bad.write_text("complex_id,receptor\nx,y\n", encoding="utf-8")
    with pytest.raises(ManifestError) as exc:
        read_manifest(bad)
    assert "missing required columns" in str(exc.value)


def test_manifest_blank_size_uses_default(tmp_path):
    m = tmp_path / "m.csv"
    m.write_text(
        "complex_id,receptor,ligand_input,center_x,center_y,center_z,size_x\n"
        "c1,r.pdbqt,l.pdbqt,1,2,3,\n",
        encoding="utf-8",
    )
    rows = read_manifest(m)
    assert rows[0].size == (20.0, 20.0, 20.0)
    assert rows[0].center == (1.0, 2.0, 3.0)


def test_manifest_bad_number_is_reported(tmp_path):
    m = tmp_path / "m.csv"
    m.write_text(
        "complex_id,receptor,ligand_input,center_x,center_y,center_z\n"
        "c1,r.pdbqt,l.pdbqt,abc,2,3\n",
        encoding="utf-8",
    )
    with pytest.raises(ManifestError):
        read_manifest(m)


def test_paired_benchmark_runs_both_arms(stub_path, examples_dir, tmp_path):
    summary = run_benchmark(
        caps=probe_engine(str(stub_path)),
        manifest_path=examples_dir / "sample_manifest.csv",
        out_dir=tmp_path / "bench",
        selector=HeuristicSelector(),
        seeds=(1, 2, 3),
        progress=False,
    )
    assert summary["n_runs"] == 2 * 2 * 3
    assert summary["n_failed_runs"] == 0
    rows = list(csv.DictReader(open(tmp_path / "bench" / "runs.csv", encoding="utf-8")))
    arms = {r["arm"] for r in rows}
    assert arms == {"baseline", "guided"}
    guided = [r for r in rows if r["arm"] == "guided"]
    baseline = [r for r in rows if r["arm"] == "baseline"]
    assert all(r["bias_applied"] == "True" for r in guided)
    assert all(r["bias_applied"] == "False" for r in baseline)
    assert all(r["anchor_source"] == "residues" for r in guided)
    # Every run must carry the RMSD method it used.
    assert all(r["rmsd_method"] for r in rows)
    assert (tmp_path / "bench" / "summary.json").exists()


def test_summary_warns_about_single_seed(stub_path, examples_dir, tmp_path):
    summary = run_benchmark(
        caps=probe_engine(str(stub_path)),
        manifest_path=examples_dir / "sample_manifest.csv",
        out_dir=tmp_path / "bench1",
        selector=HeuristicSelector(),
        seeds=(1,),
        progress=False,
    )
    assert any("seed" in c for c in summary["caveats"])


def test_one_broken_complex_does_not_abort_the_batch(stub_path, examples_dir, tmp_path):
    m = tmp_path / "m.csv"
    m.write_text(
        "complex_id,receptor,ligand_input,ligand_ref,center_x,center_y,center_z,"
        "size_x,size_y,size_z,critical_residues\n"
        f"good,{examples_dir / 'sample_receptor.pdbqt'},{examples_dir / 'sample_ligand.pdbqt'},"
        f"{examples_dir / 'sample_ligand_ref.pdbqt'},0,0,0,20,20,20,A:12\n"
        "broken,/nonexistent/r.pdbqt,/nonexistent/l.pdbqt,,0,0,0,20,20,20,A:12\n",
        encoding="utf-8",
    )
    summary = run_benchmark(
        caps=probe_engine(str(stub_path)),
        manifest_path=m,
        out_dir=tmp_path / "bench2",
        selector=HeuristicSelector(),
        seeds=(1,),
        progress=False,
    )
    rows = list(csv.DictReader(open(tmp_path / "bench2" / "runs.csv", encoding="utf-8")))
    assert len(rows) == 4
    assert summary["n_failed_runs"] == 2
    assert any(r["error"] for r in rows if r["complex_id"] == "broken")
    assert not any(r["error"] for r in rows if r["complex_id"] == "good")


def test_exact_mcnemar_values():
    assert exact_mcnemar_p(0, 0) is None
    assert exact_mcnemar_p(3, 0) == pytest.approx(0.25)
    assert exact_mcnemar_p(5, 0) == pytest.approx(0.0625)
    assert exact_mcnemar_p(2, 2) == pytest.approx(1.0)


def test_comparison_reports_ci_and_pairs():
    per_complex = {
        f"c{i}": {
            "baseline": {"successes": [i % 4 == 0], "runtimes": [10.0]},
            "guided": {"successes": [i % 2 == 0], "runtimes": [6.0]},
        }
        for i in range(24)
    }
    result = compare_arms(per_complex)
    assert result.n_complexes == 24
    assert result.guided_success_rate > result.baseline_success_rate
    assert result.ci95_low <= result.difference <= result.ci95_high
    assert result.median_runtime_ratio == pytest.approx(0.6)
    assert result.baseline_only_wins == 0


def test_comparison_handles_empty_input():
    result = compare_arms({})
    assert result.n_complexes == 0
    assert result.difference is None
    assert "no complexes" in result.note
