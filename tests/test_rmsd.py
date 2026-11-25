import numpy as np
import pytest

from guidedvina.rmsd import assignment_rmsd, ordered_rmsd, pose_rmsd


def _write(path, coords, elements=("C",), names=None):
    lines = []
    for i, xyz in enumerate(coords, start=1):
        el = elements[(i - 1) % len(elements)]
        name = names[i - 1] if names else f"{el}{i}"
        lines.append(
            f"HETATM{i:5d}  {name:<3s} LIG A   1    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00     0.000 {el:<2s}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_identical_poses_have_zero_rmsd(tmp_path):
    coords = [(0, 0, 0), (1.5, 0, 0), (3.0, 0, 0)]
    ref = _write(tmp_path / "ref.pdbqt", coords)
    pose = _write(tmp_path / "pose.pdbqt", coords)
    r = pose_rmsd(ref, pose)
    assert r.value == pytest.approx(0.0)
    assert r.method in {"graph_symmetry", "assignment_lower_bound", "ordered"}


def test_symmetry_swap_is_corrected(tmp_path):
    # Two identical carbons swapped: an ordered comparison sees a large RMSD,
    # an assignment-based one sees none.
    ref = _write(tmp_path / "ref.pdbqt", [(0, 0, 0), (4, 0, 0)])
    pose = _write(tmp_path / "pose.pdbqt", [(4, 0, 0), (0, 0, 0)])
    ordered = ordered_rmsd(np.array([[0, 0, 0], [4, 0, 0]], float),
                           np.array([[4, 0, 0], [0, 0, 0]], float))
    assert ordered == pytest.approx(4.0)
    r = pose_rmsd(ref, pose, prefer="assignment")
    assert r.value == pytest.approx(0.0)
    assert r.method == "assignment_lower_bound"


def test_element_mismatch_is_reported_not_nan(tmp_path):
    ref = _write(tmp_path / "ref.pdbqt", [(0, 0, 0), (1, 0, 0)], elements=("C", "C"))
    pose = _write(tmp_path / "pose.pdbqt", [(0, 0, 0), (1, 0, 0)], elements=("C", "N"))
    r = pose_rmsd(ref, pose)
    assert r.value is None
    assert "composition differs" in (r.error or "")


def test_assignment_respects_element_classes():
    ref = np.array([[0, 0, 0], [4, 0, 0]], float)
    pose = np.array([[4, 0, 0], [0, 0, 0]], float)
    # Same elements: swap is free.
    assert assignment_rmsd(ref, pose, ["C", "C"], ["C", "C"]) == pytest.approx(0.0)
    # Different elements: the swap is not allowed, so the distance stands.
    assert assignment_rmsd(ref, pose, ["C", "N"], ["C", "N"]) == pytest.approx(4.0)


def test_missing_pose_index_is_an_error(tmp_path):
    ref = _write(tmp_path / "ref.pdbqt", [(0, 0, 0)])
    pose = _write(tmp_path / "pose.pdbqt", [(0, 0, 0)])
    r = pose_rmsd(ref, pose, pose_index=5)
    assert r.value is None and "pose index" in (r.error or "")
