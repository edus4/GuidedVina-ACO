"""
Heavy-atom RMSD between a docked pose and a crystallographic reference.

Three methods, in decreasing rigour. Every result carries the method name, so
a success rate computed with a weaker method is never mistaken for a
symmetry-corrected one:

``graph_symmetry``
    RDKit ``GetBestRMS``: minimises over the automorphisms of the molecular
    graph. This is the method that pose-prediction papers mean by
    "symmetry-corrected RMSD".
``assignment_lower_bound``
    Optimal one-to-one assignment within each element class (Hungarian
    algorithm). It ignores bonding, so it can pair atoms that are not
    topologically equivalent; the value is therefore a *lower bound* on the
    graph-symmetry RMSD. Useful when RDKit is unavailable, and honest about
    what it is.
``ordered``
    Direct index-to-index comparison. Only valid when both files came from the
    same preparation with identical atom ordering; wrong whenever a converter
    reordered atoms, and inflated for symmetric groups.

No method is applied if the two structures disagree in element composition;
that is reported as an error rather than silently returning NaN.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .structures import Model, read_models


@dataclass
class RmsdResult:
    value: Optional[float]
    method: str
    n_atoms: int
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _composition(elements: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for e in elements:
        out[e] = out.get(e, 0) + 1
    return out


def ordered_rmsd(ref: np.ndarray, pred: np.ndarray) -> float:
    diff = ref - pred
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def assignment_rmsd(
    ref: np.ndarray,
    pred: np.ndarray,
    ref_elements: Sequence[str],
    pred_elements: Sequence[str],
) -> float:
    """
    Optimal assignment within element classes. Requires scipy.

    Minimising total squared distance subject to a one-to-one matching is a
    linear assignment problem, solved exactly per element class.
    """
    from scipy.optimize import linear_sum_assignment  # type: ignore

    total_sq = 0.0
    n = 0
    for element in sorted(set(ref_elements)):
        ri = [i for i, e in enumerate(ref_elements) if e == element]
        pi = [i for i, e in enumerate(pred_elements) if e == element]
        if len(ri) != len(pi):
            raise ValueError(
                f"Element {element}: reference has {len(ri)} atoms, pose has {len(pi)}"
            )
        if not ri:
            continue
        a = ref[ri]
        b = pred[pi]
        cost = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
        rows, cols = linear_sum_assignment(cost)
        total_sq += float(cost[rows, cols].sum())
        n += len(ri)
    if n == 0:
        raise ValueError("No atoms to compare")
    return float(np.sqrt(total_sq / n))


def _model_to_pdb_block(model: Model) -> str:
    lines = []
    for i, atom in enumerate(model.heavy(), start=1):
        x, y, z = atom.xyz
        element = atom.element
        lines.append(
            f"HETATM{i:5d} {atom.name[:4]:<4s} {atom.res_name[:3]:>3s} "
            f"{(atom.chain or 'A')[:1]}{atom.res_seq if atom.res_seq > 0 else 1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}"
        )
    lines.append("END")
    return "\n".join(lines)


def graph_symmetry_rmsd(ref_model: Model, pred_model: Model) -> Optional[float]:
    """RDKit GetBestRMS, or None when RDKit cannot build both molecules."""
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import rdMolAlign  # type: ignore
        from rdkit import RDLogger  # type: ignore
    except ImportError:
        return None
    RDLogger.DisableLog("rdApp.*")
    ref = Chem.MolFromPDBBlock(_model_to_pdb_block(ref_model), removeHs=True, sanitize=False)
    pred = Chem.MolFromPDBBlock(_model_to_pdb_block(pred_model), removeHs=True, sanitize=False)
    if ref is None or pred is None:
        return None
    try:
        Chem.SanitizeMol(ref, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        Chem.SanitizeMol(pred, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return float(rdMolAlign.GetBestRMS(pred, ref))
    except Exception:
        return None


def pose_rmsd(
    reference_path: Path,
    pose_path: Path,
    pose_index: int = 0,
    prefer: str = "auto",
) -> RmsdResult:
    """
    RMSD of ``pose_index`` in ``pose_path`` against the first model of
    ``reference_path``. ``prefer`` is one of auto/graph/assignment/ordered.
    """
    try:
        ref_models = read_models(Path(reference_path))
        pose_models = read_models(Path(pose_path))
    except Exception as exc:
        return RmsdResult(None, "none", 0, f"cannot read structures: {exc}")

    if pose_index >= len(pose_models):
        return RmsdResult(
            None, "none", 0,
            f"pose index {pose_index} requested but output has {len(pose_models)} models",
        )
    ref_model, pose_model = ref_models[0], pose_models[pose_index]
    ref_xyz = ref_model.coords(heavy_only=True)
    pose_xyz = pose_model.coords(heavy_only=True)
    ref_el = ref_model.elements(heavy_only=True)
    pose_el = pose_model.elements(heavy_only=True)

    if len(ref_xyz) == 0 or len(pose_xyz) == 0:
        return RmsdResult(None, "none", 0, "no heavy atoms in reference or pose")
    if _composition(ref_el) != _composition(pose_el):
        return RmsdResult(
            None, "none", len(ref_xyz),
            "element composition differs between reference and pose "
            f"({_composition(ref_el)} vs {_composition(pose_el)}); "
            "the files describe different molecules or were prepared differently",
        )

    if prefer in {"auto", "graph"}:
        val = graph_symmetry_rmsd(ref_model, pose_model)
        if val is not None:
            return RmsdResult(round(val, 3), "graph_symmetry", len(ref_xyz))
        if prefer == "graph":
            return RmsdResult(None, "none", len(ref_xyz), "RDKit unavailable or molecule unreadable")

    if prefer in {"auto", "assignment"}:
        try:
            val = assignment_rmsd(ref_xyz, pose_xyz, ref_el, pose_el)
            return RmsdResult(round(val, 3), "assignment_lower_bound", len(ref_xyz))
        except ImportError:
            if prefer == "assignment":
                return RmsdResult(None, "none", len(ref_xyz), "scipy not installed")
        except ValueError as exc:
            return RmsdResult(None, "none", len(ref_xyz), str(exc))

    if len(ref_xyz) != len(pose_xyz):
        return RmsdResult(None, "none", len(ref_xyz), "atom counts differ; ordered RMSD invalid")
    return RmsdResult(round(ordered_rmsd(ref_xyz, pose_xyz), 3), "ordered", len(ref_xyz))


def best_pose_rmsd(reference_path: Path, pose_path: Path, prefer: str = "auto") -> Tuple[RmsdResult, int]:
    """Lowest RMSD over all poses in the file, with its index (for 'any-pose' metrics)."""
    try:
        n = len(read_models(Path(pose_path)))
    except Exception as exc:
        return RmsdResult(None, "none", 0, f"cannot read poses: {exc}"), -1
    best, best_i = RmsdResult(None, "none", 0, "no poses"), -1
    for i in range(n):
        r = pose_rmsd(reference_path, pose_path, pose_index=i, prefer=prefer)
        if r.value is not None and (best.value is None or r.value < best.value):
            best, best_i = r, i
    return best, best_i
