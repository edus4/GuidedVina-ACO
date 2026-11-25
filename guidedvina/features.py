"""
Ligand and pocket descriptors used to choose a search budget.

Design note: rotatable-bond counts are never guessed from a correlation with
heavy-atom count. A PDBQT carries the count explicitly; for other formats we
ask RDKit. If neither is available the caller must pass the count or opt in to
a clearly-labelled estimate, so that an estimated descriptor can never be
mistaken for a measured one downstream.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from .structures import (
    Atom,
    ParseError,
    count_pdbqt_torsions,
    read_first_model,
    residues_in_box,
)

# Heavy-atoms-per-rotatable-bond ratio used only when the user explicitly
# opts in via allow_estimate=True. Derived from the median ratio across
# drug-like ligands; it is a crude stand-in, flagged as such in every record.
_FALLBACK_RB_PER_HEAVY_ATOM = 0.22


class MissingDescriptorError(RuntimeError):
    """Raised when a descriptor cannot be measured and no override was given."""


def _rdkit_rotatable_bonds(path: Path) -> Optional[int]:
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import Descriptors  # type: ignore
    except ImportError:
        return None
    suffix = path.suffix.lower()
    mol = None
    if suffix == ".pdb":
        mol = Chem.MolFromPDBFile(str(path), removeHs=False, sanitize=True)
    elif suffix in {".sdf", ".mol"}:
        supplier = Chem.SDMolSupplier(str(path), removeHs=False)
        mol = next((m for m in supplier if m is not None), None)
    elif suffix in {".mol2"}:
        mol = Chem.MolFromMol2File(str(path), removeHs=False)
    if mol is None:
        return None
    return int(Descriptors.NumRotatableBonds(mol))


@dataclass
class LigandFeatures:
    path: str
    input_format: str
    heavy_atoms: int
    rotatable_bonds: int
    rotatable_bonds_source: str   # "pdbqt_remark" | "rdkit" | "user" | "estimated"
    centroid: tuple
    max_radius: float             # furthest heavy atom from the centroid

    @property
    def is_estimated(self) -> bool:
        return self.rotatable_bonds_source == "estimated"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class PocketFeatures:
    center: tuple
    size: tuple
    box_volume: float
    heavy_atoms_in_box: int
    polar_atoms_in_box: int
    apolar_atoms_in_box: int
    buriedness: float             # fraction of box volume occupied by receptor atoms
    anchor: Optional[tuple] = None
    anchor_source: str = "none"
    anchor_residues: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def ligand_features(
    ligand_path: Path,
    rotatable_bonds: Optional[int] = None,
    allow_estimate: bool = False,
) -> LigandFeatures:
    ligand_path = Path(ligand_path)
    model = read_first_model(ligand_path)
    heavy = model.heavy()
    if not heavy:
        raise ParseError(f"No heavy atoms found in ligand {ligand_path}")
    coords = np.vstack([a.xyz for a in heavy])
    centroid = coords.mean(axis=0)
    max_radius = float(np.max(np.linalg.norm(coords - centroid, axis=1)))
    suffix = ligand_path.suffix.lower()

    if rotatable_bonds is not None:
        rb, source = int(rotatable_bonds), "user"
    else:
        rb_opt: Optional[int] = None
        source = ""
        if suffix == ".pdbqt":
            rb_opt = count_pdbqt_torsions(ligand_path)
            source = "pdbqt_remark"
        if rb_opt is None:
            rb_opt = _rdkit_rotatable_bonds(ligand_path)
            source = "rdkit" if rb_opt is not None else ""
        if rb_opt is None:
            if not allow_estimate:
                raise MissingDescriptorError(
                    f"Cannot determine rotatable bonds for {ligand_path}. "
                    "Supply --rotatable-bonds N, install RDKit, provide a .pdbqt "
                    "with torsion metadata, or pass --allow-rb-estimate to use a "
                    "heavy-atom heuristic (flagged as estimated in all outputs)."
                )
            rb_opt = max(0, int(round(len(heavy) * _FALLBACK_RB_PER_HEAVY_ATOM)))
            source = "estimated"
        rb = int(rb_opt)

    return LigandFeatures(
        path=str(ligand_path),
        input_format=suffix.lstrip(".") or "unknown",
        heavy_atoms=len(heavy),
        rotatable_bonds=rb,
        rotatable_bonds_source=source,
        centroid=tuple(float(v) for v in centroid),
        max_radius=max_radius,
    )


# AutoDock types and PDB elements counted as polar (H-bond capable).
_POLAR_ELEMENTS = frozenset({"N", "O", "S"})


def pocket_features(
    receptor_path: Path,
    center: Sequence[float],
    size: Sequence[float],
) -> PocketFeatures:
    model = read_first_model(Path(receptor_path))
    heavy = [a for a in model.atoms if not a.is_hydrogen]
    in_box = residues_in_box(heavy, center, size)
    polar = sum(1 for a in in_box if a.element in _POLAR_ELEMENTS)
    box_volume = float(size[0] * size[1] * size[2])
    # Crude occupancy: heavy atoms x 20 A^3 van der Waals volume each.
    occupied = min(box_volume, len(in_box) * 20.0)
    return PocketFeatures(
        center=tuple(float(v) for v in center),
        size=tuple(float(v) for v in size),
        box_volume=box_volume,
        heavy_atoms_in_box=len(in_box),
        polar_atoms_in_box=polar,
        apolar_atoms_in_box=len(in_box) - polar,
        buriedness=round(occupied / box_volume, 4) if box_volume > 0 else 0.0,
    )
