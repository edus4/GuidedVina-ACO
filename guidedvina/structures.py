"""
PDB / PDBQT parsing.

Fixed-column parsing per the PDB format spec; PDBQT differs only in that
columns 78-79 hold an AutoDock atom type rather than an element symbol.
Multi-model files are supported: each MODEL block is returned separately.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import numpy as np

# AutoDock atom types that denote hydrogen.
_AD_HYDROGEN_TYPES = frozenset({"H", "HD", "HS"})
# Elements treated as hydrogen in plain PDB files.
_PDB_HYDROGEN_ELEMENTS = frozenset({"H", "D"})


@dataclass(frozen=True)
class Atom:
    serial: int
    name: str
    alt_loc: str
    res_name: str
    chain: str
    res_seq: int
    insertion: str
    xyz: np.ndarray
    raw_type: str          # element (PDB) or AutoDock type (PDBQT), cols 77-78
    partial_charge: Optional[float]
    is_hydrogen: bool
    is_hetatm: bool

    @property
    def element(self) -> str:
        """Best-effort element symbol, derived from the AutoDock type if needed."""
        t = self.raw_type.strip()
        if t:
            # AutoDock types: A->C (aromatic carbon), OA->O, NA->N, SA->S, HD->H
            table = {"A": "C", "OA": "O", "NA": "N", "SA": "S", "HD": "H", "HS": "H",
                     "NS": "N", "G0": "C", "CG0": "C"}
            if t.upper() in table:
                return table[t.upper()]
            if len(t) <= 2 and t[0].isalpha():
                return t[0].upper() + t[1:].lower() if len(t) == 2 and t[1].isalpha() else t[0].upper()
        # Fall back to the atom name, which is right-justified by element in PDB.
        name = self.name.strip()
        for ch in name:
            if ch.isalpha():
                return ch.upper()
        return "X"

    @property
    def is_backbone(self) -> bool:
        return self.name.strip().upper() in {"N", "CA", "C", "O", "OXT"}

    @property
    def residue_key(self) -> tuple:
        return (self.chain, self.res_seq, self.insertion)


class ParseError(ValueError):
    """Raised when a structure file cannot be parsed."""


def _is_hydrogen(raw_type: str, atom_name: str, is_pdbqt: bool) -> bool:
    t = raw_type.strip().upper()
    name = atom_name.strip().upper()
    if is_pdbqt:
        if t in _AD_HYDROGEN_TYPES:
            return True
        # No type column: fall back to the name, but only when it cannot be a
        # heavy atom whose name happens to start with H (e.g. HG = mercury).
        if not t:
            return name.startswith("H") and name not in {"HG", "HE", "HF", "HO", "HS"}
        return False
    if t in _PDB_HYDROGEN_ELEMENTS:
        return True
    if not t:
        # PDB atom names: "1HB", " HB2" etc. Element is in cols 13-14 for
        # two-character elements, so a leading digit followed by H is hydrogen.
        if name.startswith("H") and name not in {"HG", "HE", "HF", "HO", "HS"}:
            return True
        if len(name) > 1 and name[0].isdigit() and name[1] == "H":
            return True
    return False


def _parse_atom_line(line: str, is_pdbqt: bool) -> Optional[Atom]:
    if len(line) < 54:
        return None
    try:
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
    except ValueError:
        return None
    try:
        serial = int(line[6:11])
    except ValueError:
        serial = -1
    name = line[12:16]
    alt_loc = line[16:17].strip()
    res_name = line[17:20].strip()
    chain = line[21:22].strip()
    res_seq_str = line[22:26].strip()
    try:
        res_seq = int(res_seq_str)
    except ValueError:
        res_seq = -1
    insertion = line[26:27].strip()
    raw_type = line[76:79].strip() if len(line) >= 78 else ""
    charge: Optional[float] = None
    if is_pdbqt and len(line) >= 76:
        try:
            charge = float(line[70:76])
        except ValueError:
            charge = None
    return Atom(
        serial=serial,
        name=name.strip(),
        alt_loc=alt_loc,
        res_name=res_name,
        chain=chain,
        res_seq=res_seq,
        insertion=insertion,
        xyz=np.array([x, y, z], dtype=float),
        raw_type=raw_type,
        partial_charge=charge,
        is_hydrogen=_is_hydrogen(raw_type, name, is_pdbqt),
        is_hetatm=line.startswith("HETATM"),
    )


@dataclass
class Model:
    """One MODEL block (or a whole single-model file)."""
    index: int
    atoms: List[Atom]
    remarks: List[str]

    def heavy(self) -> List[Atom]:
        return [a for a in self.atoms if not a.is_hydrogen]

    def coords(self, heavy_only: bool = True) -> np.ndarray:
        atoms = self.heavy() if heavy_only else self.atoms
        if not atoms:
            return np.zeros((0, 3), dtype=float)
        return np.vstack([a.xyz for a in atoms])

    def elements(self, heavy_only: bool = True) -> List[str]:
        atoms = self.heavy() if heavy_only else self.atoms
        return [a.element for a in atoms]

    def score(self) -> Optional[float]:
        """Vina writes 'REMARK VINA RESULT:  -7.6  0.000  0.000'."""
        for r in self.remarks:
            up = r.upper()
            if "VINA RESULT" in up or "USER    Estimated Free Energy" in r:
                for tok in r.replace(":", " ").split():
                    try:
                        return float(tok)
                    except ValueError:
                        continue
        return None


def iter_models(path: Path) -> Iterator[Model]:
    """Yield each MODEL of a .pdb/.pdbqt file. Files without MODEL yield one."""
    path = Path(path)
    is_pdbqt = path.suffix.lower() == ".pdbqt"
    current: List[Atom] = []
    remarks: List[str] = []
    index = 0
    seen_any = False
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("MODEL"):
                current, remarks = [], []
                continue
            if line.startswith("ENDMDL"):
                index += 1
                seen_any = True
                yield Model(index=index, atoms=current, remarks=remarks)
                current, remarks = [], []
                continue
            if line.startswith(("REMARK", "USER")):
                remarks.append(line.rstrip("\n"))
                continue
            if line.startswith(("ATOM", "HETATM")):
                atom = _parse_atom_line(line, is_pdbqt)
                if atom is not None:
                    current.append(atom)
    if current or not seen_any:
        index += 1
        yield Model(index=index, atoms=current, remarks=remarks)


def read_models(path: Path) -> List[Model]:
    models = [m for m in iter_models(path) if m.atoms]
    if not models:
        raise ParseError(f"No ATOM/HETATM records parsed from {path}")
    return models


def read_first_model(path: Path) -> Model:
    return read_models(path)[0]


def count_pdbqt_torsions(path: Path) -> Optional[int]:
    """
    Read the active-torsion count from a PDBQT.

    Prefers the 'REMARK  N active torsions' header written by the AutoDock
    preparation tools; falls back to counting BRANCH records, which equals the
    number of rotatable bonds the file actually encodes.
    """
    branch = 0
    declared: Optional[int] = None
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break
            low = line.lower()
            if line.startswith("REMARK") and "active torsion" in low:
                for tok in line.split():
                    if tok.isdigit():
                        declared = int(tok)
                        break
            elif line.startswith("BRANCH"):
                branch += 1
    if declared is not None:
        return declared
    return branch if branch else None


def residues_in_box(
    atoms: Sequence[Atom],
    center: Sequence[float],
    size: Sequence[float],
) -> List[Atom]:
    """Atoms whose coordinates fall inside the axis-aligned docking box."""
    if not atoms:
        return []
    c = np.asarray(center, dtype=float)
    half = np.asarray(size, dtype=float) / 2.0
    return [a for a in atoms if bool(np.all(np.abs(a.xyz - c) <= half))]
