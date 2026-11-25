"""
Locating the pocket sub-region that bias is applied to.

Two anchor sources are supported and they are not interchangeable:

* ``residues`` -- the centroid of the side-chain heavy atoms of residues the
  user names. Chain IDs are honoured, so "12" in a homodimer does not silently
  average two copies into a meaningless midpoint.
* ``ref_ligand`` -- the centroid of the crystallographic ligand. This leaks the
  answer in a redocking benchmark and is refused unless explicitly allowed.

There is deliberately no "guess the pocket from polar atoms" default: in a
20 A box the centroid of all N/O/S atoms is within a fraction of an Angstrom
of the box centre for most receptors, so such a fallback contributes nothing
while appearing to be knowledge-based.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .structures import Atom, read_first_model, residues_in_box

_RES_SPEC = re.compile(r"^(?:(?P<chain>[A-Za-z0-9]):)?(?P<resseq>-?\d+)(?P<icode>[A-Za-z])?$")


class AnchorError(ValueError):
    pass


class LeakageError(RuntimeError):
    """Raised when a benchmark would define bias from the reference answer."""


@dataclass(frozen=True)
class ResidueSpec:
    chain: Optional[str]
    res_seq: int
    insertion: str = ""

    def matches(self, atom: Atom) -> bool:
        if self.chain is not None and atom.chain.upper() != self.chain.upper():
            return False
        if self.insertion and atom.insertion.upper() != self.insertion.upper():
            return False
        return atom.res_seq == self.res_seq

    def __str__(self) -> str:
        base = f"{self.chain}:{self.res_seq}" if self.chain else str(self.res_seq)
        return base + self.insertion


def parse_residue_specs(tokens: Iterable[str]) -> List[ResidueSpec]:
    """
    Parse residue selectors.

    Accepted forms: ``12``, ``A:12``, ``A:12B`` (insertion code), and ranges
    ``A:12-18`` or ``12-18``. A bare number matches that residue number in
    every chain, which is reported so the user can see the ambiguity.
    """
    specs: List[ResidueSpec] = []
    for raw in tokens:
        for token in str(raw).replace(",", " ").replace(";", " ").split():
            token = token.strip()
            if not token:
                continue
            chain: Optional[str] = None
            body = token
            if ":" in token:
                chain, body = token.split(":", 1)
                chain = chain.strip() or None
            if "-" in body.lstrip("-"):
                lo_s, _, hi_s = body.partition("-")
                try:
                    lo, hi = int(lo_s), int(hi_s)
                except ValueError as exc:
                    raise AnchorError(f"Bad residue range: {token!r}") from exc
                if hi < lo:
                    lo, hi = hi, lo
                specs.extend(ResidueSpec(chain, n) for n in range(lo, hi + 1))
                continue
            m = _RES_SPEC.match(body if chain is None else f"{chain}:{body}")
            if not m:
                raise AnchorError(
                    f"Bad residue selector {token!r}; expected forms like 12, A:12, A:12-18"
                )
            specs.append(
                ResidueSpec(
                    chain=m.group("chain"),
                    res_seq=int(m.group("resseq")),
                    insertion=(m.group("icode") or "").strip(),
                )
            )
    return specs


@dataclass
class Anchor:
    xyz: Tuple[float, float, float]
    source: str
    residues_matched: List[str]
    chains_matched: List[str]
    atom_count: int
    radius_of_gyration: float
    warnings: List[str]


def anchor_from_residues(
    receptor_path: Path,
    specs: Sequence[ResidueSpec],
    center: Sequence[float],
    size: Sequence[float],
    sidechain_only: bool = True,
    require_in_box: bool = True,
) -> Anchor:
    if not specs:
        raise AnchorError("No residue selectors supplied")
    model = read_first_model(Path(receptor_path))
    heavy = [a for a in model.atoms if not a.is_hydrogen]
    warnings: List[str] = []

    matched: List[Atom] = []
    matched_keys: Set[tuple] = set()
    for atom in heavy:
        if not any(s.matches(atom) for s in specs):
            continue
        if sidechain_only and atom.is_backbone:
            continue
        matched.append(atom)
        matched_keys.add(atom.residue_key)

    if not matched:
        raise AnchorError(
            f"None of the requested residues {[str(s) for s in specs]} were found "
            f"in {receptor_path}. Check the numbering and chain IDs of the receptor file."
        )

    # Warn when a bare residue number hit several chains: averaging across
    # chains produces an anchor that sits in neither pocket.
    chains = sorted({a.chain for a in matched})
    if len(chains) > 1:
        warnings.append(
            f"Residue selectors matched {len(chains)} chains ({', '.join(c or '_' for c in chains)}). "
            "Qualify selectors as CHAIN:RESNUM to target one pocket."
        )

    if require_in_box:
        inside = residues_in_box(matched, center, size)
        if not inside:
            raise AnchorError(
                f"Residues {[str(s) for s in specs]} were found but none of their "
                "atoms lie inside the docking box. The box centre/size and the "
                "critical residues disagree."
            )
        if len(inside) < len(matched):
            warnings.append(
                f"{len(matched) - len(inside)} of {len(matched)} anchor atoms fall "
                "outside the docking box and were ignored."
            )
        matched = inside

    coords = np.vstack([a.xyz for a in matched])
    centroid = coords.mean(axis=0)
    rog = float(np.sqrt(np.mean(np.sum((coords - centroid) ** 2, axis=1))))
    return Anchor(
        xyz=tuple(float(v) for v in centroid),
        source="residues",
        residues_matched=sorted(f"{c or '_'}:{r}{i}" for (c, r, i) in matched_keys),
        chains_matched=[c or "_" for c in chains],
        atom_count=len(matched),
        radius_of_gyration=round(rog, 3),
        warnings=warnings,
    )


def anchor_from_reference_ligand(
    ref_ligand_path: Path,
    allow_leakage: bool = False,
) -> Anchor:
    if not allow_leakage:
        raise LeakageError(
            "Deriving the bias anchor from the reference ligand leaks the answer "
            "into a redocking benchmark; results would not be interpretable. "
            "Pass allow_leakage=True (CLI: --allow-leaky-anchor) only for "
            "sanity checks, never for reported numbers."
        )
    model = read_first_model(Path(ref_ligand_path))
    coords = model.coords(heavy_only=True)
    if coords.size == 0:
        raise AnchorError(f"No heavy atoms in reference ligand {ref_ligand_path}")
    centroid = coords.mean(axis=0)
    rog = float(np.sqrt(np.mean(np.sum((coords - centroid) ** 2, axis=1))))
    return Anchor(
        xyz=tuple(float(v) for v in centroid),
        source="ref_ligand(LEAKY)",
        residues_matched=[],
        chains_matched=[],
        atom_count=int(coords.shape[0]),
        radius_of_gyration=round(rog, 3),
        warnings=["Anchor derived from the reference ligand: results are leaky."],
    )


def resolve_anchor(
    receptor_path: Path,
    center: Sequence[float],
    size: Sequence[float],
    residue_specs: Optional[Sequence[ResidueSpec]] = None,
    ref_ligand_path: Optional[Path] = None,
    allow_leakage: bool = False,
) -> Optional[Anchor]:
    """Return an anchor, or None when no bias information is available."""
    if residue_specs:
        return anchor_from_residues(receptor_path, residue_specs, center, size)
    if ref_ligand_path is not None:
        return anchor_from_reference_ligand(ref_ligand_path, allow_leakage=allow_leakage)
    return None
