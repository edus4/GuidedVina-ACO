"""
Receptor/ligand preparation.

Conversion is only attempted for a real docking run, and the converter used is
recorded, because the atom ordering a converter produces affects how RMSD must
be computed downstream.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class PreparationError(RuntimeError):
    pass


@dataclass
class Prepared:
    path: Path
    converter: str      # "none" | "obabel" | "meeko"
    original: Path


def prepare_structure(
    path: Path,
    role: str,
    work_dir: Path,
    skip: bool = False,
    ph: float = 7.4,
) -> Prepared:
    path = Path(path)
    if path.suffix.lower() == ".pdbqt" or skip:
        return Prepared(path=path, converter="none", original=path)
    if role not in {"receptor", "ligand"}:
        raise ValueError(f"role must be 'receptor' or 'ligand', got {role!r}")

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / f"{path.stem}_{role}.pdbqt"

    if role == "ligand" and shutil.which("mk_prepare_ligand.py"):
        _run(["mk_prepare_ligand.py", "-i", str(path), "-o", str(out)])
        return Prepared(path=out, converter="meeko", original=path)
    if role == "receptor" and shutil.which("mk_prepare_receptor.py"):
        _run(["mk_prepare_receptor.py", "-i", str(path), "-o", str(out.with_suffix(""))])
        produced = out.with_suffix(".pdbqt")
        if produced.exists():
            return Prepared(path=produced, converter="meeko", original=path)
    if shutil.which("obabel"):
        cmd = ["obabel", str(path), "-O", str(out), "-p", str(ph)]
        if role == "receptor":
            cmd.append("-xr")
        _run(cmd)
        if not out.exists():
            raise PreparationError(f"obabel did not produce {out}")
        return Prepared(path=out, converter="obabel", original=path)

    raise PreparationError(
        f"Cannot convert {role} {path} to .pdbqt: no converter found. Install "
        "Open Babel ('obabel') or Meeko, or supply pre-converted .pdbqt inputs. "
        "Note that converter choice changes atom ordering, so prefer converting "
        "reference and docked ligands with the same tool."
    )


def _run(cmd) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise PreparationError(
            f"{cmd[0]} failed ({proc.returncode}): {(proc.stderr or '')[-500:]}"
        )
