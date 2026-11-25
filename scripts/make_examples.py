#!/usr/bin/env python3
"""
Regenerate the toy example inputs in examples/.

These are synthetic coordinates, not a real complex: they exist so the CLI,
the annealing schedule and the RMSD code can be exercised without downloading
anything. They must never be used to produce a performance number.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

EX = Path(__file__).resolve().parent.parent / "examples"
RESIDUES = [
    ("GLY", 12, np.array([3.0, 1.0, 0.5])),
    ("LYS", 60, np.array([-2.5, 2.0, -1.0])),
    ("ASP", 61, np.array([-1.0, -2.5, 1.5])),
    ("VAL", 95, np.array([6.0, -5.0, 4.0])),
]
SIDECHAIN = [("CB", "C"), ("CG", "C"), ("OD1", "OA"), ("NZ", "NA")]
BACKBONE = [("N", "NA"), ("CA", "C"), ("C", "C"), ("O", "OA")]


def atom_line(serial, name, res_name, chain, res_seq, xyz, ad_type, charge=0.0, het=False):
    rec = "HETATM" if het else "ATOM  "
    nm = f" {name:<3s}" if len(name) < 4 else name[:4]
    return (
        f"{rec}{serial:5d} {nm}{'':1s}{res_name:>3s} {chain}{res_seq:4d}    "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00    "
        f"{charge:6.3f} {ad_type:<2s}"
    )


def write_receptor(path: Path, chains=("A",)):
    lines = ["REMARK  Synthetic receptor for GuidedVina-ACO smoke tests"]
    serial = 1
    for chain_i, chain in enumerate(chains):
        shift = np.array([0.0, 0.0, 0.0]) if chain_i == 0 else np.array([25.0, 0.0, 0.0])
        for res_name, res_seq, base in RESIDUES:
            for k, (name, ad) in enumerate(BACKBONE):
                off = np.array([0.6 * k, 0.3 * k, -0.2 * k])
                lines.append(atom_line(serial, name, res_name, chain, res_seq,
                                       base + shift + off, ad))
                serial += 1
            for k, (name, ad) in enumerate(SIDECHAIN):
                off = np.array([-0.5 * k, 0.8 + 0.4 * k, 0.5 * k])
                lines.append(atom_line(serial, name, res_name, chain, res_seq,
                                       base + shift + off, ad))
                serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


LIGAND_ATOMS = [
    ("C1", "C"), ("C2", "A"), ("C3", "A"), ("C4", "A"),
    ("C5", "A"), ("C6", "A"), ("N1", "NA"), ("O1", "OA"),
    ("O2", "OA"), ("C7", "C"), ("C8", "C"),
]


def ligand_coords(radius=1.4, twist=0.0, shift=np.zeros(3)):
    pts = []
    for i, _ in enumerate(LIGAND_ATOMS):
        ang = 2 * math.pi * i / len(LIGAND_ATOMS) + twist
        pts.append(np.array([radius * math.cos(ang), radius * math.sin(ang), 0.3 * i]) + shift)
    return pts


def write_ligand(path: Path, coords, torsions=4):
    lines = [
        f"REMARK  {torsions} active torsions:",
        "REMARK  status: ('A' for Active; 'I' for Inactive)",
        "ROOT",
    ]
    serial = 1
    for (name, ad), xyz in zip(LIGAND_ATOMS, coords):
        if serial == 5:
            lines.append("ENDROOT")
            for t in range(torsions):
                lines.append(f"BRANCH {serial} {serial + 1}")
        lines.append(atom_line(serial, name, "LIG", "A", 1, xyz, ad, het=True))
        serial += 1
    for _ in range(torsions):
        lines.append("ENDBRANCH")
    lines.append(f"TORSDOF {torsions}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    EX.mkdir(parents=True, exist_ok=True)
    write_receptor(EX / "sample_receptor.pdbqt", chains=("A",))
    write_receptor(EX / "sample_receptor_dimer.pdbqt", chains=("A", "B"))
    start = ligand_coords(shift=np.array([4.0, 4.0, 2.0]), twist=0.7)
    ref = ligand_coords(shift=np.array([0.0, 0.3, 0.0]))
    write_ligand(EX / "sample_ligand.pdbqt", start)
    write_ligand(EX / "sample_ligand_ref.pdbqt", ref)
    manifest = EX / "sample_manifest.csv"
    manifest.write_text(
        "complex_id,receptor,ligand_input,ligand_ref,center_x,center_y,center_z,"
        "size_x,size_y,size_z,critical_residues\n"
        "toy_monomer,sample_receptor.pdbqt,sample_ligand.pdbqt,sample_ligand_ref.pdbqt,"
        "0.0,0.0,0.0,20.0,20.0,20.0,A:12;A:60;A:61\n"
        "toy_dimer,sample_receptor_dimer.pdbqt,sample_ligand.pdbqt,sample_ligand_ref.pdbqt,"
        "0.0,0.0,0.0,20.0,20.0,20.0,A:12;A:60\n",
        encoding="utf-8",
    )
    print(f"wrote examples into {EX}")


if __name__ == "__main__":
    main()
