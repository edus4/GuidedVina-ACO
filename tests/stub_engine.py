#!/usr/bin/env python3
"""
A fake Vina-family binary used by the test suite.

It advertises the same options as a bias-capable build, and it moves the
ligand towards the bias point when a bias file is supplied, so the tests can
verify that bias genuinely reaches the engine rather than only appearing in a
report. It is not a docking program and performs no energy evaluation.

A second mode, --pretend-no-bias, advertises no bias option, so the tests can
check that the adapter refuses to run a biased stage on an incapable binary.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

HELP_WITH_BIAS = """
Input:
  --receptor arg          rigid part of the receptor (PDBQT)
  --ligand arg            ligand (PDBQT)
  --bias arg              bias parameter file (BPF)
Search space:
  --center_x arg          X coordinate of the center
  --center_y arg          Y coordinate of the center
  --center_z arg          Z coordinate of the center
  --size_x arg            size in the X dimension (Angstroms)
  --size_y arg            size in the Y dimension (Angstroms)
  --size_z arg            size in the Z dimension (Angstroms)
Output:
  --out arg               output models (PDBQT)
Misc:
  --cpu arg               the number of CPUs to use
  --seed arg              explicit random seed
  --exhaustiveness arg    exhaustiveness of the global search
  --max_evals arg         number of evaluations in each MC run
  --num_modes arg         maximum number of binding modes to generate
  --local_only            do local search only
  --score_only            score only - search space can be omitted
stub-engine version 0.1
"""

HELP_NO_BIAS = HELP_WITH_BIAS.replace(
    "  --bias arg              bias parameter file (BPF)\n", ""
)


def read_bias_point(path: Path):
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 6:
                return (float(parts[0]), float(parts[1]), float(parts[2])), float(parts[3])
    except OSError:
        return None, 0.0
    return None, 0.0


def read_ligand(path: Path):
    header, atoms = [], []
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            atoms.append(line)
        elif line.startswith(("REMARK", "ROOT", "ENDROOT", "BRANCH", "ENDBRANCH", "TORSDOF")):
            header.append(line)
    return header, atoms


def shift_line(line: str, dx: float, dy: float, dz: float) -> str:
    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
    return f"{line[:30]}{x + dx:8.3f}{y + dy:8.3f}{z + dz:8.3f}{line[54:]}"


def main() -> int:
    if "--help" in sys.argv or len(sys.argv) == 1:
        sys.stdout.write(HELP_NO_BIAS if "--pretend-no-bias" in sys.argv else HELP_WITH_BIAS)
        return 0

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--receptor"); p.add_argument("--ligand")
    p.add_argument("--bias"); p.add_argument("--out")
    p.add_argument("--center_x", type=float, default=0.0)
    p.add_argument("--center_y", type=float, default=0.0)
    p.add_argument("--center_z", type=float, default=0.0)
    p.add_argument("--size_x", type=float, default=20.0)
    p.add_argument("--size_y", type=float, default=20.0)
    p.add_argument("--size_z", type=float, default=20.0)
    p.add_argument("--exhaustiveness", type=int, default=8)
    p.add_argument("--max_evals", type=int, default=None)
    p.add_argument("--num_modes", type=int, default=9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", type=int, default=1)
    p.add_argument("--local_only", action="store_true")
    p.add_argument("--score_only", action="store_true")
    p.add_argument("--pretend-no-bias", action="store_true")
    args, _ = p.parse_known_args()

    if not args.ligand or not args.out:
        sys.stderr.write("stub: --ligand and --out are required\n")
        return 2

    header, atoms = read_ligand(Path(args.ligand))
    if not atoms:
        sys.stderr.write("stub: no atoms in ligand\n")
        return 3

    rng = random.Random(args.seed)
    centroid = [
        sum(float(a[30:38]) for a in atoms) / len(atoms),
        sum(float(a[38:46]) for a in atoms) / len(atoms),
        sum(float(a[46:54]) for a in atoms) / len(atoms),
    ]
    target = [args.center_x, args.center_y, args.center_z]
    pull = 0.3
    bias_point, energy = (None, 0.0)
    if args.bias and not args.pretend_no_bias:
        bias_point, energy = read_bias_point(Path(args.bias))
        if bias_point:
            target = list(bias_point)
            pull = min(0.95, 0.3 + abs(energy) / 2.0 * 0.5)

    # Tighter spread when biased: that is the collapse the guardrail watches for.
    spread = 3.0 * (1.0 - 0.7 * (abs(energy) / 2.0)) if bias_point else 3.0
    if args.local_only:
        spread *= 0.3

    lines = []
    n_modes = max(1, min(args.num_modes, 9))
    for mode in range(n_modes):
        jitter = [rng.gauss(0.0, spread) for _ in range(3)]
        d = [
            (target[i] - centroid[i]) * pull + jitter[i]
            for i in range(3)
        ]
        lines.append(f"MODEL {mode + 1}")
        lines.append(f"REMARK VINA RESULT:  {-9.0 + mode * 0.4:.1f}      0.000      0.000")
        lines.extend(header)
        lines.extend(shift_line(a, d[0], d[1], d[2]) for a in atoms)
        lines.append("ENDMDL")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.write(f"stub: wrote {n_modes} modes to {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
