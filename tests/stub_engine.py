#!/usr/bin/env python3
"""
A fake Vina-ACO used by the test suite.

It advertises the real option set of Vina-ACO (AutoDock Vina 1.2.7 plus the
ACO search), writes genuine AutoGrid .map files for --write_maps, and reads
them back for --maps. Poses are pulled towards the lowest-energy voxel of the
loaded maps, so a well added to the maps visibly moves the output: that is how
the tests check that map-based bias reaches the engine rather than only
appearing in a report.

It performs no energy evaluation and is not a docking program.

--pretend-no-maps advertises neither --maps nor --write_maps, so the tests can
check that an un-biasable binary causes a refusal.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

HELP = """Input:
  --receptor arg             rigid part of the receptor (PDBQT)
  --flex arg                 flexible side chains, if any (PDBQT)
  --ligand arg               ligand (PDBQT)
  --batch arg                batch directory or ligands (PDBQT)
  --scoring arg (=vina)      scoring function (ad4, vina or vinardo)
Search space (required):
  --maps arg                 affinity maps for the ad4 or vina scoring function
  --center_x arg             X coordinate of the center (Angstrom)
  --center_y arg             Y coordinate of the center (Angstrom)
  --center_z arg             Z coordinate of the center (Angstrom)
  --size_x arg               size in the X dimension (Angstrom)
  --size_y arg               size in the Y dimension (Angstrom)
  --size_z arg               size in the Z dimension (Angstrom)
Output (optional):
  --out arg                  output models (PDBQT)
  --dir arg                  output directory for batch mode
  --write_maps arg           output filename (directory + prefix name) for maps
Advanced options (see the manual):
  --score_only               score only - search space can be omitted
  --local_only               do local search only
  --force_even_voxels        calculated grid maps will have an even number of voxels
Misc (optional):
  --cpu arg                  the number of CPUs to use
  --seed arg                 explicit random seed
  --exhaustiveness arg (=8)  exhaustiveness of the global search: 1+
  --max_evals arg (=0)       number of evaluations in each MC run
  --num_modes arg (=9)       maximum number of binding modes to generate
  --min_rmsd arg (=1)        minimum RMSD between output poses
ACO options:
  --search arg (=mc)         global search algorithm: mc or aco
  --aco_preset arg           ACO parameter set: fast, balanced or accurate
  --aco_budget arg           ACO: evaluations per colony
  --aco_variant arg          ACO: acor or mmas
  --aco_ants arg             ACO: ants per iteration (10)
  --aco_archive arg          ACO_R: solution archive size (10)
Information (optional):
  --help                     display usage summary
  --help_advanced            display usage summary with advanced options
  --version                  display program version

Vina-ACO 1.2.7-stub
"""

HELP_NO_MAPS = "\n".join(
    line for line in HELP.splitlines()
    if "--maps" not in line and "--write_maps" not in line
)

MAP_TYPES = ("C_H", "C_P", "N_A", "N_D", "O_A", "O_DA")
NELEMENTS = 20          # even, as the .map format requires
SPACING = 1.0


def write_maps(prefix: Path, center, size) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    n = NELEMENTS
    origin = [center[i] - n * SPACING / 2.0 for i in range(3)]
    for atom_type in MAP_TYPES:
        path = Path(f"{prefix}.{atom_type}.map")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("GRID_PARAMETER_FILE NULL\n")
            fh.write("GRID_DATA_FILE NULL\n")
            fh.write("MACROMOLECULE receptor.pdbqt\n")
            fh.write(f"SPACING {SPACING}\n")
            fh.write(f"NELEMENTS {n} {n} {n}\n")
            fh.write(f"CENTER {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}\n")
            for iz in range(n + 1):
                for iy in range(n + 1):
                    for ix in range(n + 1):
                        # A shallow bowl centred on the box, so an added well
                        # dominates and the minimum moves to the anchor.
                        dx = origin[0] + ix * SPACING - center[0]
                        dy = origin[1] + iy * SPACING - center[1]
                        dz = origin[2] + iz * SPACING - center[2]
                        fh.write(f"{0.001 * (dx * dx + dy * dy + dz * dz):.4f}\n")


def read_min_voxel(prefix: str):
    """Coordinates of the lowest-energy grid point across the loaded maps."""
    best = None
    for path in sorted(Path(prefix).parent.glob(f"{Path(prefix).name}.*.map")):
        spacing = center = None
        n = None
        values = []
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parts = line.split()
                if not parts:
                    continue
                key = parts[0].upper()
                if key == "SPACING":
                    spacing = float(parts[1])
                elif key == "NELEMENTS":
                    n = (int(parts[1]), int(parts[2]), int(parts[3]))
                elif key == "CENTER":
                    center = (float(parts[1]), float(parts[2]), float(parts[3]))
                elif key in {"GRID_PARAMETER_FILE", "GRID_DATA_FILE", "MACROMOLECULE"}:
                    continue
                else:
                    try:
                        values.append(float(parts[0]))
                    except ValueError:
                        continue
        if spacing is None or center is None or n is None or not values:
            continue
        origin = [center[i] - n[i] * spacing / 2.0 for i in range(3)]
        idx = min(range(len(values)), key=values.__getitem__)
        nx, ny = n[0] + 1, n[1] + 1
        ix = idx % nx
        iy = (idx // nx) % ny
        iz = idx // (nx * ny)
        point = (origin[0] + ix * spacing,
                 origin[1] + iy * spacing,
                 origin[2] + iz * spacing)
        if best is None or values[idx] < best[0]:
            best = (values[idx], point)
    return best[1] if best else None


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
    no_maps = "--pretend-no-maps" in sys.argv
    if "--help" in sys.argv or "--help_advanced" in sys.argv or "--version" in sys.argv \
            or len(sys.argv) == 1:
        sys.stdout.write(HELP_NO_MAPS if no_maps else HELP)
        return 0

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--receptor"); p.add_argument("--ligand")
    p.add_argument("--maps"); p.add_argument("--write_maps")
    p.add_argument("--out")
    for axis in "xyz":
        p.add_argument(f"--center_{axis}", type=float, default=0.0)
        p.add_argument(f"--size_{axis}", type=float, default=20.0)
    p.add_argument("--exhaustiveness", type=int, default=8)
    p.add_argument("--max_evals", type=int, default=0)
    p.add_argument("--num_modes", type=int, default=9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", type=int, default=1)
    p.add_argument("--search", default="mc")
    p.add_argument("--aco_preset", default="balanced")
    p.add_argument("--local_only", action="store_true")
    p.add_argument("--score_only", action="store_true")
    p.add_argument("--force_even_voxels", action="store_true")
    p.add_argument("--pretend-no-maps", action="store_true")
    args, _ = p.parse_known_args()

    center = (args.center_x, args.center_y, args.center_z)

    if args.write_maps:
        if no_maps:
            sys.stderr.write("stub: unrecognised option --write_maps\n")
            return 2
        write_maps(Path(args.write_maps), center, (args.size_x, args.size_y, args.size_z))
        sys.stdout.write(f"stub: wrote maps to {args.write_maps}\n")
        if args.score_only:
            sys.stdout.write("Estimated Free Energy of Binding: -6.0 kcal/mol\n")
            return 0

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

    target = list(center)
    spread = 3.0
    if args.maps:
        minimum = read_min_voxel(args.maps)
        if minimum is not None:
            target = list(minimum)
            spread = 1.0          # the well concentrates the ensemble
    if args.local_only:
        target = list(centroid)
        spread *= 0.3

    lines = []
    n_modes = max(1, min(args.num_modes, 9))
    for mode in range(n_modes):
        jitter = [rng.gauss(0.0, spread) for _ in range(3)]
        d = [(target[i] - centroid[i]) * 0.9 + jitter[i] for i in range(3)]
        lines.append(f"MODEL {mode + 1}")
        lines.append(f"REMARK VINA RESULT:  {-9.0 + mode * 0.4:.1f}      0.000      0.000")
        lines.extend(header)
        lines.extend(shift_line(a, d[0], d[1], d[2]) for a in atoms)
        lines.append("ENDMDL")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.write(f"stub: {args.search}/{args.aco_preset} wrote {n_modes} modes to {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
