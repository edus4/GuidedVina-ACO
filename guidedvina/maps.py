"""
Bias applied through AutoGrid affinity maps.

Vina-ACO has no bias option: it is AutoDock Vina 1.2.7 with an ant-colony
global search, and nothing in its command line accepts a bias specification.
It does, however, expose ``--write_maps`` and ``--maps``, and the scoring
function reads the ligand's energy out of those grids. So bias is applied by
writing the maps, adding an attractive well at the anchor, and docking against
the modified grids. This is the same idea as AutoDock-Bias, implemented
outside the binary.

Map file format, as written by ``cache::write``::

    GRID_PARAMETER_FILE <gpf>
    GRID_DATA_FILE <fld>
    MACROMOLECULE <receptor>
    SPACING 0.375
    NELEMENTS 40 40 40
    CENTER 12.500 -4.200 18.000
    <value>            # (nz+1) * (ny+1) * (nx+1) values,
    ...                # x fastest, then y, then z

NELEMENTS counts voxels, so each axis holds NELEMENTS+1 points. Vina refuses
to write maps when any NELEMENTS is odd, hence ``--force_even_voxels``.

Map files are named ``<prefix>.<XS type>.map`` using Vina's internal typing,
which encodes hydrogen-bonding role: ``_D`` donor, ``_A`` acceptor, ``_DA``
both, ``_P`` polar non-bonding, ``_H`` hydrophobic. That is what makes typed
bias possible: biasing only ``O_A``/``N_A`` pulls acceptors into the region
without dragging the whole ligand there.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: Vina XS atom types grouped by hydrogen-bonding role.
DONOR_TYPES = ("N_D", "N_DA", "O_D", "O_DA", "Met_D")
ACCEPTOR_TYPES = ("N_A", "N_DA", "O_A", "O_DA")
APOLAR_TYPES = ("C_H", "F_H", "Cl_H", "Br_H", "I_H")

#: Bias targets. "map" biases every atom type, as a type-agnostic well.
BIAS_TARGETS = {
    "map": None,
    "don": DONOR_TYPES,
    "acc": ACCEPTOR_TYPES,
    "apolar": APOLAR_TYPES,
}

#: Voxels whose unbiased energy is above this are inside the receptor; biasing
#: them would carve an artificial minimum into the protein interior.
REPULSION_CUTOFF = 2.0

_MAP_NAME = re.compile(r"^(?P<prefix>.+)\.(?P<type>[A-Za-z]+[A-Za-z_0-9]*)\.map$")


class MapError(RuntimeError):
    pass


@dataclass
class AffinityMap:
    path: Path
    header: List[str]
    spacing: float
    nelements: Tuple[int, int, int]
    center: Tuple[float, float, float]
    values: np.ndarray            # shape (nz+1, ny+1, nx+1), indexed [z, y, x]

    @property
    def atom_type(self) -> str:
        m = _MAP_NAME.match(self.path.name)
        return m.group("type") if m else ""

    @property
    def origin(self) -> np.ndarray:
        """Coordinates of grid point (0, 0, 0)."""
        half = np.array(self.nelements, dtype=float) * self.spacing / 2.0
        return np.array(self.center, dtype=float) - half

    def grid_coordinates(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        nx, ny, nz = self.nelements
        ox, oy, oz = self.origin
        xs = ox + np.arange(nx + 1) * self.spacing
        ys = oy + np.arange(ny + 1) * self.spacing
        zs = oz + np.arange(nz + 1) * self.spacing
        return xs, ys, zs


def read_map(path: Path) -> AffinityMap:
    path = Path(path)
    header: List[str] = []
    spacing: Optional[float] = None
    nelements: Optional[Tuple[int, int, int]] = None
    center: Optional[Tuple[float, float, float]] = None
    values: List[float] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            key = stripped.split()[0].upper()
            if key in {"GRID_PARAMETER_FILE", "GRID_DATA_FILE", "MACROMOLECULE",
                       "SPACING", "NELEMENTS", "CENTER"}:
                header.append(stripped)
                parts = stripped.split()
                if key == "SPACING":
                    spacing = float(parts[1])
                elif key == "NELEMENTS":
                    nelements = (int(parts[1]), int(parts[2]), int(parts[3]))
                elif key == "CENTER":
                    center = (float(parts[1]), float(parts[2]), float(parts[3]))
                continue
            try:
                values.append(float(stripped))
            except ValueError as exc:
                raise MapError(f"{path}: unparsable line {stripped!r}") from exc

    if spacing is None or nelements is None or center is None:
        raise MapError(f"{path}: header is missing SPACING, NELEMENTS or CENTER")
    nx, ny, nz = nelements
    expected = (nx + 1) * (ny + 1) * (nz + 1)
    if len(values) != expected:
        raise MapError(
            f"{path}: expected {expected} values for NELEMENTS {nx} {ny} {nz}, found {len(values)}"
        )
    arr = np.array(values, dtype=float).reshape((nz + 1, ny + 1, nx + 1))
    return AffinityMap(path=path, header=header, spacing=spacing,
                       nelements=nelements, center=center, values=arr)


def write_map(amap: AffinityMap, path: Optional[Path] = None) -> Path:
    """Write a map back out, preserving the header verbatim."""
    out = Path(path) if path else amap.path
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for line in amap.header:
            fh.write(line + "\n")
        for z in range(amap.values.shape[0]):
            for y in range(amap.values.shape[1]):
                for x in range(amap.values.shape[2]):
                    fh.write(f"{amap.values[z, y, x]:.4f}\n")
    return out


def find_maps(prefix: Path) -> List[Path]:
    """All ``<prefix>.<type>.map`` files for a given prefix."""
    prefix = Path(prefix)
    return sorted(prefix.parent.glob(f"{prefix.name}.*.map"))


def gaussian_well(
    amap: AffinityMap,
    anchor: Sequence[float],
    depth: float,
    radius: float,
    repulsion_cutoff: float = REPULSION_CUTOFF,
) -> Tuple[np.ndarray, int]:
    """
    An attractive Gaussian well of ``depth`` kcal/mol (a positive number) at
    ``anchor``, with sigma = radius/2 so that the well is near zero by
    ``radius``. Returns the modified values and the number of voxels changed.

    Voxels already more repulsive than ``repulsion_cutoff`` are left alone:
    inside the receptor the unbiased energy is large and positive, and
    lowering it there would invent a binding site in the protein interior.
    """
    if depth <= 0.0 or radius <= 0.0:
        return amap.values.copy(), 0
    xs, ys, zs = amap.grid_coordinates()
    ax, ay, az = (float(v) for v in anchor)
    sigma = radius / 2.0

    # Separable squared distances keep this O(n) in memory per axis.
    dx2 = (xs - ax) ** 2
    dy2 = (ys - ay) ** 2
    dz2 = (zs - az) ** 2
    r2 = dz2[:, None, None] + dy2[None, :, None] + dx2[None, None, :]

    well = -depth * np.exp(-r2 / (2.0 * sigma ** 2))
    well[r2 > (3.0 * sigma) ** 2] = 0.0          # truncate the tail
    mask = (amap.values <= repulsion_cutoff) & (well < 0.0)
    out = amap.values.copy()
    out[mask] += well[mask]
    return out, int(np.count_nonzero(mask))


@dataclass
class BiasApplication:
    source_prefix: str
    output_prefix: str
    anchor: Tuple[float, float, float]
    depth: float
    radius: float
    target: str
    maps_written: List[str]
    maps_biased: List[str]
    voxels_changed: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "source_prefix": self.source_prefix,
            "output_prefix": self.output_prefix,
            "anchor": list(self.anchor),
            "depth_kcal_per_mol": self.depth,
            "radius_angstrom": self.radius,
            "target": self.target,
            "n_maps_written": len(self.maps_written),
            "n_maps_biased": len(self.maps_biased),
            "maps_biased": self.maps_biased,
            "voxels_changed": self.voxels_changed,
        }


def apply_bias_to_maps(
    source_prefix: Path,
    output_prefix: Path,
    anchor: Sequence[float],
    depth: float,
    radius: float,
    target: str = "map",
) -> BiasApplication:
    """
    Copy a set of maps to ``output_prefix``, adding the well to those whose
    atom type is covered by ``target``.

    Every map is written, biased or not, because Vina needs the complete set
    for the ligand's atom types; only the targeted ones are modified. A target
    that matches no map present is an error rather than a silent no-op, since
    that would dock unbiased while reporting bias.
    """
    if target not in BIAS_TARGETS:
        raise MapError(f"Unknown bias target {target!r}; expected one of {sorted(BIAS_TARGETS)}")
    sources = find_maps(source_prefix)
    if not sources:
        raise MapError(
            f"No maps found at prefix {source_prefix}. Run the engine with "
            "--write_maps (and --force_even_voxels if it complains about odd voxel counts)."
        )
    wanted = BIAS_TARGETS[target]
    written: List[str] = []
    biased: List[str] = []
    total_voxels = 0
    anchor_t = (float(anchor[0]), float(anchor[1]), float(anchor[2]))

    for src in sources:
        amap = read_map(src)
        atom_type = amap.atom_type
        dest = Path(f"{output_prefix}.{atom_type}.map")
        if wanted is None or atom_type in wanted:
            values, changed = gaussian_well(amap, anchor_t, depth, radius)
            if changed:
                amap.values = values
                biased.append(atom_type)
                total_voxels += changed
        write_map(amap, dest)
        written.append(str(dest))

    if not biased:
        present = sorted(read_map(s).atom_type for s in sources)
        raise MapError(
            f"Bias target {target!r} matched no map among {present}, or the anchor "
            "lies outside the grid / entirely inside the receptor, so no voxel was "
            "changed. Docking would have run unbiased."
        )
    return BiasApplication(
        source_prefix=str(source_prefix),
        output_prefix=str(output_prefix),
        anchor=anchor_t,
        depth=depth,
        radius=radius,
        target=target,
        maps_written=written,
        maps_biased=biased,
        voxels_changed=total_voxels,
    )
