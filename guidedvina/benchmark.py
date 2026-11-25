"""
Paired batch benchmark.

Every complex is docked by both arms (baseline = engine defaults, no bias;
guided = selected budget + staged bias) with the same set of seeds, so the
comparison is paired and the stochasticity of docking is visible rather than
hidden behind a single run.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .anchors import parse_residue_specs
from .engine import EngineCapabilities
from .runner import DockingOutcome, dock_once
from .stats import compare_arms

MANIFEST_COLUMNS = (
    "complex_id", "receptor", "ligand_input", "ligand_ref",
    "center_x", "center_y", "center_z",
    "size_x", "size_y", "size_z", "critical_residues",
)


class ManifestError(ValueError):
    pass


@dataclass
class ManifestRow:
    complex_id: str
    receptor: Path
    ligand_input: Path
    ligand_ref: Optional[Path]
    center: tuple
    size: tuple
    critical_residues: List[str]


def _float_or(value: Optional[str], default: float, field: str, cid: str) -> float:
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ManifestError(f"{cid}: column {field} is not a number: {value!r}") from exc


def read_manifest(path: Path, base_dir: Optional[Path] = None) -> List[ManifestRow]:
    path = Path(path)
    base = Path(base_dir) if base_dir else path.parent
    rows: List[ManifestRow] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = {"complex_id", "receptor", "ligand_input",
                   "center_x", "center_y", "center_z"} - set(reader.fieldnames or [])
        if missing:
            raise ManifestError(
                f"{path}: manifest is missing required columns {sorted(missing)}. "
                f"Expected header: {','.join(MANIFEST_COLUMNS)}"
            )
        for i, raw in enumerate(reader, start=2):
            cid = (raw.get("complex_id") or "").strip()
            if not cid:
                raise ManifestError(f"{path}:{i}: empty complex_id")

            def resolve(key: str) -> Optional[Path]:
                v = (raw.get(key) or "").strip()
                if not v:
                    return None
                p = Path(v)
                return p if p.is_absolute() else (base / p)

            receptor = resolve("receptor")
            ligand = resolve("ligand_input")
            if receptor is None or ligand is None:
                raise ManifestError(f"{path}:{i}: receptor and ligand_input are required")
            rows.append(
                ManifestRow(
                    complex_id=cid,
                    receptor=receptor,
                    ligand_input=ligand,
                    ligand_ref=resolve("ligand_ref"),
                    center=(
                        _float_or(raw.get("center_x"), 0.0, "center_x", cid),
                        _float_or(raw.get("center_y"), 0.0, "center_y", cid),
                        _float_or(raw.get("center_z"), 0.0, "center_z", cid),
                    ),
                    size=(
                        _float_or(raw.get("size_x"), 20.0, "size_x", cid),
                        _float_or(raw.get("size_y"), 20.0, "size_y", cid),
                        _float_or(raw.get("size_z"), 20.0, "size_z", cid),
                    ),
                    critical_residues=[(raw.get("critical_residues") or "").strip()],
                )
            )
    if not rows:
        raise ManifestError(f"{path}: no data rows")
    return rows


def run_benchmark(
    caps: EngineCapabilities,
    manifest_path: Path,
    out_dir: Path,
    selector,
    seeds: Sequence[int] = (1, 2, 3),
    arms: Sequence[str] = ("baseline", "guided"),
    rmsd_cutoff: float = 2.0,
    rmsd_method: str = "auto",
    success_rule: str = "majority",
    allow_leaky_anchor: bool = False,
    allow_rb_estimate: bool = False,
    use_bias: bool = True,
    cpu: Optional[int] = None,
    dry_run: bool = False,
    timeout: Optional[float] = None,
    progress: bool = True,
) -> Dict[str, object]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(manifest_path)

    records: List[Dict[str, object]] = []
    per_complex: Dict[str, Dict[str, Dict[str, object]]] = {}
    outcomes: List[DockingOutcome] = []

    total = len(rows) * len(arms) * len(seeds)
    done = 0
    for row in rows:
        specs = parse_residue_specs(row.critical_residues) if any(row.critical_residues) else []
        for arm in arms:
            for seed in seeds:
                out_path = out_dir / "poses" / f"{row.complex_id}_{arm}_s{seed}.pdbqt"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                outcome = dock_once(
                    caps=caps,
                    complex_id=row.complex_id,
                    receptor=row.receptor,
                    ligand=row.ligand_input,
                    center=row.center,
                    size=row.size,
                    out_path=out_path,
                    arm=arm,
                    selector=selector,
                    seed=seed,
                    residue_specs=specs,
                    reference_ligand=row.ligand_ref,
                    allow_leaky_anchor=allow_leaky_anchor,
                    allow_rb_estimate=allow_rb_estimate,
                    rmsd_cutoff=rmsd_cutoff,
                    rmsd_method=rmsd_method,
                    use_bias=use_bias,
                    cpu=cpu,
                    dry_run=dry_run,
                    timeout=timeout,
                    work_dir=out_dir / "work" / row.complex_id,
                )
                outcomes.append(outcome)
                rec = outcome.to_record(rmsd_cutoff=rmsd_cutoff)
                records.append(rec)
                done += 1
                if progress:
                    status = "ok" if not outcome.error else "FAIL"
                    print(
                        f"[{done}/{total}] {row.complex_id} {arm} seed={seed} "
                        f"rmsd={rec['top1_rmsd']} {status}",
                        flush=True,
                    )
                if outcome.success is not None:
                    slot = per_complex.setdefault(row.complex_id, {}).setdefault(
                        arm, {"successes": [], "runtimes": []}
                    )
                    slot["successes"].append(bool(outcome.success))
                    slot["runtimes"].append(
                        outcome.run.total_runtime_sec if outcome.run else 0.0
                    )

    runs_csv = out_dir / "runs.csv"
    if records:
        with open(runs_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    comparison = compare_arms(per_complex, success_rule=success_rule)
    methods = sorted({r["rmsd_method"] for r in records if r["rmsd_method"]})
    summary: Dict[str, object] = {
        "manifest": str(manifest_path),
        "engine": caps.to_dict(),
        "selector": getattr(selector, "name", str(selector)),
        "arms": list(arms),
        "seeds": list(seeds),
        "rmsd_cutoff_angstrom": rmsd_cutoff,
        "rmsd_methods_used": methods,
        "n_runs": len(records),
        "n_failed_runs": sum(1 for r in records if r["error"]),
        "dry_run": dry_run,
        "comparison": comparison.to_dict(),
        "caveats": _caveats(records, methods, seeds, dry_run),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def _caveats(
    records: Sequence[Dict[str, object]],
    methods: Sequence[str],
    seeds: Sequence[int],
    dry_run: bool,
) -> List[str]:
    out: List[str] = []
    if dry_run:
        out.append("Dry run: no docking was performed, so no accuracy claim is supported.")
    if "ordered" in methods:
        out.append(
            "Some RMSD values used index-ordered comparison, which is not "
            "symmetry-corrected and may be inflated. Install RDKit or scipy."
        )
    if "assignment_lower_bound" in methods:
        out.append(
            "Some RMSD values used element-wise optimal assignment, a lower "
            "bound on graph-symmetry RMSD. Install RDKit for GetBestRMS."
        )
    if len(seeds) < 3:
        out.append(f"Only {len(seeds)} seed(s) per arm: run >=3 for a stable comparison.")
    if any(r["rb_source"] == "estimated" for r in records):
        out.append(
            "Some rotatable-bond counts were estimated from heavy-atom count; "
            "budget selection for those complexes is unreliable."
        )
    if any(r["anchor_source"] == "ref_ligand(LEAKY)" for r in records):
        out.append(
            "Bias anchors were derived from reference ligands: these numbers are "
            "leaky and must not be reported as prospective performance."
        )
    if any(not r["bias_applied"] for r in records if r["arm"] == "guided"):
        out.append(
            "At least one guided run applied no bias (no anchor available), so it "
            "tested budget selection only."
        )
    return out
