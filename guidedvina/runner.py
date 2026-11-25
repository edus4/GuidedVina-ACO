"""One complex, one arm, one seed: the single unit of work."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .anchors import Anchor, ResidueSpec, resolve_anchor
from .engine import EngineCapabilities, RunResult, run_staged
from .features import LigandFeatures, PocketFeatures, ligand_features, pocket_features
from .planner import SearchPlan, baseline_plan
from .prepare import prepare_structure
from .rmsd import RmsdResult, pose_rmsd


@dataclass
class DockingOutcome:
    complex_id: str
    arm: str
    seed: int
    ligand: Optional[LigandFeatures]
    pocket: Optional[PocketFeatures]
    anchor: Optional[Anchor]
    plan: Optional[SearchPlan]
    run: Optional[RunResult]
    rmsd: Optional[RmsdResult]
    success: Optional[bool]
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_record(self, rmsd_cutoff: float = 2.0) -> Dict[str, object]:
        run = self.run
        stages = run.stages if run else []
        return {
            "complex_id": self.complex_id,
            "arm": self.arm,
            "seed": self.seed,
            "input_format": self.ligand.input_format if self.ligand else "",
            "rotatable_bonds": self.ligand.rotatable_bonds if self.ligand else "",
            "rb_source": self.ligand.rotatable_bonds_source if self.ligand else "",
            "heavy_atoms": self.ligand.heavy_atoms if self.ligand else "",
            "buriedness": self.pocket.buriedness if self.pocket else "",
            "selector": self.plan.selector if self.plan else "",
            "regime": self.plan.regime if self.plan else "",
            "search_method": self.plan.search_method if self.plan else "",
            "aco_preset": (self.plan.aco_preset or "") if self.plan else "",
            "exhaustiveness": self.plan.exhaustiveness if self.plan else "",
            "stages": len([s for s in stages if s.stage.index >= 0]),
            "initial_bias_weight": self.plan.initial_bias_weight if self.plan else "",
            "bias_backend": run.bias_backend if run else "",
            "bias_applied": any(s.biased for s in stages),
            "bias_target": self.plan.bias_target if self.plan else "",
            "voxels_biased": sum(
                int((s.bias_detail or {}).get("voxels_changed", 0) or 0) for s in stages
            ),
            "anchor_source": self.anchor.source if self.anchor else "none",
            "guardrail_triggered": any(s.stage.guardrail_applied for s in stages),
            "final_diversity": next(
                (s.diversity_after for s in reversed(stages) if s.diversity_after is not None),
                "",
            ),
            "top1_rmsd": self.rmsd.value if self.rmsd and self.rmsd.value is not None else "",
            "rmsd_method": self.rmsd.method if self.rmsd else "",
            "rmsd_cutoff": rmsd_cutoff,
            "success": "" if self.success is None else bool(self.success),
            "runtime_sec": round(run.total_runtime_sec, 3) if run else "",
            "error": self.error or (run.error if run else "") or "",
        }


def dock_once(
    caps: EngineCapabilities,
    complex_id: str,
    receptor: Path,
    ligand: Path,
    center: Sequence[float],
    size: Sequence[float],
    out_path: Path,
    arm: str,
    selector,
    seed: int = 42,
    residue_specs: Optional[Sequence[ResidueSpec]] = None,
    reference_ligand: Optional[Path] = None,
    allow_leaky_anchor: bool = False,
    rotatable_bonds: Optional[int] = None,
    allow_rb_estimate: bool = False,
    rmsd_cutoff: float = 2.0,
    rmsd_method: str = "auto",
    use_bias: bool = True,
    bias_target: str = "map",
    stages_override: Optional[int] = None,
    search_method: Optional[str] = None,
    aco_preset: Optional[str] = None,
    cpu: Optional[int] = None,
    dry_run: bool = False,
    timeout: Optional[float] = None,
    work_dir: Optional[Path] = None,
    base_map_prefix: Optional[Path] = None,
) -> DockingOutcome:
    out = DockingOutcome(
        complex_id=complex_id, arm=arm, seed=seed,
        ligand=None, pocket=None, anchor=None, plan=None, run=None,
        rmsd=None, success=None,
    )
    work_dir = Path(work_dir) if work_dir else Path(out_path).parent / "_guidedvina_work"
    try:
        lig = ligand_features(ligand, rotatable_bonds=rotatable_bonds,
                              allow_estimate=allow_rb_estimate)
        pock = pocket_features(receptor, center, size)
        out.ligand, out.pocket = lig, pock

        if arm == "baseline":
            plan = baseline_plan()
            anchor = None
        else:
            plan = selector.plan(lig, pock)
            anchor = None
            if use_bias:
                anchor = resolve_anchor(
                    receptor, center, size,
                    residue_specs=residue_specs,
                    ref_ligand_path=reference_ligand if not residue_specs else None,
                    allow_leakage=allow_leaky_anchor,
                )
                if anchor is None:
                    out.warnings.append(
                        "no anchor available (no critical residues given); "
                        "running the guided arm without bias"
                    )
            if anchor is None:
                plan.stages = 1
                plan.initial_bias_weight = 0.0
            if stages_override:
                plan.stages = int(stages_override)
            plan.bias_target = bias_target
            if anchor is not None:
                out.warnings.extend(anchor.warnings)
        if search_method:
            plan.search_method = search_method
            if search_method != "aco":
                plan.aco_preset = None
        if aco_preset and plan.search_method == "aco":
            plan.aco_preset = aco_preset
        out.plan, out.anchor = plan, anchor

        rec_prepared = prepare_structure(Path(receptor), "receptor", work_dir, skip=dry_run)
        lig_prepared = prepare_structure(Path(ligand), "ligand", work_dir, skip=dry_run)

        run = run_staged(
            caps=caps,
            receptor=rec_prepared.path,
            ligand=lig_prepared.path,
            center=center, size=size,
            out_path=Path(out_path),
            plan=plan,
            anchor=anchor.xyz if anchor else None,
            arm=arm, seed=seed, cpu=cpu,
            work_dir=work_dir, dry_run=dry_run, timeout=timeout,
            base_map_prefix=base_map_prefix,
        )
        out.run = run
        if run.error:
            out.error = run.error
            return out

        if dry_run:
            return out

        if reference_ligand and Path(reference_ligand).exists() and Path(out_path).exists():
            r = pose_rmsd(Path(reference_ligand), Path(out_path), 0, prefer=rmsd_method)
            out.rmsd = r
            out.success = (r.value is not None and r.value <= rmsd_cutoff)
            if r.error:
                out.warnings.append(f"RMSD: {r.error}")
    except Exception as exc:  # one complex must not abort a batch
        out.error = f"{type(exc).__name__}: {exc}"
    return out
