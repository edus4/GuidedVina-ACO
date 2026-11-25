"""
Adapter around a Vina-family docking binary.

The reference target is Vina-ACO (AutoDock Vina 1.2.7 with an ant-colony
global search). Two things about it drive the design here:

* **It has no bias option.** Nothing in its command line accepts a bias
  specification. Bias is therefore applied to the affinity maps: the engine
  writes them with ``--write_maps``, this module adds an attractive well, and
  the biased stage docks against the modified grids with ``--maps``. See
  guidedvina.maps.
* **``--exhaustiveness`` changes meaning under ACO.** With ``--search aco`` it
  is the number of independent colonies and defaults to 1, not the Monte Carlo
  exhaustiveness of 8. Ant counts live in ``--aco_ants``. Conflating the two
  would silently multiply or divide the search budget.

The binary's own ``--help`` is parsed once, and anything the plan needs but
the binary lacks raises rather than being dropped, so a run never quietly
differs from the run that was planned.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .bias import (
    Stage,
    apply_diversity_guardrail,
    build_schedule,
    pose_diversity_from_file,
)
from .maps import MapError, apply_bias_to_maps, find_maps
from .planner import SearchPlan
from .structures import read_models

_FLAG_RE = re.compile(r"--([A-Za-z0-9_\-]+)")

#: Capability -> candidate option spellings, in order of preference.
#: The first group is Vina 1.2.x; the aco_* group is Vina-ACO's addition.
_FLAG_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "receptor": ("receptor",),
    "ligand": ("ligand",),
    "out": ("out",),
    "seed": ("seed",),
    "cpu": ("cpu",),
    "exhaustiveness": ("exhaustiveness",),
    "max_evals": ("max_evals",),
    "num_modes": ("num_modes",),
    "local_only": ("local_only",),
    "score_only": ("score_only",),
    "maps": ("maps",),
    "write_maps": ("write_maps",),
    "force_even_voxels": ("force_even_voxels",),
    "batch": ("batch",),
    "dir": ("dir",),
    "search": ("search",),
    "aco_preset": ("aco_preset",),
    "aco_variant": ("aco_variant",),
    "aco_ants": ("aco_ants",),
    "aco_budget": ("aco_budget",),
    # Only present on forks that implement bias directly (e.g. AutoDock-Bias).
    "bias": ("bias", "bias_file", "biasfile"),
}

#: How bias can be delivered to an engine.
BIAS_BACKENDS = ("maps", "bias_file", "none")


class EngineError(RuntimeError):
    pass


class CapabilityError(EngineError):
    """The binary does not support something the plan requires."""


@dataclass
class EngineCapabilities:
    binary: str
    version: str
    flags: frozenset
    resolved: Dict[str, Optional[str]]

    def flag(self, capability: str) -> Optional[str]:
        return self.resolved.get(capability)

    def supports(self, capability: str) -> bool:
        return self.resolved.get(capability) is not None

    @property
    def bias_backend(self) -> str:
        """How this binary can be biased, best option first."""
        if self.supports("bias"):
            return "bias_file"
        if self.supports("maps") and self.supports("write_maps"):
            return "maps"
        return "none"

    @property
    def is_aco(self) -> bool:
        return self.supports("search") and self.supports("aco_preset")

    def to_dict(self) -> Dict[str, object]:
        return {
            "binary": self.binary,
            "version": self.version,
            "bias_backend": self.bias_backend,
            "aco_available": self.is_aco,
            "resolved_flags": {k: v for k, v in self.resolved.items() if v},
            "missing_flags": sorted(k for k, v in self.resolved.items() if not v),
        }


def probe_engine(binary: str, timeout: float = 60.0) -> EngineCapabilities:
    exe = shutil.which(binary) or binary
    if shutil.which(binary) is None and not Path(exe).exists():
        raise EngineError(
            f"Docking binary {binary!r} not found on PATH. Build Vina-ACO "
            "(./build.sh -> bin/vina_aco) and pass --engine-bin /path/to/bin/vina_aco."
        )
    help_text = ""
    for args in (["--help"], ["--help_advanced"], ["--version"]):
        try:
            proc = subprocess.run([exe, *args], capture_output=True, text=True, timeout=timeout)
            help_text += (proc.stdout or "") + (proc.stderr or "")
        except (subprocess.TimeoutExpired, OSError):
            continue
    if not help_text.strip():
        raise EngineError(
            f"{binary!r} produced no --help output, so its options cannot be verified."
        )
    flags = frozenset(_FLAG_RE.findall(help_text))
    resolved = {
        cap: next((c for c in cands if c in flags), None)
        for cap, cands in _FLAG_CANDIDATES.items()
    }
    version = ""
    m = re.search(r"(?:Vina-ACO|AutoDock Vina|version|v)\s*[: ]?\s*([0-9][0-9A-Za-z.\-_]*)",
                  help_text, re.I)
    if m:
        version = m.group(1)
    return EngineCapabilities(binary=exe, version=version, flags=flags, resolved=resolved)


@dataclass
class StageResult:
    stage: Stage
    command: List[str]
    out_path: str
    bias_artifact: Optional[str]          # bias file or biased-map prefix
    bias_detail: Optional[Dict[str, object]]
    returncode: int
    runtime_sec: float
    diversity_after: Optional[float]
    stdout_tail: str = ""

    @property
    def biased(self) -> bool:
        return self.bias_artifact is not None


@dataclass
class RunResult:
    arm: str
    plan: SearchPlan
    stages: List[StageResult] = field(default_factory=list)
    final_out: Optional[str] = None
    total_runtime_sec: float = 0.0
    engine: Optional[Dict[str, object]] = None
    anchor: Optional[Dict[str, object]] = None
    bias_backend: str = "none"
    map_prefix: Optional[str] = None
    error: Optional[str] = None
    dry_run: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "arm": self.arm,
            "plan": self.plan.to_dict(),
            "engine": self.engine,
            "anchor": self.anchor,
            "bias_backend": self.bias_backend,
            "map_prefix": self.map_prefix,
            "stages": [
                {
                    **s.stage.to_dict(),
                    "command": " ".join(s.command),
                    "bias_artifact": s.bias_artifact,
                    "bias_detail": s.bias_detail,
                    "out": s.out_path,
                    "returncode": s.returncode,
                    "runtime_sec": round(s.runtime_sec, 3),
                    "diversity_after": s.diversity_after,
                }
                for s in self.stages
            ],
            "final_out": self.final_out,
            "total_runtime_sec": round(self.total_runtime_sec, 3),
            "dry_run": self.dry_run,
            "error": self.error,
        }


def write_base_maps(
    caps: EngineCapabilities,
    receptor: Path,
    ligand: Path,
    center: Sequence[float],
    size: Sequence[float],
    prefix: Path,
    cpu: Optional[int] = None,
    timeout: Optional[float] = None,
    dry_run: bool = False,
) -> Tuple[List[str], Path]:
    """
    Generate the unbiased affinity maps for this receptor and box.

    Runs the engine in ``--score_only`` mode, which computes the maps for the
    ligand's atom types, writes them and exits without docking.
    ``--force_even_voxels`` is passed because the .map format needs an even
    voxel count per axis and the engine aborts otherwise.
    """
    if not caps.supports("write_maps"):
        raise CapabilityError(
            f"{caps.binary} has no --write_maps option, so map-based bias is impossible."
        )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        caps.binary,
        f"--{caps.flag('receptor')}", str(receptor),
        f"--{caps.flag('ligand')}", str(ligand),
        "--center_x", f"{center[0]:.3f}", "--center_y", f"{center[1]:.3f}",
        "--center_z", f"{center[2]:.3f}",
        "--size_x", f"{size[0]:.3f}", "--size_y", f"{size[1]:.3f}",
        "--size_z", f"{size[2]:.3f}",
        f"--{caps.flag('write_maps')}", str(prefix),
    ]
    if caps.supports("force_even_voxels"):
        cmd.append(f"--{caps.flag('force_even_voxels')}")
    if caps.supports("score_only"):
        cmd.append(f"--{caps.flag('score_only')}")
    if cpu and caps.supports("cpu"):
        cmd += [f"--{caps.flag('cpu')}", str(cpu)]

    if dry_run:
        return cmd, prefix
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise EngineError(
            f"writing affinity maps failed ({proc.returncode}): "
            f"{(proc.stderr or proc.stdout or '')[-400:]}"
        )
    if not find_maps(prefix):
        raise EngineError(
            f"the engine reported success but wrote no maps at {prefix}. "
            "Check the box size and that the ligand's atom types are supported."
        )
    return cmd, prefix


def _build_command(
    caps: EngineCapabilities,
    receptor: Optional[Path],
    ligand: Path,
    center: Sequence[float],
    size: Sequence[float],
    out_path: Path,
    stage: Stage,
    plan: SearchPlan,
    seed: int,
    bias_file: Optional[Path],
    biased_map_prefix: Optional[Path],
    cpu: Optional[int],
) -> List[str]:
    cmd = [caps.binary]

    # --receptor and --maps are mutually exclusive: biased stages dock against
    # the modified grids, which already carry the box.
    if biased_map_prefix is not None:
        if not caps.supports("maps"):
            raise CapabilityError(f"{caps.binary} has no --maps option")
        cmd += [f"--{caps.flag('maps')}", str(biased_map_prefix)]
    else:
        if receptor is None:
            raise EngineError("an unbiased stage needs a receptor")
        cmd += [f"--{caps.flag('receptor')}", str(receptor)]
        cmd += [
            "--center_x", f"{center[0]:.3f}", "--center_y", f"{center[1]:.3f}",
            "--center_z", f"{center[2]:.3f}",
            "--size_x", f"{size[0]:.3f}", "--size_y", f"{size[1]:.3f}",
            "--size_z", f"{size[2]:.3f}",
        ]
    cmd += [f"--{caps.flag('ligand')}", str(ligand)]

    if plan.search_method == "aco":
        if not caps.is_aco:
            raise CapabilityError(
                f"{caps.binary} has no --search/--aco_preset options, so the ACO "
                "search this plan asks for is unavailable. Re-run with "
                "--search-method mc, or point --engine-bin at a Vina-ACO build."
            )
        cmd += [f"--{caps.flag('search')}", "aco"]
        if plan.aco_preset:
            cmd += [f"--{caps.flag('aco_preset')}", plan.aco_preset]
    elif caps.supports("search"):
        cmd += [f"--{caps.flag('search')}", "mc"]

    # Under ACO this is the colony count; under MC it is Vina's exhaustiveness.
    if caps.supports("exhaustiveness"):
        cmd += [f"--{caps.flag('exhaustiveness')}", str(stage.exhaustiveness)]
    if plan.max_evals and caps.supports("max_evals"):
        cmd += [f"--{caps.flag('max_evals')}", str(plan.max_evals)]
    if caps.supports("num_modes"):
        cmd += [f"--{caps.flag('num_modes')}", str(plan.num_modes)]
    if caps.supports("seed"):
        cmd += [f"--{caps.flag('seed')}", str(seed)]
    else:
        raise CapabilityError(
            f"{caps.binary} exposes no --seed option. Runs could not be made "
            "reproducible and repeated seeds would be indistinguishable, so a "
            "multi-seed benchmark would be meaningless."
        )
    if cpu and caps.supports("cpu"):
        cmd += [f"--{caps.flag('cpu')}", str(cpu)]
    if stage.local_only:
        if not caps.supports("local_only"):
            raise CapabilityError(
                f"{caps.binary} has no --local_only option, which the final "
                "refinement stage requires. Re-run with --stages 1."
            )
        cmd.append(f"--{caps.flag('local_only')}")
    if bias_file is not None:
        if not caps.supports("bias"):
            raise CapabilityError(f"{caps.binary} does not accept a bias file")
        cmd += [f"--{caps.flag('bias')}", str(bias_file)]

    cmd += [f"--{caps.flag('out')}", str(out_path)]
    return cmd


def run_staged(
    caps: EngineCapabilities,
    receptor: Path,
    ligand: Path,
    center: Sequence[float],
    size: Sequence[float],
    out_path: Path,
    plan: SearchPlan,
    anchor: Optional[Sequence[float]],
    arm: str,
    seed: int = 42,
    cpu: Optional[int] = None,
    work_dir: Optional[Path] = None,
    dry_run: bool = False,
    timeout: Optional[float] = None,
    base_map_prefix: Optional[Path] = None,
) -> RunResult:
    """
    Execute the annealing schedule.

    Biased stages dock against maps carrying a well whose depth follows the
    schedule; the final stage is unbiased and docks against the receptor, so
    the reported pose and score come from the unmodified scoring function.
    """
    out_path = Path(out_path)
    work_dir = Path(work_dir) if work_dir else out_path.parent / "_guidedvina_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    backend = caps.bias_backend
    result = RunResult(arm=arm, plan=plan, engine=caps.to_dict(),
                       dry_run=dry_run, bias_backend=backend)
    if anchor is not None:
        result.anchor = {"xyz": [float(v) for v in anchor]}

    schedule = build_schedule(plan)
    needs_bias = any(s.bias_weight > 0 for s in schedule) and anchor is not None
    if needs_bias and backend == "none":
        raise CapabilityError(
            f"{caps.binary} supports neither a bias option nor --write_maps/--maps, "
            "so bias cannot be applied. Run with --no-bias to benchmark budget "
            "selection alone."
        )

    base_prefix: Optional[Path] = None
    if needs_bias and backend == "maps":
        base_prefix = Path(base_map_prefix) if base_map_prefix else work_dir / "basemaps"
        if dry_run or not find_maps(base_prefix):
            cmd, base_prefix = write_base_maps(
                caps, Path(receptor), Path(ligand), center, size,
                base_prefix, cpu=cpu, timeout=timeout, dry_run=dry_run,
            )
            if dry_run:
                result.stages.append(
                    StageResult(
                        stage=Stage(index=-1, bias_weight=0.0,
                                    exhaustiveness=0, local_only=False),
                        command=cmd, out_path=str(base_prefix),
                        bias_artifact=None,
                        bias_detail={"step": "write unbiased base maps"},
                        returncode=0, runtime_sec=0.0, diversity_after=None,
                    )
                )
        result.map_prefix = str(base_prefix)

    current_ligand = Path(ligand)
    measured_diversity: Optional[float] = None
    t_total = time.perf_counter()

    for stage in schedule:
        weight = stage.bias_weight
        if measured_diversity is not None and plan.diversity_floor_angstrom > 0:
            weight, applied = apply_diversity_guardrail(
                weight, measured_diversity, plan.diversity_floor_angstrom
            )
            stage.guardrail_applied = applied
        stage.bias_weight = weight
        stage.measured_diversity = measured_diversity

        bias_file: Optional[Path] = None
        biased_prefix: Optional[Path] = None
        bias_detail: Optional[Dict[str, object]] = None

        if weight > 0.0 and anchor is not None:
            depth = weight * 2.0   # kcal/mol at weight 1.0
            if backend == "maps":
                biased_prefix = work_dir / f"{out_path.stem}_stage{stage.index}_maps"
                if not dry_run:
                    try:
                        application = apply_bias_to_maps(
                            source_prefix=base_prefix,
                            output_prefix=biased_prefix,
                            anchor=anchor,
                            depth=depth,
                            radius=plan.anchor_radius,
                            target=plan.bias_target,
                        )
                        bias_detail = application.to_dict()
                    except MapError as exc:
                        result.error = f"stage {stage.index}: {exc}"
                        break
                else:
                    bias_detail = {
                        "step": "add well to maps",
                        "depth_kcal_per_mol": round(depth, 3),
                        "radius_angstrom": plan.anchor_radius,
                        "target": plan.bias_target,
                    }
            elif backend == "bias_file":
                from .bias import anchor_bias_points, write_bias_file
                points = anchor_bias_points(anchor, weight, plan.anchor_radius)
                bias_file = write_bias_file(
                    points, work_dir / f"{out_path.stem}_stage{stage.index}.bpf"
                )
                bias_detail = {"depth_kcal_per_mol": round(depth, 3)}

        stage_out = (
            out_path if stage.index == len(schedule) - 1
            else work_dir / f"{out_path.stem}_stage{stage.index}.pdbqt"
        )
        try:
            cmd = _build_command(
                caps,
                None if biased_prefix is not None else Path(receptor),
                current_ligand, center, size, stage_out, stage, plan, seed,
                bias_file, biased_prefix, cpu,
            )
        except EngineError as exc:
            result.error = str(exc)
            raise

        t0 = time.perf_counter()
        rc, tail = 0, ""
        if not dry_run:
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                rc, tail = proc.returncode, (proc.stdout or "")[-2000:]
                if rc != 0:
                    tail = ((proc.stderr or "") + tail)[-2000:]
            except subprocess.TimeoutExpired:
                result.error = f"stage {stage.index} timed out after {timeout}s"
                rc, tail = -9, ""
        elapsed = time.perf_counter() - t0

        diversity_after = (
            pose_diversity_from_file(stage_out)
            if (not dry_run and rc == 0 and stage_out.exists()) else None
        )
        result.stages.append(
            StageResult(
                stage=stage, command=cmd, out_path=str(stage_out),
                bias_artifact=str(biased_prefix or bias_file) if (biased_prefix or bias_file) else None,
                bias_detail=bias_detail,
                returncode=rc, runtime_sec=elapsed,
                diversity_after=diversity_after, stdout_tail=tail,
            )
        )
        if rc != 0:
            result.error = result.error or (
                f"stage {stage.index} exited with code {rc}: {tail.strip()[-300:]}"
            )
            break
        measured_diversity = diversity_after

        if stage.index < len(schedule) - 1 and not dry_run and stage_out.exists():
            try:
                seeded = work_dir / f"{out_path.stem}_seed{stage.index + 1}.pdbqt"
                _write_single_model(stage_out, seeded)
                read_models(seeded)
                current_ligand = seeded
            except Exception:
                current_ligand = Path(ligand)

    result.total_runtime_sec = time.perf_counter() - t_total
    result.final_out = str(out_path) if (dry_run or out_path.exists()) else None
    return result


def _write_single_model(src: Path, dest: Path) -> Path:
    """Copy the first MODEL of a multi-model PDBQT into a single-model file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    started = False
    with open(src, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("MODEL"):
                if started:
                    break
                started = True
                continue
            if line.startswith("ENDMDL"):
                break
            lines.append(line)
    with open(dest, "w", encoding="utf-8") as fh:
        fh.writelines(lines)
    return dest


def assumed_capabilities(binary: str = "vina_aco") -> EngineCapabilities:
    """
    Capabilities assumed for a dry run when no binary is installed: those of
    Vina-ACO, whose option set is known. A dry run therefore shows what would
    be issued against that engine, and cannot validate another build.
    """
    resolved = {cap: cands[0] for cap, cands in _FLAG_CANDIDATES.items()}
    resolved["bias"] = None          # Vina-ACO has no bias option
    return EngineCapabilities(
        binary=binary,
        version="assumed(dry-run)",
        flags=frozenset(v for v in resolved.values() if v),
        resolved=resolved,
    )
