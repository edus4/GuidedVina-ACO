"""
Adapter around a Vina-family docking binary.

Vina forks differ in which options they accept: the bias flag may be
``--bias``, ``--bias_file`` or absent altogether, and ACO builds may expose
an ant count instead of ``--exhaustiveness``. Rather than assume, the adapter
reads the binary's own ``--help`` once and refuses to run a biased stage on a
binary that cannot accept bias. A silent fall-back to unbiased docking would
produce a benchmark that looks like it tested the method when it did not.
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
    anchor_bias_points,
    apply_diversity_guardrail,
    build_schedule,
    pose_diversity_from_file,
    write_bias_file,
)
from .planner import SearchPlan
from .structures import read_models

_FLAG_RE = re.compile(r"--([A-Za-z0-9_\-]+)")

#: Candidate spellings for each capability, in order of preference.
_FLAG_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "bias": ("bias", "bias_file", "biasfile"),
    "exhaustiveness": ("exhaustiveness",),
    "ants": ("ants", "n_ants", "num_ants"),
    "max_evals": ("max_evals", "maxevals"),
    "num_modes": ("num_modes", "nmodes"),
    "local_only": ("local_only",),
    "score_only": ("score_only",),
    "seed": ("seed",),
    "cpu": ("cpu",),
    "out": ("out",),
    "receptor": ("receptor",),
    "ligand": ("ligand",),
}


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

    def to_dict(self) -> Dict[str, object]:
        return {
            "binary": self.binary,
            "version": self.version,
            "resolved_flags": self.resolved,
        }


def probe_engine(binary: str, timeout: float = 30.0) -> EngineCapabilities:
    exe = shutil.which(binary) or binary
    if not Path(exe).exists() and shutil.which(binary) is None:
        raise EngineError(
            f"Docking binary {binary!r} not found on PATH. Build or install it "
            "(e.g. CinarogluLab/Vina-ACO) and pass --engine-bin /path/to/binary."
        )
    help_text = ""
    for args in (["--help"], ["--help_advanced"], []):
        try:
            proc = subprocess.run(
                [exe, *args], capture_output=True, text=True, timeout=timeout
            )
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
    m = re.search(r"(?:version|v)\s*[: ]\s*([0-9][0-9A-Za-z.\-_]*)", help_text, re.I)
    if m:
        version = m.group(1)
    return EngineCapabilities(binary=exe, version=version, flags=flags, resolved=resolved)


@dataclass
class StageResult:
    stage: Stage
    command: List[str]
    out_path: str
    bias_file: Optional[str]
    returncode: int
    runtime_sec: float
    diversity_after: Optional[float]
    stdout_tail: str = ""


@dataclass
class RunResult:
    arm: str
    plan: SearchPlan
    stages: List[StageResult] = field(default_factory=list)
    final_out: Optional[str] = None
    total_runtime_sec: float = 0.0
    engine: Optional[Dict[str, object]] = None
    anchor: Optional[Dict[str, object]] = None
    error: Optional[str] = None
    dry_run: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "arm": self.arm,
            "plan": self.plan.to_dict(),
            "engine": self.engine,
            "anchor": self.anchor,
            "stages": [
                {
                    **s.stage.to_dict(),
                    "command": " ".join(s.command),
                    "bias_file": s.bias_file,
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


def _build_command(
    caps: EngineCapabilities,
    receptor: Path,
    ligand: Path,
    center: Sequence[float],
    size: Sequence[float],
    out_path: Path,
    stage: Stage,
    plan: SearchPlan,
    seed: int,
    bias_file: Optional[Path],
    cpu: Optional[int],
) -> List[str]:
    cmd = [caps.binary]
    cmd += [f"--{caps.flag('receptor') or 'receptor'}", str(receptor)]
    cmd += [f"--{caps.flag('ligand') or 'ligand'}", str(ligand)]
    cmd += [
        "--center_x", f"{center[0]:.3f}",
        "--center_y", f"{center[1]:.3f}",
        "--center_z", f"{center[2]:.3f}",
        "--size_x", f"{size[0]:.3f}",
        "--size_y", f"{size[1]:.3f}",
        "--size_z", f"{size[2]:.3f}",
    ]
    # Search breadth: prefer an explicit ant count on ACO builds, since
    # --exhaustiveness and an ant population are not the same parameter.
    if caps.supports("ants"):
        cmd += [f"--{caps.flag('ants')}", str(stage.exhaustiveness)]
    elif caps.supports("exhaustiveness"):
        cmd += [f"--{caps.flag('exhaustiveness')}", str(stage.exhaustiveness)]
    if plan.max_evals and caps.supports("max_evals"):
        cmd += [f"--{caps.flag('max_evals')}", str(plan.max_evals)]
    if caps.supports("num_modes"):
        cmd += [f"--{caps.flag('num_modes')}", str(plan.num_modes)]
    if caps.supports("seed"):
        cmd += [f"--{caps.flag('seed')}", str(seed)]
    if cpu and caps.supports("cpu"):
        cmd += [f"--{caps.flag('cpu')}", str(cpu)]
    if stage.local_only:
        if not caps.supports("local_only"):
            raise CapabilityError(
                f"{caps.binary} has no --local_only option, which the final "
                "refinement stage requires. Re-run with --stages 1 to disable "
                "staged annealing, or use a build that supports local refinement."
            )
        cmd += [f"--{caps.flag('local_only')}"]
    if bias_file is not None:
        if not caps.supports("bias"):
            raise CapabilityError(
                f"{caps.binary} does not accept a bias parameter file (looked for "
                f"{_FLAG_CANDIDATES['bias']}). Biased docking cannot be performed "
                "with this binary; run with --no-bias to benchmark budget "
                "selection alone, or point --engine-bin at a bias-capable build."
            )
        cmd += [f"--{caps.flag('bias')}", str(bias_file)]
    cmd += [f"--{caps.flag('out') or 'out'}", str(out_path)]
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
) -> RunResult:
    """
    Execute the annealing schedule.

    Each stage writes its own output file; the next stage starts from the best
    pose of the previous one. The bias weight of a stage is the nominal
    schedule value attenuated by the diversity actually measured in the
    previous stage's poses.
    """
    out_path = Path(out_path)
    work_dir = Path(work_dir) if work_dir else out_path.parent / "_guidedvina_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    result = RunResult(arm=arm, plan=plan, engine=caps.to_dict(), dry_run=dry_run)
    if anchor is not None:
        result.anchor = {"xyz": [float(v) for v in anchor]}

    schedule = build_schedule(plan)
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
        if weight > 0.0:
            if anchor is None:
                raise EngineError(
                    "A biased stage was scheduled but no anchor is available. "
                    "Supply --critical-residues, or run with --no-bias."
                )
            points = anchor_bias_points(anchor, weight, plan.anchor_radius)
            bias_file = write_bias_file(
                points, work_dir / f"{out_path.stem}_stage{stage.index}.bpf"
            )

        stage_out = (
            out_path
            if stage.index == len(schedule) - 1
            else work_dir / f"{out_path.stem}_stage{stage.index}.pdbqt"
        )
        cmd = _build_command(
            caps, receptor, current_ligand, center, size, stage_out,
            stage, plan, seed, bias_file, cpu,
        )

        t0 = time.perf_counter()
        rc, tail = 0, ""
        if not dry_run:
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=timeout
                )
                rc, tail = proc.returncode, (proc.stdout or "")[-2000:]
            except subprocess.TimeoutExpired:
                result.error = f"stage {stage.index} timed out after {timeout}s"
                rc, tail = -9, ""
        elapsed = time.perf_counter() - t0

        diversity_after = (
            pose_diversity_from_file(stage_out)
            if (not dry_run and rc == 0 and stage_out.exists())
            else None
        )
        result.stages.append(
            StageResult(
                stage=stage,
                command=cmd,
                out_path=str(stage_out),
                bias_file=str(bias_file) if bias_file else None,
                returncode=rc,
                runtime_sec=elapsed,
                diversity_after=diversity_after,
                stdout_tail=tail,
            )
        )
        if rc != 0:
            result.error = result.error or (
                f"stage {stage.index} exited with code {rc}: {tail.strip()[-300:]}"
            )
            break
        measured_diversity = diversity_after

        # Hand the best pose of this stage to the next one.
        if stage.index < len(schedule) - 1 and not dry_run and stage_out.exists():
            try:
                seeded = work_dir / f"{out_path.stem}_seed{stage.index + 1}.pdbqt"
                _write_single_model(stage_out, seeded)
                read_models(seeded)  # validate before handing it to the engine
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


def assumed_capabilities(binary: str = "vina-aco") -> EngineCapabilities:
    """
    Capabilities assumed for a dry run when no binary is installed.

    Used only with --dry-run, so that the planning and annealing logic can be
    inspected without a docking engine. The returned object claims support for
    every option, which is why a dry run can never validate that a real binary
    accepts the commands it prints.
    """
    resolved = {cap: cands[0] for cap, cands in _FLAG_CANDIDATES.items()}
    resolved["ants"] = None  # prefer --exhaustiveness in printed commands
    return EngineCapabilities(
        binary=binary,
        version="assumed(dry-run)",
        flags=frozenset(resolved.values()) - {None},
        resolved=resolved,
    )
