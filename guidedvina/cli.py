"""Single-complex command line interface."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .anchors import parse_residue_specs
from .engine import assumed_capabilities, probe_engine
from .planner import get_selector
from .runner import dock_once


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="guidedvina",
        description=(
            "Flexibility-aware search budget and staged biased docking on top of a "
            "Vina-family engine."
        ),
    )
    p.add_argument("--receptor", type=Path, required=True, help="Receptor .pdbqt or .pdb")
    p.add_argument("--ligand", type=Path, required=True, help="Ligand .pdbqt/.pdb/.sdf/.mol2")
    p.add_argument("--center", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    p.add_argument("--size", type=float, nargs=3, default=[20.0, 20.0, 20.0],
                   metavar=("SX", "SY", "SZ"))
    p.add_argument("--out", type=Path, default=Path("docked_guided.pdbqt"))

    p.add_argument("--engine-bin", default="vina_aco",
                   help="Docking binary (default: vina_aco)")
    p.add_argument("--search-method", choices=["aco", "mc"], default=None,
                   help="aco = ant colony (Vina-ACO), mc = upstream Vina Monte Carlo")
    p.add_argument("--aco-preset", choices=["fast", "balanced", "accurate"], default=None,
                   help="Override the ACO parameter set chosen by the selector")
    p.add_argument("--bias-target", choices=["map", "don", "acc", "apolar"], default="map",
                   help="Which atom-type maps the bias well is added to")
    p.add_argument("--critical-residues", nargs="*", default=None,
                   metavar="SPEC",
                   help="Bias anchor residues: 12, A:12, A:12-18 (chain-qualified is safer)")
    p.add_argument("--reference-ligand", type=Path, default=None,
                   help="Crystallographic ligand, for RMSD evaluation")
    p.add_argument("--allow-leaky-anchor", action="store_true",
                   help="Permit deriving the anchor from the reference ligand (leaky)")
    p.add_argument("--no-bias", action="store_true",
                   help="Select a budget but apply no bias (budget-only mode)")
    p.add_argument("--stages", type=int, default=None,
                   help="Override the number of annealing stages")
    p.add_argument("--selector-model", type=Path, default=None,
                   help="Trained selector JSON from scripts/train_selector.py")
    p.add_argument("--rotatable-bonds", type=int, default=None,
                   help="Supply the rotatable-bond count explicitly")
    p.add_argument("--allow-rb-estimate", action="store_true",
                   help="Permit estimating rotatable bonds from heavy-atom count")
    p.add_argument("--rmsd-method", choices=["auto", "graph", "assignment", "ordered"],
                   default="auto")
    p.add_argument("--rmsd-cutoff", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", type=int, default=None)
    p.add_argument("--timeout", type=float, default=None, help="Per-stage timeout, seconds")
    p.add_argument("--arm", choices=["guided", "baseline"], default="guided")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan, schedule and commands without docking")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    caps = (
        assumed_capabilities(args.engine_bin)
        if args.dry_run
        else probe_engine(args.engine_bin)
    )
    specs = parse_residue_specs(args.critical_residues) if args.critical_residues else []
    outcome = dock_once(
        caps=caps,
        complex_id=args.out.stem,
        receptor=args.receptor,
        ligand=args.ligand,
        center=tuple(args.center),
        size=tuple(args.size),
        out_path=args.out,
        arm=args.arm,
        selector=get_selector(args.selector_model),
        seed=args.seed,
        residue_specs=specs,
        reference_ligand=args.reference_ligand,
        allow_leaky_anchor=args.allow_leaky_anchor,
        rotatable_bonds=args.rotatable_bonds,
        allow_rb_estimate=args.allow_rb_estimate,
        rmsd_cutoff=args.rmsd_cutoff,
        rmsd_method=args.rmsd_method,
        use_bias=not args.no_bias,
        bias_target=args.bias_target,
        stages_override=args.stages,
        search_method=args.search_method,
        aco_preset=args.aco_preset,
        cpu=args.cpu,
        dry_run=args.dry_run,
        timeout=args.timeout,
    )
    report = {
        "complex_id": outcome.complex_id,
        "arm": outcome.arm,
        "seed": outcome.seed,
        "ligand_features": outcome.ligand.to_dict() if outcome.ligand else None,
        "pocket_features": outcome.pocket.to_dict() if outcome.pocket else None,
        "anchor": (
            {
                "xyz": outcome.anchor.xyz,
                "source": outcome.anchor.source,
                "residues_matched": outcome.anchor.residues_matched,
                "chains_matched": outcome.anchor.chains_matched,
                "atom_count": outcome.anchor.atom_count,
            }
            if outcome.anchor else None
        ),
        "plan": outcome.plan.to_dict() if outcome.plan else None,
        "run": outcome.run.to_dict() if outcome.run else None,
        "rmsd": outcome.rmsd.to_dict() if outcome.rmsd else None,
        "success": outcome.success,
        "warnings": outcome.warnings,
        "error": outcome.error,
    }
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    if outcome.error:
        print(f"\nERROR: {outcome.error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
