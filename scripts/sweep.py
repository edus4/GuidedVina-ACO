#!/usr/bin/env python3
"""
Run every configuration in the grid on every complex of a manifest.

The output is the training set for scripts/train_selector.py: one row per
(complex, configuration, seed) with the measured outcome. This is the step
that turns the hand-written priors in planner.HeuristicSelector into a fitted
model, so run it before claiming the selector is learned.

Usage:
  python3 scripts/sweep.py --manifest examples/sample_manifest.csv \
      --engine-bin /path/to/vina-aco --out-dir sweep_results --seeds 1 2 3
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from guidedvina.anchors import parse_residue_specs            # noqa: E402
from guidedvina.benchmark import read_manifest                 # noqa: E402
from guidedvina.engine import assumed_capabilities, probe_engine  # noqa: E402
from guidedvina.grid import DEFAULT_GRID, FixedPlanSelector, grid_plans  # noqa: E402
from guidedvina.runner import dock_once                        # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--engine-bin", default="vina_aco")
    p.add_argument("--out-dir", type=Path, default=Path("sweep_results"))
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--rmsd-cutoff", type=float, default=2.0)
    p.add_argument("--rmsd-method", default="auto")
    p.add_argument("--allow-rb-estimate", action="store_true")
    p.add_argument("--timeout", type=float, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    caps = (
        assumed_capabilities(args.engine_bin) if args.dry_run
        else probe_engine(args.engine_bin)
    )
    rows = read_manifest(args.manifest)
    plans = grid_plans()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    total = len(rows) * len(plans) * len(args.seeds)
    done = 0
    for row in rows:
        specs = parse_residue_specs(row.critical_residues) if any(row.critical_residues) else []
        for plan in plans:
            for seed in args.seeds:
                out_path = (
                    args.out_dir / "poses" /
                    f"{row.complex_id}_{plan.regime}_s{seed}.pdbqt"
                )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                outcome = dock_once(
                    caps=caps,
                    complex_id=row.complex_id,
                    receptor=row.receptor,
                    ligand=row.ligand_input,
                    center=row.center, size=row.size,
                    out_path=out_path,
                    arm="guided" if plan.initial_bias_weight > 0 else "baseline",
                    selector=FixedPlanSelector(plan),
                    seed=seed,
                    residue_specs=specs,
                    reference_ligand=row.ligand_ref,
                    allow_rb_estimate=args.allow_rb_estimate,
                    rmsd_cutoff=args.rmsd_cutoff,
                    rmsd_method=args.rmsd_method,
                    use_bias=plan.initial_bias_weight > 0,
                    dry_run=args.dry_run,
                    timeout=args.timeout,
                    work_dir=args.out_dir / "work" / row.complex_id,
                    base_map_prefix=args.out_dir / "work" / row.complex_id / "basemaps",
                )
                rec = outcome.to_record(rmsd_cutoff=args.rmsd_cutoff)
                rec["config_id"] = plan.regime
                rec["cfg_exhaustiveness"] = plan.exhaustiveness
                rec["cfg_search_method"] = plan.search_method
                rec["cfg_aco_preset"] = plan.aco_preset or ""
                rec["cfg_stages"] = plan.stages
                rec["cfg_initial_bias_weight"] = plan.initial_bias_weight
                rec["cfg_annealing_tau"] = plan.annealing_tau
                rec["cfg_anchor_radius"] = plan.anchor_radius
                rec["pocket_box_volume"] = outcome.pocket.box_volume if outcome.pocket else ""
                rec["pocket_heavy_in_box"] = outcome.pocket.heavy_atoms_in_box if outcome.pocket else ""
                rec["pocket_polar_in_box"] = outcome.pocket.polar_atoms_in_box if outcome.pocket else ""
                rec["ligand_max_radius"] = outcome.ligand.max_radius if outcome.ligand else ""
                records.append(rec)
                done += 1
                print(f"[{done}/{total}] {row.complex_id} {plan.regime} seed={seed} "
                      f"rmsd={rec['top1_rmsd']}", flush=True)

    out_csv = args.out_dir / "sweep.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({"rows": len(records), "csv": str(out_csv),
                      "configurations": len(plans)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
