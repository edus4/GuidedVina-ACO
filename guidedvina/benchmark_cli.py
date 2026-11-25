"""Batch benchmark command line interface."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .benchmark import run_benchmark
from .engine import assumed_capabilities, probe_engine
from .planner import get_selector


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="guidedvina-benchmark",
        description="Paired baseline-vs-guided docking benchmark over a CSV manifest.",
    )
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("benchmark_results"))
    p.add_argument("--engine-bin", default="vina_aco")
    p.add_argument("--search-method", choices=["aco", "mc"], default=None)
    p.add_argument("--aco-preset", choices=["fast", "balanced", "accurate"], default=None)
    p.add_argument("--bias-target", choices=["map", "don", "acc", "apolar"], default="map")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3],
                   help="Seeds per arm (>=3 recommended; docking is stochastic)")
    p.add_argument("--arms", nargs="+", default=["baseline", "guided"],
                   choices=["baseline", "guided"])
    p.add_argument("--selector-model", type=Path, default=None)
    p.add_argument("--rmsd-cutoff", type=float, default=2.0)
    p.add_argument("--rmsd-method", choices=["auto", "graph", "assignment", "ordered"],
                   default="auto")
    p.add_argument("--success-rule", choices=["majority", "any"], default="majority")
    p.add_argument("--allow-leaky-anchor", action="store_true")
    p.add_argument("--allow-rb-estimate", action="store_true")
    p.add_argument("--no-bias", action="store_true")
    p.add_argument("--cpu", type=int, default=None)
    p.add_argument("--timeout", type=float, default=None)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    caps = (
        assumed_capabilities(args.engine_bin)
        if args.dry_run
        else probe_engine(args.engine_bin)
    )
    summary = run_benchmark(
        caps=caps,
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        selector=get_selector(args.selector_model),
        seeds=args.seeds,
        arms=args.arms,
        rmsd_cutoff=args.rmsd_cutoff,
        rmsd_method=args.rmsd_method,
        success_rule=args.success_rule,
        allow_leaky_anchor=args.allow_leaky_anchor,
        allow_rb_estimate=args.allow_rb_estimate,
        use_bias=not args.no_bias,
        bias_target=args.bias_target,
        search_method=args.search_method,
        aco_preset=args.aco_preset,
        cpu=args.cpu,
        dry_run=args.dry_run,
        timeout=args.timeout,
        progress=not args.quiet,
    )
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    for c in summary.get("caveats", []):
        print(f"CAVEAT: {c}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
