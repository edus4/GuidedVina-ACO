#!/usr/bin/env python3
"""
Fit the configuration selector from a sweep.

For each configuration in the sweep, two models are fitted over the ligand and
pocket descriptors:

  * P(top-1 RMSD <= cutoff)  -- logistic regression (IRLS with ridge damping)
  * expected runtime         -- ridge-regularised least squares

At prediction time the selector picks the cheapest configuration whose
predicted success is within a tolerance of the best. The export is plain JSON
so that the trained model is inspectable and has no runtime dependency beyond
numpy.

Grouped cross-validation by complex is reported, because a model validated on
the same complexes it was fitted on tells you nothing about a new target.

Usage:
  python3 scripts/train_selector.py --sweep sweep_results/sweep.csv \
      --out models/selector.json --folds 5
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from guidedvina.planner import FEATURE_ORDER  # noqa: E402

_COLUMN_FOR_FEATURE = {
    "rotatable_bonds": "rotatable_bonds",
    "heavy_atoms": "heavy_atoms",
    "max_radius": "ligand_max_radius",
    "box_volume": "pocket_box_volume",
    "heavy_atoms_in_box": "pocket_heavy_in_box",
    "polar_atoms_in_box": "pocket_polar_in_box",
    "buriedness": "buriedness",
}


def load_sweep(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    missing = set(_COLUMN_FOR_FEATURE.values()) - set(rows[0].keys()) if rows else set()
    if missing:
        raise SystemExit(
            f"{path} is missing columns {sorted(missing)}. Generate it with scripts/sweep.py."
        )
    usable = []
    for r in rows:
        if r.get("error"):
            continue
        if r.get("success") in ("", None):
            continue
        try:
            [float(r[_COLUMN_FOR_FEATURE[f]]) for f in FEATURE_ORDER]
            float(r["runtime_sec"])
        except (KeyError, ValueError):
            continue
        usable.append(r)
    if not usable:
        raise SystemExit(f"{path}: no usable rows (all runs errored or lacked RMSD)")
    return usable


def design_matrix(rows: Sequence[dict]) -> np.ndarray:
    return np.array(
        [[float(r[_COLUMN_FOR_FEATURE[f]]) for f in FEATURE_ORDER] for r in rows],
        dtype=float,
    )


def fit_logistic(X: np.ndarray, y: np.ndarray, ridge: float = 1.0,
                 iterations: int = 100) -> Tuple[np.ndarray, float]:
    """IRLS with ridge damping; returns (coefficients, intercept)."""
    n, d = X.shape
    Xa = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    penalty = ridge * np.eye(d + 1)
    penalty[-1, -1] = 0.0  # do not penalise the intercept
    for _ in range(iterations):
        z = Xa @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        s = np.clip(p * (1 - p), 1e-6, None)
        grad = Xa.T @ (y - p) - penalty @ w
        hess = -(Xa.T * s) @ Xa - penalty
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        w_new = w - step
        if not np.all(np.isfinite(w_new)) or np.max(np.abs(w_new - w)) < 1e-8:
            w = w_new if np.all(np.isfinite(w_new)) else w
            break
        w = w_new
    return w[:-1], float(w[-1])


def fit_ridge(X: np.ndarray, y: np.ndarray, ridge: float = 1.0) -> Tuple[np.ndarray, float]:
    n, d = X.shape
    Xa = np.hstack([X, np.ones((n, 1))])
    penalty = ridge * np.eye(d + 1)
    penalty[-1, -1] = 0.0
    w = np.linalg.solve(Xa.T @ Xa + penalty, Xa.T @ y)
    return w[:-1], float(w[-1])


def standardise(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-9] = 1.0
    return (X - mean) / std, mean, std


def unstandardise_linear(coefs: np.ndarray, intercept: float,
                         mean: np.ndarray, std: np.ndarray) -> Tuple[np.ndarray, float]:
    """Convert coefficients fitted on standardised features back to raw units."""
    raw = coefs / std
    raw_intercept = intercept - float(np.dot(raw, mean))
    return raw, raw_intercept


def grouped_folds(groups: Sequence[str], n_folds: int, seed: int = 0) -> List[np.ndarray]:
    unique = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    assign = {g: i % n_folds for i, g in enumerate(unique)}
    idx = np.array([assign[g] for g in groups])
    return [np.where(idx == k)[0] for k in range(min(n_folds, len(unique)))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", type=Path, required=True, help="sweep.csv from scripts/sweep.py")
    ap.add_argument("--out", type=Path, default=Path("models/selector.json"))
    ap.add_argument("--name", default="selector")
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-rows-per-config", type=int, default=8)
    args = ap.parse_args()

    rows = load_sweep(args.sweep)
    by_config: Dict[str, List[dict]] = {}
    for r in rows:
        by_config.setdefault(r.get("config_id") or r.get("regime", "unknown"), []).append(r)

    configurations = []
    skipped = []
    for cfg_id, cfg_rows in sorted(by_config.items()):
        if len(cfg_rows) < args.min_rows_per_config:
            skipped.append({"config_id": cfg_id, "rows": len(cfg_rows),
                            "reason": "too few runs to fit"})
            continue
        X = design_matrix(cfg_rows)
        y = np.array([1.0 if str(r["success"]).lower() == "true" else 0.0 for r in cfg_rows])
        t = np.array([float(r["runtime_sec"]) for r in cfg_rows])
        Xs, mean, std = standardise(X)

        if len(set(y.tolist())) == 1:
            # No variation to fit: fall back to the observed rate as an intercept.
            rate = float(np.clip(y.mean(), 1e-3, 1 - 1e-3))
            s_coefs = np.zeros(X.shape[1])
            s_intercept = float(np.log(rate / (1 - rate)))
        else:
            cs, ic = fit_logistic(Xs, y, ridge=args.ridge)
            s_coefs, s_intercept = unstandardise_linear(cs, ic, mean, std)

        rc, ri = fit_ridge(Xs, t, ridge=args.ridge)
        r_coefs, r_intercept = unstandardise_linear(rc, ri, mean, std)

        first = cfg_rows[0]
        configurations.append({
            "config_id": cfg_id,
            "regime": cfg_id,
            "exhaustiveness": int(float(first.get("cfg_exhaustiveness", first.get("exhaustiveness", 1)))),
            "search_method": first.get("cfg_search_method", "aco") or "aco",
            "aco_preset": first.get("cfg_aco_preset") or "balanced",
            "stages": int(float(first.get("cfg_stages", 1))),
            "initial_bias_weight": float(first.get("cfg_initial_bias_weight", 0.0)),
            "annealing_tau": float(first.get("cfg_annealing_tau", 0.45)),
            "anchor_radius": float(first.get("cfg_anchor_radius", 5.0)),
            "diversity_floor_angstrom": 1.2 if float(first.get("cfg_initial_bias_weight", 0.0)) > 0 else 0.0,
            "num_modes": 9,
            "max_evals": None,
            "n_training_runs": len(cfg_rows),
            "observed_success_rate": round(float(y.mean()), 4),
            "observed_median_runtime": round(float(np.median(t)), 3),
            "success_coefs": [round(float(v), 6) for v in s_coefs],
            "success_intercept": round(float(s_intercept), 6),
            "runtime_coefs": [round(float(v), 6) for v in r_coefs],
            "runtime_intercept": round(float(r_intercept), 6),
        })

    if not configurations:
        raise SystemExit(
            "No configuration had enough runs to fit. Collect more sweep data "
            f"(min {args.min_rows_per_config} usable runs per configuration)."
        )

    # Grouped cross-validation of the selection rule itself.
    cv = _cross_validate(rows, args.folds, args.ridge, args.min_rows_per_config)

    model = {
        "name": args.name,
        "feature_order": list(FEATURE_ORDER),
        "configurations": configurations,
        "training": {
            "sweep_csv": str(args.sweep),
            "n_usable_runs": len(rows),
            "n_complexes": len({r["complex_id"] for r in rows}),
            "skipped_configurations": skipped,
        },
        "cross_validation": cv,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(model, fh, indent=2)
    print(json.dumps({"written": str(args.out),
                      "configurations": len(configurations),
                      "cross_validation": cv}, indent=2))
    return 0


def _cross_validate(rows: List[dict], n_folds: int, ridge: float,
                    min_rows: int) -> Dict[str, object]:
    """
    Compare, per held-out complex, the selected configuration against the
    best and the cheapest configuration available for that complex.
    """
    complexes = sorted({r["complex_id"] for r in rows})
    if len(complexes) < 3:
        return {"note": f"only {len(complexes)} complexes: cross-validation skipped"}

    groups = [r["complex_id"] for r in rows]
    folds = grouped_folds(groups, n_folds)
    selected_success, oracle_success, default_success = [], [], []

    for fold in folds:
        test_ids = {groups[i] for i in fold}
        train = [r for r in rows if r["complex_id"] not in test_ids]
        test = [r for r in rows if r["complex_id"] in test_ids]
        if not train or not test:
            continue
        fitted = {}
        by_cfg: Dict[str, List[dict]] = {}
        for r in train:
            by_cfg.setdefault(r.get("config_id", "unknown"), []).append(r)
        for cfg_id, cfg_rows in by_cfg.items():
            if len(cfg_rows) < min_rows:
                continue
            X = design_matrix(cfg_rows)
            y = np.array([1.0 if str(r["success"]).lower() == "true" else 0.0 for r in cfg_rows])
            t = np.array([float(r["runtime_sec"]) for r in cfg_rows])
            Xs, mean, std = standardise(X)
            if len(set(y.tolist())) == 1:
                rate = float(np.clip(y.mean(), 1e-3, 1 - 1e-3))
                sc, si = np.zeros(X.shape[1]), float(np.log(rate / (1 - rate)))
            else:
                cs, ic = fit_logistic(Xs, y, ridge=ridge)
                sc, si = unstandardise_linear(cs, ic, mean, std)
            rc, ri = fit_ridge(Xs, t, ridge=ridge)
            rcoef, rint = unstandardise_linear(rc, ri, mean, std)
            fitted[cfg_id] = (sc, si, rcoef, rint)
        if not fitted:
            continue

        per_complex: Dict[str, Dict[str, List[dict]]] = {}
        for r in test:
            per_complex.setdefault(r["complex_id"], {}).setdefault(
                r.get("config_id", "unknown"), []
            ).append(r)

        for cid, cfgs in per_complex.items():
            feats = design_matrix([next(iter(cfgs.values()))[0]])[0]
            scores = []
            for cfg_id, (sc, si, rcoef, rint) in fitted.items():
                if cfg_id not in cfgs:
                    continue
                z = float(np.dot(sc, feats) + si)
                p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
                cost = float(np.dot(rcoef, feats) + rint)
                scores.append((p, cost, cfg_id))
            if not scores:
                continue
            best_p = max(s[0] for s in scores)
            affordable = sorted([s for s in scores if s[0] >= best_p - 0.02], key=lambda s: s[1])
            chosen = affordable[0][2]

            def rate(cfg_id: str) -> float:
                runs = cfgs[cfg_id]
                return float(np.mean([1.0 if str(r["success"]).lower() == "true" else 0.0
                                      for r in runs]))

            selected_success.append(rate(chosen))
            oracle_success.append(max(rate(c) for c in cfgs))
            cheapest = min(cfgs, key=lambda c: float(np.median(
                [float(r["runtime_sec"]) for r in cfgs[c]])))
            default_success.append(rate(cheapest))

    if not selected_success:
        return {"note": "cross-validation produced no comparable folds"}
    return {
        "n_held_out_complexes": len(selected_success),
        "selected_success_rate": round(float(np.mean(selected_success)), 4),
        "oracle_success_rate": round(float(np.mean(oracle_success)), 4),
        "cheapest_config_success_rate": round(float(np.mean(default_success)), 4),
        "note": "grouped by complex; selected vs best-possible vs cheapest configuration",
    }


if __name__ == "__main__":
    raise SystemExit(main())
