# GuidedVina-ACO

A control layer for Vina-family molecular docking engines. It does two things:

1. **Chooses a search budget** from ligand flexibility and pocket descriptors,
   instead of applying one preset to every ligand.
2. **Applies knowledge-based bias that is annealed away.** Docking starts with
   an attractive potential at a pocket sub-region you nominate, and the bias is
   reduced stage by stage until the final stage runs unbiased — so the reported
   pose and score are not artefacts of the bias.

The point of the second mechanism is to recover poses that a global search
misses on flexible ligands, without locking the search into a wrong local
minimum.

## Implementation status

Being explicit about this, because a wrapper that reports a bias weight it
never applied is worse than no wrapper at all.

| Component | Status |
|---|---|
| Ligand/pocket descriptors from `.pdbqt`/`.pdb` | implemented |
| Chain-aware anchor from nominated residues | implemented |
| Bias parameter file (`.bpf`) emission, passed to the engine | implemented |
| Staged annealing with the final stage unbiased | implemented |
| Pose-diversity guardrail measured from engine output | implemented |
| Paired baseline-vs-guided benchmark over multiple seeds | implemented |
| Symmetry-corrected RMSD | implemented (RDKit `GetBestRMS`; scipy assignment fallback) |
| Budget selector trained on sweep data | implemented (`scripts/sweep.py` → `scripts/train_selector.py`); **default budgets are documented priors, not fitted values, until you train one** |
| PoseBusters physical-validity checks | not implemented |
| Virtual-screening enrichment metrics (EF, AUC) | not implemented; this repo evaluates pose prediction only |

**There are no benchmark results in this repository yet.** The example files
are synthetic coordinates for exercising the code paths; they cannot support
any accuracy or speedup claim.

## Requirements

- Python ≥ 3.9, numpy
- scipy (recommended): enables assignment-based symmetry handling in RMSD
- RDKit (optional): graph-symmetry RMSD and exact rotatable-bond counts for
  non-PDBQT ligands
- A **bias-capable Vina-family binary** for real docking, e.g.
  [CinarogluLab/Vina-ACO](https://github.com/CinarogluLab/Vina-ACO) or an
  AutoDock-Bias build. The wrapper reads your binary's `--help` and refuses to
  run a biased stage if it finds no bias option, rather than silently docking
  without bias.
- Open Babel (`obabel`) or Meeko, only if you pass `.pdb` instead of `.pdbqt`

```bash
git clone https://github.com/edus4/GuidedVina-ACO.git
cd GuidedVina-ACO
pip install -r requirements.txt     # or: pip install -e ".[dev]"
```

## Try it without a docking binary

`--dry-run` prints the descriptors, the chosen budget, the annealing schedule
and the exact commands that would be issued. No engine required.

```bash
python3 -m guidedvina.cli \
  --receptor examples/sample_receptor.pdbqt \
  --ligand   examples/sample_ligand.pdbqt \
  --center 0 0 0 --size 20 20 20 \
  --critical-residues A:12 A:60 A:61 \
  --dry-run
```

A dry run assumes the engine supports every option, so it validates the
planning logic only — never that your binary accepts those commands.

## Real docking

```bash
python3 -m guidedvina.cli \
  --engine-bin /path/to/vina-aco \
  --receptor receptor.pdbqt \
  --ligand   ligand.pdbqt \
  --center 12.5 -4.2 18.0 --size 20 20 20 \
  --critical-residues A:12 A:60 A:61 \
  --reference-ligand crystal_ligand.pdbqt \
  --out docked_guided.pdbqt
```

`--critical-residues` accepts `12`, `A:12`, `A:12B` (insertion code) and
`A:12-18`. **Qualify the chain.** A bare `12` matches residue 12 in every
chain, and the anchor then becomes the centroid across copies — a point that
lies in no pocket at all. The tool warns when this happens; the warning means
your anchor is wrong, not that it is noted.

The numbering is whatever is in your receptor file, which is not always the
UniProt numbering of the paper you took the residues from.

## Benchmarking

The benchmark runs **both arms on every complex with the same seeds**:

- `baseline` — engine defaults, no bias
- `guided` — selected budget plus the staged bias schedule

```bash
python3 -m guidedvina.benchmark_cli \
  --manifest my_manifest.csv \
  --engine-bin /path/to/vina-aco \
  --out-dir benchmark_results \
  --seeds 1 2 3 4 5
```

Docking is stochastic, so a single seed per arm cannot support a comparison;
three is the minimum and the summary says so when you use fewer.

Outputs in `--out-dir`:

- `runs.csv` — one row per (complex, arm, seed), including which RMSD method
  was used, whether bias was actually applied, the measured pose diversity,
  and whether the diversity guardrail fired
- `summary.json` — paired comparison: success rate per arm, the difference
  with a bootstrap 95% CI, discordant-pair counts, an exact McNemar p-value,
  the median runtime ratio, and a `caveats` list
- `poses/`, `work/` — docked poses, bias files and per-stage outputs

The unit of analysis is the complex, not the run. A complex counts as a
success under the default `--success-rule majority` when more than half its
seeds land within the cutoff.

### Manifest format

```csv
complex_id,receptor,ligand_input,ligand_ref,center_x,center_y,center_z,size_x,size_y,size_z,critical_residues
1abc,1abc_rec.pdbqt,1abc_lig.pdbqt,1abc_ref.pdbqt,12.5,-4.2,18.0,20,20,20,A:12;A:60;A:61
```

- `complex_id`, `receptor`, `ligand_input`, `center_*` are required;
  `size_*` default to 20 Å; relative paths resolve against the manifest's own
  directory
- `ligand_ref` is the crystallographic ligand; without it no RMSD and no
  success can be computed for that complex
- `critical_residues` is `;`-separated and accepts the same forms as the CLI

## Training the budget selector

Until you do this, the budgets come from `HeuristicSelector`, which is a
documented piecewise prior over flexibility bands. To fit a real one:

```bash
# 1. Run the whole configuration grid on every complex
python3 scripts/sweep.py --manifest my_manifest.csv \
  --engine-bin /path/to/vina-aco --out-dir sweep_results --seeds 1 2 3

# 2. Fit per-configuration success and runtime models
python3 scripts/train_selector.py --sweep sweep_results/sweep.csv \
  --out models/selector.json --folds 5

# 3. Use it
python3 -m guidedvina.benchmark_cli --manifest my_manifest.csv \
  --engine-bin /path/to/vina-aco --selector-model models/selector.json \
  --out-dir benchmark_results
```

The selector picks the cheapest configuration whose predicted success is
within 2% of the best predicted. `train_selector.py` reports cross-validation
grouped by complex, alongside the oracle and cheapest-configuration rates, so
you can see whether the model beats "always use the cheap setting".

Fitting and evaluating the selector on the same complexes would be
meaningless; use the grouped CV numbers, and keep a test set of complexes the
sweep never saw.

## Avoiding a leaky benchmark

Taking the bias anchor from the reference ligand tells the search where the
answer is. The tool refuses to do it unless you pass `--allow-leaky-anchor`,
and any run that used it is flagged in `summary.json`'s caveats. Use it for
sanity checks, never for a reported number.

## Files

```
guidedvina/
  structures.py   fixed-column PDB/PDBQT parsing, multi-model aware
  features.py     ligand and pocket descriptors
  anchors.py      residue selectors, chain-aware anchor, leakage guard
  planner.py      HeuristicSelector (priors) and TrainedSelector (fitted)
  bias.py         .bpf emission, annealing schedule, measured pose diversity
  engine.py       binary capability probing, staged execution
  prepare.py      .pdb -> .pdbqt conversion via Open Babel or Meeko
  rmsd.py         symmetry-corrected RMSD with explicit method labelling
  stats.py        paired bootstrap CI and exact McNemar test
  runner.py       one complex, one arm, one seed
  benchmark.py    manifest parsing and the paired batch benchmark
  grid.py         configuration grid for selector training
scripts/
  make_examples.py    regenerate the synthetic examples
  sweep.py            run the grid to collect training data
  train_selector.py   fit and cross-validate the selector
tests/              pytest suite, incl. a stub engine that fakes a real binary
docs/METHOD.md      what each mechanism does and why
```

## Known limitations

- Annealing is implemented as a sequence of docking runs, because Vina-family
  engines expose no hook to vary bias within one run. Each stage is a fresh
  search seeded from the previous stage's best pose.
- The bias is a single spherical well at one anchor. Multi-point and
  pharmacophore-typed bias (`don`/`acc`/`aro`) are supported by the file
  format and by `bias.BiasPoint`, but nothing yet constructs them.
- Converting `.pdb` to `.pdbqt` changes atom ordering. Prepare the reference
  and docked ligands with the same tool, or rely on the symmetry-corrected
  RMSD methods rather than `ordered`.
- `buriedness` is a crude occupancy estimate (heavy atoms × 20 Å³), not a
  solvent-accessibility calculation.
- No PoseBusters integration, so physical validity of poses is unchecked.

## Tests

```bash
python3 -m pytest -q
```

The suite runs the whole pipeline against `tests/stub_engine.py`, a fake
binary that advertises a real Vina-style `--help` and writes multi-model
output. Among other things it asserts that `--bias` reaches the command line,
that the baseline arm sends none, and that a binary without a bias option
causes a refusal rather than a silent unbiased run.

## Citing

See `CITATION.cff`. If you use the Vina-ACO engine itself, cite that project
separately; this repository is a wrapper and does not implement a docking
algorithm.

## License

MIT — see `LICENSE`.
