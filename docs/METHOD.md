# Method

## 1. Descriptors

From the ligand: heavy-atom count, rotatable-bond count, and the radius of the
furthest heavy atom from the centroid (a size proxy that does not depend on
bond perception).

The rotatable-bond count is the input the budget depends on most, so it is
never inferred from a correlation. A PDBQT states it (`REMARK N active
torsions`, or the number of `BRANCH` records); for other formats RDKit is
asked. If neither is available the run stops and asks for
`--rotatable-bonds N`. Passing `--allow-rb-estimate` substitutes
`0.22 × heavy atoms` and marks every downstream record as `estimated`, so the
plan can be discounted when reading results.

From the pocket: box volume, heavy and polar receptor atoms inside the box,
and a crude occupancy ratio.

## 2. Anchor

The bias needs a point. It comes from residues you nominate — the centroid of
their side-chain heavy atoms, restricted to atoms inside the docking box.

Chain IDs matter. In a homodimer, residue 12 exists in both chains; averaging
both copies puts the anchor between the two pockets, in neither. Selectors are
therefore matched on (chain, residue number, insertion code), and a selector
that matches several chains raises a warning naming them.

Two failure modes stop the run rather than degrading quietly:

- no nominated residue exists in the receptor → error naming the numbering as
  the likely cause;
- the residues exist but no atom of theirs is inside the box → error, since
  the box and the nominated site disagree.

There is no "infer the pocket from polar atoms" fallback. In a 20 Å box the
centroid of all N/O/S atoms sits within a fraction of an Ångström of the box
centre for most receptors, so such a fallback adds nothing while looking
knowledge-based. With no residues given, the guided arm runs without bias and
says so.

## 3. Bias

A bias parameter file is written and handed to the engine:

```
x y z Vset r type
  12.500   -4.200   18.000    -1.30   5.00 map
```

`Vset` is an energy offset in kcal/mol applied within radius `r`; negative
values favour the region. At bias weight `w`, `Vset = -w × 2.0` kcal/mol.
`type` is `map` for a type-agnostic well; `don`, `acc` and `aro` restrict the
bias to donors, acceptors or aromatic atoms.

The engine's own `--help` is parsed once to find which spelling of the bias
option it accepts (`--bias`, `--bias_file`, `--biasfile`). If none is present,
a biased stage raises `CapabilityError`. Likewise, an ACO build exposing an
ant-count option gets the search breadth through that option rather than
through `--exhaustiveness`, since the two are not the same parameter.

## 4. Annealing

Vina-family engines have no hook to vary bias during a run, so the schedule is
a sequence of runs. With `n` stages and progress `p = k/(n-1)`:

```
w_k = w_0 · exp(-p / tau)        for k < n-1
w_{n-1} = 0
```

Each stage starts from the best pose of the previous one. The last stage runs
unbiased and, where the engine supports `--local_only`, as a local refinement.
That matters for interpretation: the pose and score you report come from a
stage in which no bias potential was present, so the bias influenced *where
the search looked*, not *what the final score says*.

### Diversity guardrail

Strong bias can collapse the ensemble onto one position, which is exactly when
the search stops exploring. After each stage the mean pairwise heavy-atom RMSD
across the output models is measured. Below a floor (default 1.2 Å) the next
stage's weight is attenuated:

```
w <- w · (diversity / floor)²
```

so it reaches zero at zero diversity. The quantity is measured from the
engine's multi-model output; it is not a function of stage index.

## 5. Evaluation

**RMSD.** Three methods, and every record states which was used:

| Method | Meaning |
|---|---|
| `graph_symmetry` | RDKit `GetBestRMS`: minimises over graph automorphisms. This is what "symmetry-corrected RMSD" means in the pose-prediction literature. |
| `assignment_lower_bound` | Optimal one-to-one assignment within each element class (Hungarian). Ignores bonding, so it can pair atoms that are not topologically equivalent: a lower bound on the above. |
| `ordered` | Index-to-index. Valid only when both files share atom ordering; inflated for symmetric groups. |

If the two files disagree in element composition, no number is produced and
the reason is recorded — rather than a NaN that reads as missing data.

**Arms.** `baseline` is engine defaults with no bias; `guided` is the selected
budget plus the schedule. Both run on every complex with the same seeds, so
the comparison is paired and within-complex.

**Statistics.** The unit of analysis is the complex. Per complex, each arm is
collapsed over seeds by majority (or `any`). Reported: success rate per arm,
the paired difference with a bootstrap 95% CI over complexes, the discordant
pair counts, an exact two-sided McNemar p-value on those pairs, and the median
within-complex runtime ratio.

A 2-point difference on 20 complexes is one discordant pair; the CI will say
so.

## 6. Budget selection

`HeuristicSelector` is a prior: flexibility bands at ≤3, 4–8 and >8 rotatable
bonds, with exhaustiveness scaled by `(box volume / 8000 Å³)^0.25` clamped to
[0.85, 1.35]. These numbers were chosen to be reasonable, not fitted, and the
plan's `notes` field says so on every run.

`TrainedSelector` replaces it. The sweep runs every configuration in
`grid.DEFAULT_GRID` on every complex. For each configuration,
`train_selector.py` fits a ridge-damped logistic regression for
P(top-1 ≤ cutoff) and a ridge regression for runtime, over the descriptors in
`planner.FEATURE_ORDER`. Coefficients are fitted on standardised features and
converted back to raw units, so the exported JSON is readable and applies
directly to descriptor values.

Prediction picks the cheapest configuration whose predicted success is within
2% of the best predicted success — the selector's job is to avoid paying for
budget that buys nothing, not to maximise accuracy at any cost.

Validation is grouped by complex: a model fitted and evaluated on the same
targets says nothing about a new one. The report gives the selected
configuration's held-out success rate next to the oracle rate (best available
configuration per complex) and the cheapest-configuration rate, which is the
baseline the selector has to beat to be worth having.

## What is not here

- No enrichment metrics. This evaluates pose prediction; screening power
  against decoy sets is a different experiment with different inputs.
- No PoseBusters checks, so a pose within 2 Å may still be physically
  implausible.
- A single spherical bias well, not a pharmacophore.
