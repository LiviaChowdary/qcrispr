# qcrispr

Guide-RNA selection for multiplex CRISPR editing, posed as a quadratic binary
optimization problem and solved several ways, including with QAOA.

Two results, both measured rather than asserted.

**Off-target prediction.** On the refined TrueOT benchmark, trained on
CHANGE-seq and fine-tuned on cellular GUIDE-seq with all benchmark guides
excluded: AUPR 0.4274, bootstrap interval [0.341, 0.522]. That is above the
<=0.237 ceiling reported for every prior published tool, and short of the
0.535 held by the transfer-learning model that defines the state of the art.
On bulge-containing sites, 0.2445 [0.112, 0.427] against <=0.045 for prior
tools.

**The quantum solver does not earn its place.** Across eight instances a
swap-neighbourhood annealer is exactly optimal on all of them in 1.2 s, while
QAOA with the best constraint encoding manages four and takes eight seconds. A
0.1 ms greedy beats QAOA on both hit rate and mean gap. That is reported here
rather than buried, because it is the result.

## What this does that existing tools do not

Every guide design tool scores guides one at a time. CRISPOR, Benchling, IDT,
Guide Picker and the rest rank candidates independently and leave you to take
the top k. The aggregate scores that exist, Elevation-aggregate,
MOFF-aggregate and crispAI-aggregate, collapse one guide's off-targets into a
single specificity number; crispAI's paper states plainly that it scores one
sgRNA at a time.

Nothing scores a *set* jointly. That matters for multiplex editing, where the
risk is not only that each guide cuts somewhere it should not, but that two
guides cut the *same* wrong place, or cut the target close enough together to
delete the span between them. Those are interactions between guides, so no
amount of per-guide ranking sees them.

This tool makes those interactions the objective:

```
minimise   sum_i [ w_off * burden_i  -  w_eff * efficiency_i ] * x_i
         + sum_i<j [ w_shared * shared_loci(i,j) + w_spacing * too_close(i,j) ] * x_i * x_j
subject to sum_i x_i = k
```

The quadratic term is the point. Drop it and the problem is a sort.

## Prior art

Singhal, Maheshwari and Joshi published
[COMET](https://arxiv.org/abs/2607.02622) (arXiv:2607.02622, 2 July 2026),
which formulates multiplex guide selection as a QUBO and compares a
penalty-encoded QAOA against an XY mixer on IBM hardware. If you are reading
this expecting the QUBO-plus-QAOA framing to be new, it is not.

Where this differs:

| | COMET | here |
|---|---|---|
| off-target scores | synthetic | published CFD tables (240 weights) |
| cross-guide coupling | `Q_ij = 0` within a gene | real shared off-target loci |
| initial state | one-hot product state | true Dicke state |
| bulges | not handled | DNA and RNA bulges, validated |
| classical baselines | brute force, constrained annealing | adds greedy, penalty+repair, hardness ranking |

COMET says its 12-qubit instance is classically trivial. This repo agrees
about its own instances and measures how trivial.

## Pipeline

```bash
pip install -e .
python scripts/import_cfd.py <mismatch.pkl> <pam.pkl>   # published CFD weights
qcrispr demo-genome                                      # synthetic test genome
qcrispr validate                                         # enumeration recall
qcrispr candidates --top 8
qcrispr solve --n 14 --k 4 --p 3
qcrispr benchmark --instances 8 --n 14 --k 4
```

`qcrispr/sequence.py` handles FASTA and PAM scanning. `offtarget.py` indexes
every PAM-adjacent protospacer in the genome and aligns guides against it.
`scoring.py` holds the CFD model. `design.py` assembles candidates,
`qubo.py` builds the objective, `solvers.py` has the solvers.

## Results

### Bulge-aware search finds a tenth of sites that mismatch-only search cannot

Roughly a fifth of experimentally measured off-targets involve a bulge, an
insertion or deletion between guide and DNA. A mismatch-only search cannot
represent them at any threshold. They are not scored low, they are absent. In
the published CHOPOFF comparison, Cas-OFFinder and CRISPRitz each missed
thousands of bulge sites at edit distance 2 to 3 for this reason.

Against 159 planted sites with known ground truth:

| site class | mismatch-only | with bulges |
|---|---|---|
| 1 mismatch | 36/36, 100% | 36/36, 100% |
| 2 mismatch | 37/37, 100% | 37/37, 100% |
| 3 mismatch | 32/32, 100% | 32/32, 100% |
| 4 mismatch | 30/30, 100% | 30/30, 100% |
| DNA bulge | 5/12, 42% | 12/12, 100% |
| RNA bulge | 3/12, 25% | 12/12, 100% |
| **overall** | **143/159, 90%** | **159/159, 100%** |

The bulge sites a mismatch-only search does find are coincidence: a bulge site
with few mismatches sometimes still aligns within the mismatch budget.

With at most one bulge the alignment needs no dynamic programming. The only
freedom is which single base to skip, so `offtarget.py` tries every choice and
keeps the best, which is exact and vectorises over the whole genome at once.

### The hand-written scoring model was not close

Before importing the published tables, this repo scored mismatches with a
smooth positional curve encoding "mismatches nearer the PAM are more
disruptive". Against the real 240-cell CFD table:

```
mean absolute error   0.219
max  absolute error   0.828
Pearson correlation   0.485
```

The failure is specific. Published CFD gives retained activity 1.000 for a
guide C against an observed A at position 16, and 0.938 for G against A at
position 20, both PAM-proximal, where the positional curve predicts 0.27 and
0.11. Mismatch identity matters far more than position. The familiar summary
is true on average and wrong cell by cell.

`scripts/import_cfd.py` imports the published tables without calling
`pickle.load`, since unpickling a downloaded file runs arbitrary code by
design. It verifies the file is pure-ASCII protocol 0, rejects every opcode
that can import a name or call a callable, then extracts the literals.

It also resolves a convention trap. A CFD key `rA:dG,1` names the base on the
strand *opposite* the protospacer, so the `d` base is the complement of what
you observe. Reading the observed base directly returns the wrong weight for
every mismatch, and the scores still look plausible. The importer complements
on import so everything downstream shares one convention.

### The quantum solver loses, and so does almost every other solver

One instance, 14 candidates choose 4, published CFD scoring:

| solver | energy | gap | feasible | time |
|---|---|---|---|---|
| brute force | −2.36943 | 0 | yes | 1.9 ms |
| **greedy** | **−2.36943** | **0** | yes | **0.1 ms** |
| penalised annealing | −0.00630 | +2.36 | yes | 254 ms |
| constrained annealing | −2.36943 | 0 | yes | 1.2 s |
| QAOA, penalty + X mixer | +21.05817 | +23.43 | **no** | 17.8 s |
| QAOA, penalty + X + repair | −2.36943 | 0 | yes | 17.8 s |
| QAOA, Dicke + XY mixer | −2.36943 | 0 | yes | 23.2 s |

A single instance overstates greedy, though. Across eight instances drawn from
the real candidate pool, 14 candidates choose 4:

| solver | optimum | feasible | mean gap | mean time |
|---|---|---|---|---|
| brute force | 8/8 | 8/8 | 0 | 1.6 ms |
| greedy | 6/8 | 8/8 | +0.114 | 0.1 ms |
| penalised annealing | 0/8 | 8/8 | +1.096 | 248 ms |
| **constrained annealing** | **8/8** | 8/8 | **0** | 1.2 s |
| QAOA penalty + X, p=2 | 0/8 | **0/8** | +40.68 | 4.7 s |
| QAOA penalty + X + repair, p=2 | 6/8 | 8/8 | +0.114 | 4.7 s |
| QAOA Dicke + XY, p=2 | 2/8 | 8/8 | +0.119 | 6.0 s |
| QAOA penalty + X, p=3 | 0/8 | **0/8** | +35.33 | 6.4 s |
| QAOA penalty + X + repair, p=3 | 6/8 | 8/8 | +0.114 | 6.4 s |
| QAOA Dicke + XY, p=3 | 4/8 | 8/8 | +0.147 | 8.0 s |

Greedy is not always optimal, so the instances do discriminate a little. The
constrained annealer is, on all eight, roughly 6,700 times faster than the
best QAOA configuration. Going from depth 2 to depth 3 raised the XY mixer's
hit rate from 2 to 4 but also raised its mean gap, which is what a small
sample of a noisy optimiser looks like rather than a trend. Repairing the
penalty encoding reproduces greedy's numbers exactly, at 47,000 times the
cost, because repairing a near-random bitstring is itself close to running
greedy.

Three things in that table are worth more than the quantum result.

The penalty-encoded QAOA put only 40% of its amplitude on feasible states and
returned an invalid three-guide set. The XY mixer, which conserves Hamming
weight by construction, stayed 100% feasible and found the optimum. That
reproduces COMET's central finding.

But repairing the penalty encoding's output, one greedy add or drop, also
reaches the optimum. So the XY mixer's advantage over a *practitioner's*
penalty pipeline is zero here, which matches the warning in the Ising-machine
benchmarking literature that cheap feasibility repair washes out the
difference between constraint encodings.

And the penalised annealer, which anneals single-flip moves against a penalty
of 23, lands nowhere near the optimum while the constrained annealer, which
swaps one selected guide for one unselected one and so never leaves the
feasible set, finds it every time. An earlier version of this repo shipped
only the penalised annealer, which made QAOA look good by comparison. That was
a measurement artifact of a bad baseline.

### How hard are these instances? Not very

Reporting solver results without reporting instance hardness is how this field
produces meaningless benchmarks. Greedy's rank among all feasible sets, on
instances drawn from the real candidate pool, 20 instances each:

| collision weight | greedy optimal | mean rank | worst rank |
|---|---|---|---|
| 1 | 19/20 | 1.1 | 2 of 1001 |
| 4 | 19/20 | 1.1 | 3 of 1001 |
| 16 | 17/20 | 1.7 | 8 of 1001 |

Greedy is optimal most of the time and never far off. There is no room for a
quantum advantage at this scale because there is barely room for any
optimizer. `qcrispr solve` prints this hardness line before the solver table
and says so explicitly when greedy is already optimal.

## Measured data: a model beats published CFD by 2.3x

The synthetic genome above is for testing the search. This section uses real
cleavage measurements.

CHANGE-seq, via `OrensteinLab/SysEvalOffTarget`: 3,567,646 candidate sites
enumerated at up to six mismatches, joined against 202,043 measured sites
across 110 guides. At the standard reads>100 activity threshold that gives
67,637 active sites, an imbalance of 1:52. The published construction reports
67,476, so the join reproduces it to within 161 sites out of 67,000.

Protocol: leave-11-guides-out, ten folds, metrics computed per guide and
reported as a distribution. 109 of 110 guides have at least one active site
and so are evaluable.

| | mean AUPR | median | p10 | p90 | mean Spearman |
|---|---|---|---|---|---|
| published CFD | 0.1917 | 0.1591 | 0.0963 | 0.2967 | 0.1792 |
| gradient boosting, 24 features | 0.4391 | 0.4335 | 0.2909 | 0.5881 | 0.2047 |
| **+ per-position mismatch identity** | **0.4756** | 0.4644 | 0.3241 | 0.6960 | 0.2024 |

This comparison is the one the field does not have. The largest independent
deep-learning benchmark on this data, Zhang et al. 2023 in *Briefings in
Bioinformatics*, omits CFD entirely, and the papers that do include it report
only that it loses. On identical folds, CFD reaches 0.19 and a gradient-boosted
model on the same alignment information reaches 0.48, two and a half times
better.

Adding per-position mismatch identity was worth +0.037 mean AUPR, and most of
it landed in the upper tail: p90 moved from 0.588 to 0.696 while the median
moved only 0.031. The model got better at the guides it was already handling
well rather than at the hard ones. Those twenty columns are categorical codes
(0 for a match, 1-12 for which mismatch pair), which is the compact form of
the sequence encoding the stronger published models use: as a one-hot it
would be 240 columns and 3.4 GB, as categoricals it is 20 columns and the
histogram booster splits on them natively. `--basic-features` reproduces the
24-feature row.

Still short of the published state of the art. Yaish, Asif and Orenstein
report average AUPR 0.643 on this dataset. The remaining gap is
representational: their model reads the full sequence context, while these
features summarise an alignment. Closing it needs a sequence model, not more
boosting rounds.

Note what the Spearman column does *not* do. It barely moves between the three
rows, while AUPR more than doubles. Ranking which sites are active and
predicting how active they are come apart here, so reporting only one of them
would hide half the behaviour.

### 10,202 measured sites cannot be scored at all

Five percent of the measured sites have no candidate row. The candidate file
was enumerated mismatch-only, so a site whose alignment needs an insertion or
deletion has nothing to attach to at any mismatch threshold. The loader
reports the count rather than dropping them quietly.

That is the same blind spot the bulge section demonstrates on planted sites,
measured here on real cleavage data. The measured table writes the alignment
in directly, which makes these sites recoverable: a 23nt sequence containing
`-` marks a position where the DNA has no base, and a 24nt sequence carries
one extra DNA base.

`qcrispr bulges` pulls them out and scores them: 10,352 sites, 7,353 RNA
bulges and 2,999 DNA bulges, with 163 malformed spreadsheet rows skipped.

| subset | n | CFD-style rho | mismatch-count rho | median reads |
|---|---|---|---|---|
| all | 10,350 | **0.2248** | 0.1655 | 34 |
| RNA bulge | 7,351 | 0.2044 | 0.1586 | 32 |
| DNA bulge | 2,999 | 0.2774 | 0.1855 | 40 |

Scoring the aligned mismatch positions beats counting mismatches by about a
third in rank correlation, and DNA bulges rank better than RNA bulges. Both
numbers are low in absolute terms, which is the expected state of this
subproblem.

These are rank correlations against measured reads, **not** AUPR, and they are
not comparable to the ~0.045 AUPR that bulge-capable classifiers report. The
reason is structural: there are no paired negatives to compute AUPR against,
because standard enumeration cannot generate bulge-containing candidates in
the first place. Producing a negative set means bulge-aware enumeration across
the genome, which `offtarget.py` can do and this evaluation has not yet done.

## The published benchmark: refined TrueOT

The evaluation above is within-assay. This one is the protocol the field
treats as honest: train on a biochemical assay, test on a cellular,
individually validated benchmark, because in-vitro and in-cell readouts are
different distributions.

Training set: the bulge-containing CHANGE-seq table from
`OrensteinLab/CRISPR-Bulge`, 4,936,278 aligned sites of which 1,749,342 carry
a bulge. Benchmark: refined TrueOT, 354,352 sites, 100 active, 27 of those
with bulges, which matches the published composition exactly.

### A 60% leakage problem, found by checking

Six of the ten refined-TrueOT guides also appear in the CHANGE-seq training
file. Training on all of it scores the model partly on guides it has already
seen. Guide exclusion is therefore the default, and `--keep-overlap`
reproduces the leaked number so the effect is measurable:

| subset | leaked | leakage-free | leakage-free + transfer |
|---|---|---|---|
| all | 0.2041 | 0.2878 | **0.4274** |
| bulge only | 0.1314 | 0.1876 | **0.2445** |
| mismatch only | 0.2374 | 0.3274 | **0.4989** |

Removing training data raised every score, which is backwards. The bootstrap
below explains why that was not worth celebrating: with 100 positives the
intervals are wide enough that an 0.08 shift is not clearly a real effect.

### Transfer learning is the whole gap

Training on a biochemical assay and testing on a cellular one asks the model
to cross a distribution shift. The fix is the one the published work uses:
train on CHANGE-seq, then fine-tune on cellular GUIDE-seq, here by feeding the
biochemical model's score to a second model trained on 903,340 cellular rows
from 72 guides. All ten benchmark guides are excluded from both sources.

That single change moved the full benchmark from 0.2878 to 0.4274, which
dwarfs every feature-engineering gain in this repo. The within-assay advantage
really does not survive the assay shift, and no amount of better features
substitutes for training on the readout you will be scored on.

### What the intervals establish, and what they do not

AUPR with 95% percentile bootstrap intervals, 500 resamples:

| subset | n | active | CFD | this model | prior tools | Yaish & Orenstein |
|---|---|---|---|---|---|---|
| all | 351,483 | 100 | 0.1446 [0.067, 0.193] | **0.4274 [0.341, 0.522]** | <=0.237 | 0.535 |
| bulge only | 118,170 | 27 | 0.0293 [0.012, 0.072] | **0.2445 [0.112, 0.427]** | <=0.045 | >=0.213 |
| mismatch only | 233,313 | 73 | 0.2335 [0.149, 0.332] | **0.4989 [0.397, 0.600]** | - | - |

Established:

* It beats every prior published tool on this benchmark. The lower bound,
  0.341, sits above the <=0.237 ceiling reported for COSMID, CRISTA,
  CRISPR-Net and CRISPR-IP.
* On bulge sites it beats prior tools by a wide margin: lower bound 0.112
  against their <=0.045, roughly eight times the point estimate of the best
  prior tool.
* It beats CFD on every subset, with non-overlapping intervals on the full
  benchmark.

Not established:

* Parity with the 0.535 state of the art. The upper bound is 0.522, just
  short, so the honest statement is that this is close to but still behind
  the best published model.
* Beating the specialist bulge model. The point estimate of 0.2445 is above
  its >=0.213, but with 27 positives the interval runs to 0.427 and contains
  it. A point estimate above a published number is not a win when the
  interval is this wide.

That last distinction is the reason the intervals are here at all. With 100
positives it would have been easy to report 0.2445 against 0.213 as a victory.

Caveats attached to every number above: training keeps 30% of negatives to
make 4.9M rows tractable, which shifts the class balance the model sees but
not the ranking it is scored on, since the benchmark is never subsampled.
2,869 benchmark rows (0.81%) failed alignment parsing and were dropped, none
of them active, so no positive was lost. Predictions are saved to
`results/trueot_predictions.npz` so the metric can be re-analysed without
retraining.

## What is not done

The genome is synthetic. The next real step is CHANGE-seq: 110 guides and
202,043 measured off-target sites, GEO `GSE149295` and BioProject
`PRJNA625995`, with a tidied version at `OrensteinLab/SysEvalOffTarget`. Until
then every number here describes a toy.

The on-target score is a crude proxy, three design rules in a trig function,
not a trained model. `pip install rs3` gives Rule Set 3 and is maintained.

Bulge scoring is a placeholder. CFD has no bulge term and none exists;
`BULGE_PENALTY = 0.3` makes bulge sites cost something rather than scoring as
perfect matches. It is fitted to nothing. Bulge-capable predictors reach about
0.045 AUPR on bulge-only subsets, so nobody scores these well. What matters is
that the sites are found.

The QAOA ceiling is 18 qubits of exact statevector, so candidates are
pre-filtered by a linear score before the quantum solver sees them. Any guide
dropped there is invisible no matter how good the optimisation is. The
classical solvers never need that filter.

No claim here is validated against measured cleavage. The scores are unitless.

## Tests

```bash
pytest
```

They cover the strand-coordinate mapping, the bulge alignment in both
directions, the CFD convention and its refusal to fall back silently when a
real table is loaded, the QUBO energy convention, Dicke state preparation,
that the XY mixer conserves Hamming weight at any angle, that the swap
neighbourhood's incremental delta matches full recomputation, and that no
solver ever beats the exact optimum.

One test asserts the hand-written parameterisation is *worse* than published
CFD by a wide margin, so that silently failing to load the real table shows up
as a failure instead of as slightly different numbers.
