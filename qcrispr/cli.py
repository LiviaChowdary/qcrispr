"""Command line interface.

    python -m qcrispr.cli demo-genome
    python -m qcrispr.cli candidates --top 20
    python -m qcrispr.cli solve --k 3 --n 14 --p 3
    python -m qcrispr.cli benchmark --instances 8 --n 12 --k 3
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from .design import collect_candidates, prefilter
from .offtarget import GenomeIndex
from .qubo import build_guide_qubo
from .scoring import MismatchModel, load_cfd_model
from .sequence import make_demo_genome, read_fasta, write_fasta
from .solvers import (
    SolverResult,
    brute_force,
    constrained_annealing,
    feasible_spectrum,
    greedy,
    qaoa,
    rank_of,
    repair_to_cardinality,
    simulated_annealing,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GENOME = ROOT / "data" / "demo_genome.fa"
DEFAULT_PLANTED = ROOT / "data" / "demo_planted.tsv"
DEFAULT_RESULTS = ROOT / "results" / "benchmark.tsv"

PLANTED_COLUMNS = (
    "guide", "chrom", "start", "site_start", "strand", "protospacer", "pam",
    "mismatches", "bulge",
)
BENCHMARK_COLUMNS = (
    "instance", "n", "k", "solver", "p", "energy", "optimum", "gap", "matched",
    "weight", "feasible_mass", "seconds",
)


DEFAULT_CFD_MISMATCH = ROOT / "data" / "cfd_mismatch.tsv"
DEFAULT_CFD_PAM = ROOT / "data" / "cfd_pam.tsv"


def _load_model(args) -> MismatchModel | None:
    """Prefer the published CFD tables; fall back only if they are absent.

    Returning None means the parameterised model in scoring.py is used, which
    is a development convenience and not a substitute for the real weights.
    """
    if getattr(args, "parameterised", False):
        return None
    mismatch = Path(args.cfd_weights)
    pam = Path(args.cfd_pam)
    if mismatch.exists() and pam.exists():
        return load_cfd_model(mismatch, pam)
    return None


def _describe_model(model: MismatchModel | None) -> str:
    if model is None or not model.is_published:
        return "scoring: PARAMETERISED weights (not published CFD)"
    return f"scoring: published CFD tables ({len(model.table)} weights)"


def _build_index(args) -> GenomeIndex:
    records = read_fasta(args.genome)
    return GenomeIndex(records)


def _candidate_pool(args, index: GenomeIndex):
    model = _load_model(args)
    print(_describe_model(model))
    return collect_candidates(
        index,
        args.chrom,
        args.start,
        args.end,
        max_mismatches=args.max_mismatches,
        model=model,
        locus_window=args.locus_window,
        max_bulges=args.max_bulges_search,
    )


def cmd_demo_genome(args) -> None:
    seq, planted = make_demo_genome(
        length=args.length,
        seed=args.seed,
        target_start=args.target_start,
        target_len=args.target_len,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_fasta(out, {"demo_chr": seq})

    planted_path = Path(args.planted)
    with open(planted_path, "w") as fh:
        fh.write("\t".join(PLANTED_COLUMNS) + "\n")
        for row in planted:
            fh.write("\t".join(str(row[c]) for c in PLANTED_COLUMNS) + "\n")

    print(f"genome   {out}  ({len(seq):,} bp, seed {args.seed})")
    print(f"planted  {planted_path}  ({len(planted)} off-target sites)")
    print(
        f"target window  demo_chr:{args.target_start}-"
        f"{args.target_start + args.target_len}"
    )


def _candidate_table(title: str, rows) -> None:
    print(f"\n{title}")
    print(
        f"{'guide':>22}  {'locus':>18}  {'burden':>8}  "
        f"{'worst':>8}  {'n_off':>5}  {'loci':>5}  {'eff':>5}"
    )
    for c in rows:
        print(
            f"{c.spacer:>22}  {c.label:>18}  {c.burden:8.4f}  "
            f"{c.worst_offtarget:8.4f}  {len(c.offtargets):5d}  "
            f"{len(c.loci):5d}  {c.efficiency:5.2f}"
        )


def cmd_candidates(args) -> None:
    index = _build_index(args)
    pool = _candidate_pool(args, index)
    pool.sort(key=lambda c: (c.burden, -c.efficiency))
    print(f"indexed {len(index):,} protospacers across {len(index.records)} record(s)")
    print(f"{len(pool)} NGG candidates in {args.chrom}:{args.start}-{args.end}")

    zero = sum(1 for c in pool if c.burden == 0.0)
    total_off = sum(len(c.offtargets) for c in pool)
    all_loci = set().union(*(c.loci for c in pool)) if pool else set()
    shared = sum(
        1
        for i, a in enumerate(pool)
        for b in pool[i + 1 :]
        if a.loci & b.loci
    )
    print(
        f"{total_off} off-target hits over {len(all_loci)} distinct loci; "
        f"{zero} candidates with zero burden; {shared} colliding pairs"
    )

    _candidate_table("lowest burden (best guides)", pool[: args.top])
    _candidate_table("highest burden (worst guides)", pool[-args.top :][::-1])


def cmd_dataset(args) -> None:
    """Join measured CHANGE-seq reads onto the enumerated candidate sites."""
    from .data import build_dataset

    dataset = build_dataset(args.candidates, args.measured, out_path=args.out)
    print(dataset.summary(args.threshold))
    print(
        f"{dataset.dropped_unmatched:,} measured sites had no candidate row and "
        f"were dropped; those need bulge-aware enumeration to be scoreable at all"
    )
    print(f"wrote {args.out}")


def cmd_evaluate(args) -> None:
    """Cross-validate the model against published CFD, split by guide."""
    from .data import load_dataset
    from .evaluate import run_evaluation, write_results
    from .features import categorical_mask, cfd_column, feature_names, featurize

    dataset = load_dataset(args.dataset)
    model = load_cfd_model(args.cfd_weights, args.cfd_pam)
    extended = not args.basic_features
    names = feature_names(extended)
    print(dataset.summary(args.threshold))
    print(
        f"featurising {len(dataset):,} rows x {len(names)} features "
        f"({'with' if extended else 'without'} per-position mismatch identity)"
    )

    matrix = featurize(dataset, model, extended=extended)
    cfd = cfd_column(matrix).astype(np.float64)
    categorical = categorical_mask(extended)
    print(
        f"feature matrix {matrix.shape}, {matrix.nbytes / 1e6:.0f} MB, "
        f"{int(categorical.sum())} categorical\n"
    )

    reports = run_evaluation(
        dataset,
        matrix,
        cfd,
        n_folds=args.folds,
        threshold=args.threshold,
        seed=args.seed,
        max_iter=args.max_iter,
        categorical_features=categorical if categorical.any() else None,
    )
    print()
    for report in reports.values():
        print(report.summary())
    write_results(reports, args.out)
    print(f"\nwrote {args.out}")


def cmd_trueot(args) -> None:
    """Train on CHANGE-seq with bulges, test on the refined TrueOT benchmark.

    This is the published protocol: a biochemical assay for training, a
    cellular and individually validated benchmark for testing. Cross-assay
    transfer is the honest test, because in-vitro and in-cell readouts are
    different distributions.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    from .data import guides_in_csv, load_aligned_csv
    from .evaluate import average_precision, bootstrap_average_precision
    from .features import alignment_feature_names, features_from_alignments

    model = load_cfd_model(args.cfd_weights, args.cfd_pam)
    extended = not args.basic_features
    names = alignment_feature_names(extended=extended)

    # Leakage check first. Six of the ten refined-TrueOT guides also appear in
    # the CHANGE-seq training file, so training on everything and testing on
    # TrueOT scores the model partly on guides it has already seen. Excluding
    # them is the default; --keep-overlap reproduces the leaked number so the
    # size of the effect is visible rather than asserted.
    test_guides = guides_in_csv(args.test)
    train_guides = guides_in_csv(args.train)
    overlap = train_guides & test_guides
    share = len(overlap) / max(1, len(test_guides))
    print(
        f"guides: {len(train_guides)} train, {len(test_guides)} test, "
        f"{len(overlap)} shared ({share:.0%} of test)"
    )
    # Exclude every benchmark guide from every training source, not just the
    # ones that happen to overlap one file.
    excluded = set() if args.keep_overlap else test_guides
    print(
        "  KEEPING overlap, so this result is leaked"
        if args.keep_overlap
        else "  excluding shared guides from training"
    )

    print(f"\nloading training set (keeping {args.negative_rate:.0%} of negatives)")
    train_records, y_train, train_bulges, _, skipped = load_aligned_csv(
        args.train,
        "reads",
        threshold=args.threshold,
        negative_rate=args.negative_rate,
        seed=args.seed,
        exclude_guides=excluded or None,
    )
    print(
        f"  {len(train_records):,} rows, {int(y_train.sum()):,} active, "
        f"{int((train_bulges > 0).sum()):,} with bulges, {skipped:,} skipped"
    )

    print("loading refined TrueOT (no subsampling)")
    test_records, y_test, test_bulges, _, skipped_test = load_aligned_csv(
        args.test, "label", threshold=None
    )
    print(
        f"  {len(test_records):,} rows, {int(y_test.sum()):,} active, "
        f"{int((test_bulges > 0).sum()):,} with bulges, {skipped_test:,} skipped"
    )

    print(f"\nfeaturising with {len(names)} features")
    x_train = features_from_alignments(train_records, model, extended=extended)
    x_test = features_from_alignments(test_records, model, extended=extended)
    cfd_test = x_test[:, names.index("cfd")].astype(np.float64)

    print("training")
    regressor = HistGradientBoostingRegressor(
        max_iter=args.max_iter,
        learning_rate=0.1,
        max_leaf_nodes=63,
        min_samples_leaf=50,
        l2_regularization=1.0,
        random_state=args.seed,
    )
    regressor.fit(x_train, y_train.astype(np.float64))
    predictions = regressor.predict(x_test)

    if args.finetune:
        # Two-stage transfer. Stage one learns from a biochemical assay, which
        # is plentiful; stage two learns from a cellular assay, which is what
        # the benchmark measures, using the stage-one score as a feature. The
        # distribution shift between in-vitro and in-cell readouts is the
        # single largest gap between this pipeline and the published state of
        # the art, and training on one assay alone cannot close it.
        from .data import load_aligned_csv as _load

        print(f"\nfine-tuning on cellular data: {Path(args.finetune).name}")
        ft_records, y_ft, ft_bulges, _, ft_skipped = _load(
            args.finetune,
            "reads",
            threshold=args.finetune_threshold,
            negative_rate=args.negative_rate,
            seed=args.seed,
            exclude_guides=excluded or None,
        )
        print(
            f"  {len(ft_records):,} rows, {int(y_ft.sum()):,} active "
            f"(reads>{args.finetune_threshold:g}), "
            f"{int((ft_bulges > 0).sum()):,} with bulges, {ft_skipped:,} skipped"
        )
        x_ft = features_from_alignments(ft_records, model, extended=extended)
        x_ft_stacked = np.hstack(
            [x_ft, regressor.predict(x_ft)[:, None].astype(np.float32)]
        )
        x_test_stacked = np.hstack(
            [x_test, predictions[:, None].astype(np.float32)]
        )
        stage_two = HistGradientBoostingRegressor(
            max_iter=args.max_iter,
            learning_rate=0.1,
            max_leaf_nodes=63,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=args.seed,
        )
        stage_two.fit(x_ft_stacked, y_ft.astype(np.float64))
        predictions = stage_two.predict(x_test_stacked)

    subsets = {
        "all": np.ones(len(y_test), dtype=bool),
        "bulge only": test_bulges > 0,
        "mismatch only": test_bulges == 0,
    }
    rows: list[tuple] = []
    print()
    print(
        f"{'subset':>14}  {'n':>8}  {'active':>7}  {'CFD AUPR':>22}  "
        f"{'model AUPR':>22}"
    )
    for name, mask in subsets.items():
        if y_test[mask].sum() == 0:
            continue
        cfd_score, cfd_low, cfd_high = bootstrap_average_precision(
            y_test[mask], cfd_test[mask], n_boot=args.bootstrap, seed=args.seed
        )
        model_score, model_low, model_high = bootstrap_average_precision(
            y_test[mask], predictions[mask], n_boot=args.bootstrap, seed=args.seed
        )
        rows.append(
            (name, int(mask.sum()), int(y_test[mask].sum()), cfd_score, model_score)
        )
        print(
            f"{name:>14}  {int(mask.sum()):>8,}  {int(y_test[mask].sum()):>7}  "
            f"{cfd_score:7.4f} [{cfd_low:.4f},{cfd_high:.4f}]  "
            f"{model_score:7.4f} [{model_low:.4f},{model_high:.4f}]"
        )

    # Keep the predictions so this benchmark can be re-analysed without
    # retraining: a question about the metric should not cost a training run.
    predictions_path = Path(args.out).with_name("trueot_predictions.npz")
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        predictions_path,
        labels=y_test,
        predictions=predictions,
        cfd=cfd_test,
        bulges=test_bulges,
    )
    print(f"\nsaved predictions to {predictions_path}")

    print(
        "\npublished reference on this benchmark (Yaish & Orenstein 2024):\n"
        "  all sites   their model 0.535, prior tools <=0.237\n"
        "  bulge only  their model >=0.213, prior tools <=0.045"
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as handle:
        handle.write("subset\tn\tn_active\tcfd_aupr\tmodel_aupr\n")
        for name, n, active, cfd_score, model_score in rows:
            handle.write(f"{name}\t{n}\t{active}\t{cfd_score:.6f}\t{model_score:.6f}\n")
    print(f"\nwrote {out}")


def cmd_bulges(args) -> None:
    """Score the measured bulge sites that standard pipelines cannot reach."""
    from .data import load_bulge_sites
    from .evaluate import evaluate_bulge_sites

    sites, malformed = load_bulge_sites(args.measured)
    model = load_cfd_model(args.cfd_weights, args.cfd_pam)
    by_kind: dict[str, int] = {}
    for site in sites:
        by_kind[site.kind] = by_kind.get(site.kind, 0) + 1

    print(f"{len(sites):,} measured bulge sites ({malformed} malformed rows skipped)")
    for kind, count in sorted(by_kind.items()):
        print(f"  {kind} bulge: {count:,}")
    print(
        "\nThese have no candidate row under mismatch-only enumeration, so every\n"
        "model trained on the standard candidate set scores them not badly but\n"
        "never. There are no paired negatives, so this is rank correlation\n"
        "against measured reads, not AUPR.\n"
    )

    results = evaluate_bulge_sites(sites, model)
    print(f"{'subset':>10}  {'n':>7}  {'CFD rho':>9}  {'mismatch rho':>13}  {'median reads':>13}")
    for name in ("all", "rna", "dna"):
        if name not in results:
            continue
        r = results[name]
        print(
            f"{name:>10}  {int(r['n']):>7,}  {r['cfd_spearman']:9.4f}  "
            f"{r['mismatch_spearman']:13.4f}  {r['median_reads']:13.0f}"
        )


def cmd_validate(args) -> None:
    """Measure how many planted off-targets the enumerator actually recovers.

    A scoring model is worth nothing if the search step silently misses sites,
    and a miss is invisible without ground truth. The demo genome records every
    site it plants, so recall is directly measurable, broken down by mismatch
    count.
    """
    index = _build_index(args)

    planted: list[dict] = []
    with open(args.planted) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            if line.strip():
                planted.append(dict(zip(header, line.rstrip("\n").split("\t"))))
    if not planted:
        raise SystemExit(f"no planted sites in {args.planted}")

    by_guide: dict[str, list[dict]] = {}
    for row in planted:
        by_guide.setdefault(row["guide"], []).append(row)

    def label(row: dict) -> str:
        kind = row.get("bulge", "none")
        return f"{row['mismatches']} mismatch" if kind == "none" else f"{kind} bulge"

    def recall(bulge_budget: int) -> tuple[dict[str, list[int]], list[tuple[str, dict]]]:
        counts: dict[str, list[int]] = {}
        missed: list[tuple[str, dict]] = []
        for guide, rows in by_guide.items():
            hits = {
                o.site.start
                for o in index.find_offtargets(
                    guide,
                    max_mismatches=args.max_mismatches,
                    min_cfd=0.0,
                    max_bulges=bulge_budget,
                )
            }
            for row in rows:
                tally = counts.setdefault(label(row), [0, 0])
                tally[1] += 1
                if int(row["site_start"]) in hits:
                    tally[0] += 1
                else:
                    missed.append((guide, row))
        return counts, missed

    print(f"indexed {len(index):,} protospacers")
    print(
        f"{len(planted)} planted sites across {len(by_guide)} guides, "
        f"searching up to {args.max_mismatches} mismatches"
    )

    # Mismatch-only first, then with the bulge budget. The difference is the
    # whole argument for bulge-aware search: the sites a mismatch-only
    # enumerator misses are not ranked low, they are structurally invisible.
    without, _ = recall(0)
    with_bulges, missed = recall(args.max_bulges_search)

    print(f"\n{'site class':>16}  {'mismatch-only':>16}  {'with bulges':>16}")
    for key in sorted(with_bulges, key=lambda k: ("bulge" in k, k)):
        a_found, a_total = without.get(key, [0, 0])
        b_found, b_total = with_bulges[key]
        print(
            f"{key:>16}  {a_found:>5} / {a_total:<4} {a_found / a_total:6.0%}  "
            f"{b_found:>5} / {b_total:<4} {b_found / b_total:6.0%}"
        )

    a_found = sum(v[0] for v in without.values())
    b_found = sum(v[0] for v in with_bulges.values())
    total = len(planted)
    print(
        f"{'overall':>16}  {a_found:>5} / {total:<4} {a_found / total:6.0%}  "
        f"{b_found:>5} / {total:<4} {b_found / total:6.0%}"
    )

    if missed:
        print(f"\n{len(missed)} site(s) still missed with bulges enabled:")
        for guide, row in missed[:10]:
            print(
                f"  {guide}  {row['chrom']}:{row['site_start']}  "
                f"{row['protospacer']}|{row['pam']}  "
                f"{row['mismatches']}mm {row.get('bulge', 'none')}"
            )
        raise SystemExit(
            f"\nenumeration recall is {b_found / total:.1%}, not 100% -- "
            f"the search step is losing sites that were planted"
        )
    print("\nevery planted site was recovered with bulges enabled")


def _report(result, optimum: float | None) -> None:
    gap = "" if optimum is None else f"{result.energy - optimum:+.5f}"
    print(
        f"{result.name:>20}  energy {result.energy:10.5f}  gap {gap:>10}  "
        f"|S|={int(result.x.sum())}  {result.seconds * 1000:8.1f} ms"
    )
    if "expectation" in result.extra:
        print(
            f"{'':>20}  expectation {result.extra['expectation']:10.5f}  "
            f"argmax energy {result.extra['argmax_energy']:10.5f}  "
            f"p(argmax)={result.extra['top_probability']:.4f}  "
            f"feasible mass={result.extra['feasible_mass']:.4f}"
        )


def cmd_solve(args) -> None:
    index = _build_index(args)
    pool = _candidate_pool(args, index)
    if not pool:
        raise SystemExit("no candidates in that window")
    candidates = prefilter(pool, args.n)
    shared = dict(
        w_offtarget=args.w_offtarget,
        w_efficiency=args.w_efficiency,
        w_shared=args.w_shared,
        w_spacing=args.w_spacing,
        min_spacing=args.min_spacing,
    )
    qubo = build_guide_qubo(candidates, k=args.k, **shared)
    # The same problem without the cardinality penalty. The XY mixer enforces
    # "exactly k" through the state space instead, so it needs this version.
    objective = build_guide_qubo(candidates, k=args.k, penalty=0.0, **shared)

    print(f"indexed {len(index):,} protospacers")
    print(f"candidate pool {len(pool)} -> {len(candidates)} after prefilter")
    print(f"QUBO n={qubo.n}  k={args.k}  penalty={qubo.meta['penalty']:.3f}")
    pairs = qubo.n * (qubo.n - 1) // 2
    problem_pairs = int(np.count_nonzero(np.triu(objective.Q, 1)))
    print(
        f"off-diagonal couplings: {problem_pairs} of {pairs} pairs from the "
        f"problem itself (the penalty couples all {pairs})\n"
    )

    exact = brute_force(qubo) if qubo.n <= 22 else None
    optimum = exact.energy if exact else None

    # Say how hard the instance is before reporting who solved it.
    spectrum = feasible_spectrum(objective, args.k)
    greedy_probe = greedy(objective, args.k)
    print(
        f"instance hardness: {spectrum.size} feasible sets, "
        f"optimum {spectrum[0]:.5f}, worst {spectrum[-1]:.5f}, "
        f"greedy ranks {rank_of(greedy_probe.energy, spectrum)}/{spectrum.size}"
    )
    if rank_of(greedy_probe.energy, spectrum) == 1:
        print(
            "  NOTE: greedy is already optimal here, so this instance does not "
            "discriminate between solvers.\n"
        )
    else:
        print()

    results = []
    if exact:
        results.append(exact)
    results.append(greedy(objective, args.k))
    results.append(simulated_annealing(qubo, seed=args.seed))
    results.append(constrained_annealing(objective, args.k, seed=args.seed))

    qx = qaoa(qubo, p=args.p, seed=args.seed, mixer="x", k=args.k)
    results.append(qx)
    # Judge the penalty encoding on what a practitioner would ship: its output
    # repaired to feasibility, not its raw infeasible bitstring.
    repaired = repair_to_cardinality(qx.x, objective, args.k)
    results.append(
        SolverResult(
            "qaoa_x_repaired", repaired, objective.energy(repaired), qx.seconds, {}
        )
    )
    results.append(qaoa(objective, p=args.p, seed=args.seed, mixer="xy", k=args.k))

    for r in results:
        _report(r, optimum)

    # The two encodings must agree on every feasible point, or the energies
    # above are not comparable.
    xy = results[-1]
    if int(xy.x.sum()) == args.k:
        drift = abs(objective.energy(xy.x) - qubo.energy(xy.x))
        print(
            f"\nencoding check: penalised and penalty-free energies agree on the "
            f"xy solution to {drift:.2e}"
        )
    else:
        print(f"\nencoding check: xy solution has weight {int(xy.x.sum())}, expected {args.k}")

    best = exact or min(results, key=lambda r: r.energy)
    print(f"\nselected guide set (from {best.name}):")
    for i in best.support:
        c = candidates[i]
        print(
            f"  {c.spacer}  {c.label}  burden={c.burden:.4f}  "
            f"worst={c.worst_offtarget:.4f}  eff={c.efficiency:.2f}  cut={c.cut}"
        )


def cmd_benchmark(args) -> None:
    index = _build_index(args)
    pool = _candidate_pool(args, index)
    if len(pool) < args.n:
        raise SystemExit(f"pool has {len(pool)} candidates, need at least {args.n}")

    ps = [int(x) for x in args.p_values.split(",")]
    rng = random.Random(args.seed)
    rows: list[tuple] = []

    print(
        f"pool {len(pool)} candidates; {args.instances} instances "
        f"of n={args.n}, k={args.k}"
    )
    print(f"QAOA depths: {ps}\n")

    weights = dict(
        w_offtarget=args.w_offtarget,
        w_efficiency=args.w_efficiency,
        w_shared=args.w_shared,
        w_spacing=args.w_spacing,
        min_spacing=args.min_spacing,
    )

    def row(inst, n, solver, p, result, optimum):
        return (
            inst, n, args.k, solver, p, result.energy, optimum,
            result.energy - optimum,
            int(abs(result.energy - optimum) < 1e-9),
            int(result.x.sum()),
            result.extra.get("feasible_mass", float("nan")),
            result.seconds,
        )

    for inst in range(args.instances):
        candidates = rng.sample(pool, args.n)
        qubo = build_guide_qubo(candidates, k=args.k, **weights)
        objective = build_guide_qubo(candidates, k=args.k, penalty=0.0, **weights)

        exact = brute_force(qubo)
        optimum = exact.energy
        per_instance = [row(inst, qubo.n, "brute_force", 0, exact, optimum)]

        per_instance.append(
            row(inst, qubo.n, "greedy", 0, greedy(objective, args.k), optimum)
        )

        sa = simulated_annealing(qubo, seed=args.seed + inst)
        per_instance.append(row(inst, qubo.n, "penalised_annealing", 0, sa, optimum))

        csa = constrained_annealing(objective, args.k, seed=args.seed + inst)
        per_instance.append(row(inst, qubo.n, "constrained_annealing", 0, csa, optimum))

        for p in ps:
            qx = qaoa(
                qubo, p=p, seed=args.seed + inst,
                restarts=args.qaoa_restarts, mixer="x", k=args.k,
            )
            per_instance.append(row(inst, qubo.n, "qaoa_x", p, qx, optimum))
            fixed = repair_to_cardinality(qx.x, objective, args.k)
            per_instance.append(
                row(
                    inst, qubo.n, "qaoa_x_repaired", p,
                    SolverResult(
                        "qaoa_x_repaired", fixed, objective.energy(fixed),
                        qx.seconds, {},
                    ),
                    optimum,
                )
            )
            qxy = qaoa(
                objective, p=p, seed=args.seed + inst,
                restarts=args.qaoa_restarts, mixer="xy", k=args.k,
            )
            per_instance.append(row(inst, qubo.n, "qaoa_xy", p, qxy, optimum))

        rows.extend(per_instance)
        print(f"instance {inst}: optimum {optimum:.5f}")
        for r in per_instance:
            tag = f"{r[3]}" + (f" p={r[4]}" if r[4] else "")
            print(
                f"    {tag:>24}  energy {r[5]:10.5f}  gap {r[7]:+10.5f}  "
                f"|S|={r[9]:<3d} {r[11] * 1000:9.1f} ms"
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        fh.write("\t".join(BENCHMARK_COLUMNS) + "\n")
        for r in rows:
            fh.write("\t".join(str(v) for v in r) + "\n")

    print(f"\nwrote {out}")
    print("\nhit rate (instances reaching the exact optimum) and mean runtime:")
    combos: list[tuple[str, int]] = []
    for r in rows:
        if (r[3], r[4]) not in combos:
            combos.append((r[3], r[4]))
    for name, p in combos:
        sel = [r for r in rows if r[3] == name and r[4] == p]
        hits = sum(r[8] for r in sel)
        feasible = sum(1 for r in sel if r[9] == args.k)
        mean_ms = sum(r[11] for r in sel) / max(1, len(sel)) * 1000
        mean_gap = sum(r[7] for r in sel) / max(1, len(sel))
        label = name + (f" p={p}" if p else "")
        print(
            f"  {label:>24}  optimum {hits}/{len(sel)}  feasible {feasible}/{len(sel)}"
            f"  mean gap {mean_gap:+9.5f}  mean {mean_ms:9.1f} ms"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qcrispr", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_genome_args(p):
        p.add_argument("--genome", default=str(DEFAULT_GENOME))
        p.add_argument("--chrom", default="demo_chr")
        p.add_argument("--start", type=int, default=50_000)
        p.add_argument("--end", type=int, default=50_300)
        p.add_argument("--max-mismatches", dest="max_mismatches", type=int, default=4)
        p.add_argument(
            "--max-bulges",
            dest="max_bulges_search",
            type=int,
            default=1,
            help="bulge budget for off-target search (0 disables bulge alignment)",
        )
        p.add_argument(
            "--locus-window",
            dest="locus_window",
            type=int,
            default=30,
            help="cut sites within this many bp count as one off-target locus",
        )
        p.add_argument(
            "--cfd-weights",
            dest="cfd_weights",
            default=str(DEFAULT_CFD_MISMATCH),
            help="TSV of CFD mismatch weights (scripts/import_cfd.py writes it)",
        )
        p.add_argument(
            "--cfd-pam",
            dest="cfd_pam",
            default=str(DEFAULT_CFD_PAM),
            help="TSV of CFD PAM factors (scripts/import_cfd.py writes it)",
        )
        p.add_argument(
            "--parameterised",
            dest="parameterised",
            action="store_true",
            help="force the built-in approximation instead of the published tables",
        )

    def add_weight_args(p):
        p.add_argument("--w-offtarget", dest="w_offtarget", type=float, default=1.0)
        p.add_argument("--w-efficiency", dest="w_efficiency", type=float, default=1.0)
        p.add_argument("--w-shared", dest="w_shared", type=float, default=1.0)
        p.add_argument("--w-spacing", dest="w_spacing", type=float, default=1.0)
        p.add_argument("--min-spacing", dest="min_spacing", type=int, default=50)

    g = sub.add_parser("demo-genome", help="write a deterministic synthetic genome")
    g.add_argument("--out", default=str(DEFAULT_GENOME))
    g.add_argument("--planted", default=str(DEFAULT_PLANTED))
    g.add_argument("--length", type=int, default=120_000)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--target-start", dest="target_start", type=int, default=50_000)
    g.add_argument("--target-len", dest="target_len", type=int, default=300)
    g.set_defaults(func=cmd_demo_genome)

    c = sub.add_parser("candidates", help="list guides and their off-target burden")
    add_genome_args(c)
    c.add_argument("--top", type=int, default=20)
    c.set_defaults(func=cmd_candidates)

    d = sub.add_parser("dataset", help="build the measured CHANGE-seq dataset")
    d.add_argument(
        "--candidates",
        default=str(ROOT / "external" / "syseval" / "output_file_pam_change.txt"),
    )
    d.add_argument(
        "--measured", default=str(ROOT / "external" / "syseval" / "changeseq.xlsx")
    )
    d.add_argument("--out", default=str(ROOT / "data" / "changeseq.npz"))
    d.add_argument("--threshold", type=float, default=100.0)
    d.set_defaults(func=cmd_dataset)

    e = sub.add_parser("evaluate", help="cross-validate against published CFD")
    e.add_argument("--dataset", default=str(ROOT / "data" / "changeseq.npz"))
    e.add_argument("--cfd-weights", dest="cfd_weights", default=str(DEFAULT_CFD_MISMATCH))
    e.add_argument("--cfd-pam", dest="cfd_pam", default=str(DEFAULT_CFD_PAM))
    e.add_argument("--folds", type=int, default=10)
    e.add_argument("--threshold", type=float, default=100.0)
    e.add_argument("--max-iter", dest="max_iter", type=int, default=200)
    e.add_argument(
        "--basic-features",
        dest="basic_features",
        action="store_true",
        help="drop the per-position mismatch-identity columns",
    )
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out", default=str(ROOT / "results" / "offtarget_eval.tsv"))
    e.set_defaults(func=cmd_evaluate)

    t = sub.add_parser("trueot", help="train on CHANGE-seq, test on refined TrueOT")
    t.add_argument(
        "--train",
        default=str(
            ROOT / "external" / "crispr_bulge" / "datasets" / "CHANGEseq"
            / "include_on_targets" / "CHANGEseq_CR_Lazzarotto_2020_dataset.csv"
        ),
    )
    t.add_argument(
        "--test",
        default=str(
            ROOT / "external" / "crispr_bulge" / "datasets" / "Refined_TrueOT.csv"
        ),
    )
    t.add_argument("--cfd-weights", dest="cfd_weights", default=str(DEFAULT_CFD_MISMATCH))
    t.add_argument("--cfd-pam", dest="cfd_pam", default=str(DEFAULT_CFD_PAM))
    t.add_argument("--threshold", type=float, default=100.0)
    t.add_argument("--negative-rate", dest="negative_rate", type=float, default=0.3)
    t.add_argument("--max-iter", dest="max_iter", type=int, default=300)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--basic-features", dest="basic_features", action="store_true")
    t.add_argument(
        "--keep-overlap",
        dest="keep_overlap",
        action="store_true",
        help="do NOT exclude shared guides; reproduces the leaked result",
    )
    t.add_argument(
        "--finetune",
        default=None,
        help="cellular assay CSV to fine-tune on after biochemical training",
    )
    t.add_argument(
        "--finetune-threshold",
        dest="finetune_threshold",
        type=float,
        default=0.0,
        help="activity threshold for the fine-tuning set (GUIDE-seq uses >0)",
    )
    t.add_argument("--bootstrap", type=int, default=500)
    t.add_argument("--out", default=str(ROOT / "results" / "trueot.tsv"))
    t.set_defaults(func=cmd_trueot)

    b2 = sub.add_parser("bulges", help="score the measured bulge sites")
    b2.add_argument(
        "--measured", default=str(ROOT / "external" / "syseval" / "changeseq.xlsx")
    )
    b2.add_argument("--cfd-weights", dest="cfd_weights", default=str(DEFAULT_CFD_MISMATCH))
    b2.add_argument("--cfd-pam", dest="cfd_pam", default=str(DEFAULT_CFD_PAM))
    b2.set_defaults(func=cmd_bulges)

    v = sub.add_parser(
        "validate", help="measure enumeration recall against planted ground truth"
    )
    add_genome_args(v)
    v.add_argument("--planted", default=str(DEFAULT_PLANTED))
    v.set_defaults(func=cmd_validate)

    s = sub.add_parser("solve", help="build the QUBO and solve it three ways")
    add_genome_args(s)
    add_weight_args(s)
    s.add_argument("--k", type=int, default=3)
    s.add_argument("--n", type=int, default=14, help="candidates kept for the QUBO")
    s.add_argument("--p", type=int, default=3, help="QAOA depth")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_solve)

    b = sub.add_parser("benchmark", help="compare solvers across random instances")
    add_genome_args(b)
    add_weight_args(b)
    b.add_argument("--instances", type=int, default=8)
    b.add_argument("--n", type=int, default=12)
    b.add_argument("--k", type=int, default=3)
    b.add_argument("--p-values", dest="p_values", default="1,2,3")
    b.add_argument(
        "--qaoa-restarts", dest="qaoa_restarts", type=int, default=3,
        help="random restarts per QAOA run, on top of the ramp warm start",
    )
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", default=str(DEFAULT_RESULTS))
    b.set_defaults(func=cmd_benchmark)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
