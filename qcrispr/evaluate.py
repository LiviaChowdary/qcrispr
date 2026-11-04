"""Leave-guides-out evaluation on measured CHANGE-seq data.

The protocol follows what this literature argues for, not what is convenient.

**Split by guide, never at random.** The same sgRNAs recur across public
off-target tables, so a random row split puts near-identical sites on both
sides of the boundary and inflates everything. Folds here are groups of
guides, matching the leave-11-guides-out design used on this dataset.

**Score per guide, then report the distribution.** Yaish and Orenstein put it
bluntly: evaluating all sgRNAs pooled together "is not informative", because
the guides with the most sites dominate the metric. A real design decision
concerns one guide at a time, so the per-guide spread is the thing to look at.

**Report AUPR, not ROC-AUC.** ROC-AUC is not invalid here, it is saturated:
at imbalance past 500:1 a model can read 0.99 while finding half the sites.
Published numbers on this data make the point, with AUROC 0.996 reported
alongside AUPRC 0.520 for the same model.

**Report rank correlation on read counts too.** Activity is a measured
quantity, not a class, and thresholding throws that away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .data import ACTIVE_READ_THRESHOLD, OffTargetDataset

RESULT_COLUMNS = ("model", "guide_index", "n_sites", "n_active", "aupr", "spearman")


@dataclass
class GuideScore:
    model: str
    guide_index: int
    n_sites: int
    n_active: int
    aupr: float
    spearman: float


@dataclass
class ModelReport:
    name: str
    scores: list[GuideScore] = field(default_factory=list)

    def _finite(self, attribute: str) -> np.ndarray:
        values = np.array([getattr(s, attribute) for s in self.scores], dtype=float)
        return values[np.isfinite(values)]

    def summary(self) -> str:
        aupr = self._finite("aupr")
        rho = self._finite("spearman")
        return (
            f"{self.name:>22}  "
            f"AUPR mean {aupr.mean():.4f}  median {np.median(aupr):.4f}  "
            f"p10 {np.percentile(aupr, 10):.4f}  p90 {np.percentile(aupr, 90):.4f}"
            f"  |  Spearman mean {rho.mean():.4f}  median {np.median(rho):.4f}"
            f"  (n={aupr.size} guides)"
        )


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve, as the step sum.

    Equivalent to sklearn's average_precision_score. Implemented here so the
    metric is auditable rather than a black box, since the whole argument of
    this module is about which metric gets reported.
    """
    positives = int(labels.sum())
    if positives == 0 or positives == labels.size:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    hits = labels[order].astype(np.float64)
    true_positives = np.cumsum(hits)
    precision = true_positives / np.arange(1, hits.size + 1)
    return float((precision * hits).sum() / positives)


def bootstrap_average_precision(
    labels: np.ndarray,
    scores: np.ndarray,
    n_boot: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Point AUPR plus a percentile bootstrap interval.

    Needed because this benchmark has 100 positives. A difference of a few
    hundredths between two models on 100 positives can be sampling noise, and
    without an interval there is no way to tell. Rows are resampled with
    replacement, so the positive count varies between replicates exactly as it
    would across repeated experiments.
    """
    point = average_precision(labels, scores)
    rng = np.random.default_rng(seed)
    n = labels.size
    draws: list[float] = []
    for _ in range(n_boot):
        index = rng.integers(0, n, size=n)
        value = average_precision(labels[index], scores[index])
        if np.isfinite(value):
            draws.append(value)
    if not draws:
        return point, float("nan"), float("nan")
    low, high = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(low), float(high)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    # Average ranks within tied groups. This matters a lot here: most read
    # counts are exactly zero, so ties are the common case.
    sorted_values = values[order]
    start = 0
    for i in range(1, sorted_values.size + 1):
        if i == sorted_values.size or sorted_values[i] != sorted_values[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation, with average ranks for ties."""
    if x.size < 3:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denominator = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denominator == 0:
        return float("nan")
    return float((rx * ry).sum() / denominator)


def guide_folds(n_guides: int, n_folds: int, seed: int = 0) -> list[np.ndarray]:
    """Partition guide indices into `n_folds` disjoint groups."""
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_guides)
    return [np.sort(part) for part in np.array_split(shuffled, n_folds)]


def score_per_guide(
    name: str,
    dataset: OffTargetDataset,
    predictions: np.ndarray,
    guide_subset: np.ndarray,
    threshold: float = ACTIVE_READ_THRESHOLD,
) -> list[GuideScore]:
    labels = dataset.labels(threshold)
    out: list[GuideScore] = []
    for guide in guide_subset:
        mask = dataset.guide_idx == guide
        y = labels[mask]
        if y.sum() == 0:
            continue
        score = predictions[mask]
        out.append(
            GuideScore(
                model=name,
                guide_index=int(guide),
                n_sites=int(mask.sum()),
                n_active=int(y.sum()),
                aupr=average_precision(y, score),
                spearman=spearman(score, dataset.reads[mask]),
            )
        )
    return out


def run_evaluation(
    dataset: OffTargetDataset,
    features: np.ndarray,
    cfd: np.ndarray,
    n_folds: int = 10,
    threshold: float = ACTIVE_READ_THRESHOLD,
    seed: int = 0,
    max_iter: int = 200,
    categorical_features: np.ndarray | None = None,
    verbose: bool = True,
) -> dict[str, ModelReport]:
    """Cross-validate a gradient-boosted model against the CFD baseline.

    The model regresses log1p(reads) rather than classifying. Yaish and
    Orenstein found a regression model also solves the classification task on
    this data, and regression keeps the measured magnitudes instead of
    collapsing them at a threshold.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    reports = {
        "cfd": ModelReport("cfd (published)"),
        "model": ModelReport("gradient boosting"),
    }
    target = np.log1p(dataset.reads.astype(np.float64))
    folds = guide_folds(dataset.n_guides, n_folds, seed=seed)

    for number, test_guides in enumerate(folds, start=1):
        test_mask = np.isin(dataset.guide_idx, test_guides)
        train_mask = ~test_mask

        regressor = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=0.1,
            max_leaf_nodes=63,
            min_samples_leaf=50,
            l2_regularization=1.0,
            random_state=seed,
            categorical_features=categorical_features,
        )
        regressor.fit(features[train_mask], target[train_mask])

        predictions = np.full(len(dataset), np.nan, dtype=np.float64)
        predictions[test_mask] = regressor.predict(features[test_mask])

        fold_scores = score_per_guide(
            "model", dataset, predictions, test_guides, threshold
        )
        reports["model"].scores.extend(fold_scores)
        reports["cfd"].scores.extend(
            score_per_guide("cfd", dataset, cfd, test_guides, threshold)
        )
        if verbose:
            recent = [s.aupr for s in fold_scores]
            mean = np.nanmean(recent) if recent else float("nan")
            print(
                f"  fold {number}/{len(folds)}: {len(test_guides)} test guides, "
                f"mean AUPR {mean:.4f}",
                flush=True,
            )
    return reports


def align_bulge_site(site) -> tuple[list[tuple[int, str, str]], str, int]:
    """(mismatches, PAM, bulge count) for a measured bulge site.

    For an RNA bulge the table already carries the alignment: guide position i
    faces sequence character i, and '-' means no DNA base there, so comparing
    index by index *is* the alignment. For a DNA bulge one extra base has to be
    attributed, so every skip position is tried and the best kept, the same
    rule `offtarget.py` uses.
    """
    guide = site.guide
    spacer_len = len(guide) - 3

    if site.kind == "rna":
        pam = site.sequence[spacer_len:]
        mismatches = [
            (i + 1, guide[i], site.sequence[i])
            for i in range(spacer_len)
            if site.sequence[i] != "-" and guide[i] != site.sequence[i]
        ]
        return mismatches, pam, site.sequence[:spacer_len].count("-")

    observed_spacer = site.sequence[: spacer_len + 1]
    pam = site.sequence[spacer_len + 1 :]
    best: list[tuple[int, str, str]] | None = None
    for skip in range(spacer_len + 1):
        collapsed = observed_spacer[:skip] + observed_spacer[skip + 1 :]
        mismatches = [
            (i + 1, guide[i], collapsed[i])
            for i in range(spacer_len)
            if guide[i] != collapsed[i]
        ]
        if best is None or len(mismatches) < len(best):
            best = mismatches
    return best or [], pam, 1


def evaluate_bulge_sites(sites: list, model) -> dict[str, dict[str, float]]:
    """Can a CFD-style score rank measured bulge-site activity?

    There are no negatives to pair with these sites, because the standard
    candidate enumeration cannot generate bulge-containing candidates at all.
    So this reports rank correlation against measured reads rather than AUPR,
    and says so instead of inventing a negative set.

    Two baselines, because a single number proves nothing: mismatch count
    alone (negated, since more mismatches should mean less cleavage) and the
    CFD score over the aligned mismatch positions.
    """
    from .scoring import cfd_score_alignment

    groups: dict[str, list[tuple[float, float, float]]] = {}
    for site in sites:
        mismatches, pam, n_bulges = align_bulge_site(site)
        if len(pam) != 3 or any(c not in "ACGT" for c in pam):
            continue
        cfd = cfd_score_alignment(mismatches, pam, model=model, n_bulges=n_bulges)
        row = (site.reads, cfd, -float(len(mismatches)))
        groups.setdefault("all", []).append(row)
        groups.setdefault(site.kind, []).append(row)

    out: dict[str, dict[str, float]] = {}
    for name, rows in groups.items():
        reads = np.array([r[0] for r in rows])
        out[name] = {
            "n": float(len(rows)),
            "cfd_spearman": spearman(np.array([r[1] for r in rows]), reads),
            "mismatch_spearman": spearman(np.array([r[2] for r in rows]), reads),
            "median_reads": float(np.median(reads)),
        }
    return out


def write_results(reports: dict[str, ModelReport], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\t".join(RESULT_COLUMNS) + "\n")
        for report in reports.values():
            for s in report.scores:
                handle.write(
                    f"{s.model}\t{s.guide_index}\t{s.n_sites}\t{s.n_active}\t"
                    f"{s.aupr:.6f}\t{s.spearman:.6f}\n"
                )
