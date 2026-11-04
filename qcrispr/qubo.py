"""Guide-set selection as a QUBO.

One binary variable per candidate guide: x_i = 1 means "include guide i in the
multiplex set". The objective has four parts.

Linear, per guide:
    + w_offtarget * (burden_i / burden_scale)      predicted off-target load
    - w_efficiency * efficiency_i                  on-target efficiency

Quadratic, per pair:
    + w_shared  * shared_loci(i, j) / shared_scale   two guides whose off-target
                                                     sets collide at the same
                                                     locus, so that locus gets
                                                     cut twice
    + w_spacing                                      cuts closer than
                                                     min_spacing on the target,
                                                     which deletes the span
                                                     between them

Constraint:
    + A * (sum_i x_i - k)^2                        select exactly k guides

The quadratic terms are the reason this is a QUBO and not a sorted list. Without
them, minimising a sum of independent per-guide costs is solved by taking the k
smallest, and no optimiser of any kind is needed. The pair interactions are what
make the choice combinatorial: the best set of three guides is not necessarily
the three best guides.

Energy convention: energy(x) = x^T Q x + offset, with linear coefficients on the
diagonal (valid because x_i^2 = x_i for binary x) and off-diagonal pair
coefficients split symmetrically across (i, j) and (j, i).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .design import GuideCandidate


@dataclass
class QUBO:
    Q: np.ndarray
    offset: float = 0.0
    labels: tuple[str, ...] = ()
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.Q.shape[0])

    def energy(self, x) -> float:
        xv = np.asarray(x, dtype=np.float64)
        return float(xv @ self.Q @ xv + self.offset)

    def cardinality(self, x) -> int:
        return int(np.sum(np.asarray(x, dtype=np.int64)))


class QUBOBuilder:
    def __init__(self, n: int) -> None:
        self.Q = np.zeros((n, n), dtype=np.float64)
        self.offset = 0.0

    def linear(self, i: int, c: float) -> None:
        self.Q[i, i] += c

    def quadratic(self, i: int, j: int, c: float) -> None:
        if i == j:
            self.Q[i, i] += c
        else:
            self.Q[i, j] += c / 2.0
            self.Q[j, i] += c / 2.0

    def constant(self, c: float) -> None:
        self.offset += c

    def build(self, labels: tuple[str, ...] = (), meta: dict | None = None) -> QUBO:
        return QUBO(self.Q, self.offset, labels, meta or {})


def shared_loci(a: GuideCandidate, b: GuideCandidate) -> int:
    return len(a.loci & b.loci)


def build_guide_qubo(
    candidates: list[GuideCandidate],
    k: int,
    w_offtarget: float = 1.0,
    w_efficiency: float = 1.0,
    w_shared: float = 1.0,
    w_spacing: float = 1.0,
    min_spacing: int = 50,
    penalty: float | None = None,
) -> QUBO:
    """Assemble the QUBO for choosing exactly `k` of `candidates`."""
    n = len(candidates)
    if n == 0:
        raise ValueError("no candidates")
    if not 0 < k <= n:
        raise ValueError(f"k must be in 1..{n}, got {k}")

    burden_scale = max((c.burden for c in candidates), default=0.0) or 1.0

    linear = np.zeros(n)
    for i, c in enumerate(candidates):
        linear[i] = (
            w_offtarget * (c.burden / burden_scale) - w_efficiency * c.efficiency
        )

    pair = np.zeros((n, n))
    raw_shared = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            raw_shared[i, j] = raw_shared[j, i] = shared_loci(
                candidates[i], candidates[j]
            )
    shared_scale = raw_shared.max() or 1.0

    for i in range(n):
        for j in range(i + 1, n):
            value = w_shared * (raw_shared[i, j] / shared_scale)
            same_chrom = candidates[i].site.chrom == candidates[j].site.chrom
            if same_chrom and abs(candidates[i].cut - candidates[j].cut) < min_spacing:
                value += w_spacing
            pair[i, j] = pair[j, i] = value

    # The cardinality penalty has to outweigh any gain from adding or dropping a
    # guide, otherwise the optimum quietly ignores k. Moving one step away from
    # k changes (sum - k)^2 by at least 1, so A above the largest single-variable
    # swing is sufficient.
    if penalty is None:
        max_linear = float(np.abs(linear).max())
        max_row = float(np.abs(pair).sum(axis=1).max())
        penalty = 2.0 * (max_linear + max_row) + 1.0

    builder = QUBOBuilder(n)
    for i in range(n):
        builder.linear(i, linear[i])
    for i in range(n):
        for j in range(i + 1, n):
            if pair[i, j]:
                builder.quadratic(i, j, pair[i, j])

    # A * (sum x - k)^2 = A * (sum x_i + 2*sum_{i<j} x_i x_j - 2k*sum x_i + k^2)
    for i in range(n):
        builder.linear(i, penalty * (1.0 - 2.0 * k))
    for i in range(n):
        for j in range(i + 1, n):
            builder.quadratic(i, j, 2.0 * penalty)
    builder.constant(penalty * k * k)

    return builder.build(
        labels=tuple(c.label for c in candidates),
        meta={
            "k": k,
            "penalty": penalty,
            "burden_scale": burden_scale,
            "shared_scale": shared_scale,
            "min_spacing": min_spacing,
            "weights": {
                "offtarget": w_offtarget,
                "efficiency": w_efficiency,
                "shared": w_shared,
                "spacing": w_spacing,
            },
        },
    )
