"""Features for off-target activity prediction.

The feature set is deliberately small and mechanistic rather than a learned
sequence embedding. Two reasons.

First, the published evidence says most of the signal is cheap. Yaish, Asif
and Orenstein found that adding a single mismatch-distance feature lifted
median AUPR from 0.217 to 0.515 on this dataset. Feature choice dominates
model capacity here.

Second, a small feature set makes the comparison against CFD interpretable.
CFD is itself a product of per-mismatch retention weights, so it appears here
as one feature among many, and whatever the model adds on top of it is
visible as the gap between the two.

Everything is computed from the published CFD tables plus the alignment, so
there is nothing fitted in this module.
"""

from __future__ import annotations

import numpy as np

from .data import OffTargetDataset
from .scoring import MismatchModel

GUIDE_LEN = 20
FEATURE_NAMES = (
    "mismatch_count",
    "cfd",
    "log_cfd",
    "pam_factor",
    "mm_pos_1_5",
    "mm_pos_6_10",
    "mm_pos_11_15",
    "mm_pos_16_20",
    "min_retention",
    "retention_product",
    "retention_deficit",
    "most_pam_proximal_mm",
    "most_pam_distal_mm",
    "proto_gc",
    "guide_gc",
    "gc_difference",
    "n_transitions",
    "n_transversions",
    "n_wobble",
    "pam_base2",
    "pam_base3",
    "mean_mm_position",
    "max_consecutive_mm",
    "seed_mismatches",
)

# A -> 0, C -> 1, G -> 2, T -> 3 (see sequence.encode)
_TRANSITION_PAIRS = {(0, 2), (2, 0), (1, 3), (3, 1)}
_WOBBLE_PAIRS = {(2, 3), (3, 2)}  # guide G vs observed T, guide T vs observed G


def retention_lookup(model: MismatchModel) -> np.ndarray:
    """(20, 4, 4) retained activity by position, guide base, observed base.

    Diagonal entries are 1.0: a match costs nothing. A cell the published
    table does not define would be a silent hole, so any such cell is filled
    with the table minimum. CFD defines all 12 mismatch pairs at all 20
    positions, so this should not trigger.
    """
    table = np.ones((GUIDE_LEN, 4, 4), dtype=np.float64)
    bases = "ACGT"
    for position in range(1, GUIDE_LEN + 1):
        for g, guide_base in enumerate(bases):
            for d, observed_base in enumerate(bases):
                if guide_base == observed_base:
                    continue
                try:
                    table[position - 1, g, d] = model.retention(
                        position, guide_base, observed_base
                    )
                except KeyError:
                    table[position - 1, g, d] = np.nan
    if np.isnan(table).any():
        table = np.where(np.isnan(table), np.nanmin(table), table)
    return table


def pam_lookup(model: MismatchModel) -> np.ndarray:
    """(4, 4) PAM factor by the two PAM bases that CFD scores."""
    bases = "ACGT"
    table = np.zeros((4, 4), dtype=np.float64)
    for i, first in enumerate(bases):
        for j, second in enumerate(bases):
            table[i, j] = model.pam_factor("N" + first + second)
    return table


# Per-position mismatch identity, as one categorical column per guide
# position: 0 for a match, 1-12 for which of the twelve mismatch pairs it is.
#
# This is the compact form of the "sequence" encoding that the strongest
# published models on this dataset use. A one-hot over 20 positions x 12 pairs
# would be 240 columns and about 3.4 GB at float32; as 20 categoricals it is
# 20 columns, and a histogram-based gradient booster splits on them natively
# without ever expanding the one-hot.
POSITION_FEATURE_NAMES = tuple(f"mm_type_pos_{i}" for i in range(1, GUIDE_LEN + 1))

_PAIR_INDEX = np.zeros(16, dtype=np.uint8)
_rank = 1
for _g in range(4):
    for _d in range(4):
        if _g != _d:
            _PAIR_INDEX[_g * 4 + _d] = _rank
            _rank += 1


def _max_run(flags: np.ndarray) -> np.ndarray:
    """Longest run of True along axis 1, one pass per column."""
    n, width = flags.shape
    best = np.zeros(n, dtype=np.int16)
    current = np.zeros(n, dtype=np.int16)
    for column in range(width):
        current = np.where(flags[:, column], current + 1, 0)
        best = np.maximum(best, current)
    return best


def feature_names(extended: bool = True) -> tuple[str, ...]:
    return FEATURE_NAMES + POSITION_FEATURE_NAMES if extended else FEATURE_NAMES


def categorical_mask(extended: bool = True) -> np.ndarray:
    """Which columns are categorical, for the gradient booster."""
    names = feature_names(extended)
    return np.array([n.startswith("mm_type_pos_") for n in names], dtype=bool)


def featurize(
    dataset: OffTargetDataset,
    model: MismatchModel,
    chunk: int = 400_000,
    extended: bool = True,
) -> np.ndarray:
    """(n, len(feature_names(extended))) float32 feature matrix.

    Built in chunks: the dataset is millions of rows, and materialising every
    intermediate at once costs more memory than the result does.
    """
    if not model.is_published:
        raise ValueError(
            "featurize needs the published CFD tables; run scripts/import_cfd.py"
        )
    retention = retention_lookup(model)
    pam_factors = pam_lookup(model)
    guide_codes = dataset.guide_codes()

    n = len(dataset)
    out = np.zeros((n, len(feature_names(extended))), dtype=np.float32)
    positions = np.arange(GUIDE_LEN)

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        proto = dataset.proto[start:stop]
        guide = guide_codes[dataset.guide_idx[start:stop]]

        # Clip code 4 (N, or any non-ACGT) into range so the lookups stay
        # valid. Such a position still counts as a mismatch below.
        proto_spacer = np.clip(proto[:, :GUIDE_LEN], 0, 3)
        guide_spacer = np.clip(guide[:, :GUIDE_LEN], 0, 3)
        mismatch = proto[:, :GUIDE_LEN] != guide[:, :GUIDE_LEN]

        ret = retention[positions[None, :], guide_spacer, proto_spacer]
        ret = np.where(mismatch, ret, 1.0)

        pam = np.clip(proto[:, GUIDE_LEN + 1 : GUIDE_LEN + 3], 0, 3)
        pam_value = pam_factors[pam[:, 0], pam[:, 1]]
        product = ret.prod(axis=1)
        cfd = product * pam_value

        proto_gc = np.isin(proto_spacer, (1, 2)).mean(axis=1)
        guide_gc = np.isin(guide_spacer, (1, 2)).mean(axis=1)

        is_transition = np.zeros_like(mismatch)
        is_wobble = np.zeros_like(mismatch)
        for g, d in _TRANSITION_PAIRS:
            is_transition |= (guide_spacer == g) & (proto_spacer == d)
        for g, d in _WOBBLE_PAIRS:
            is_wobble |= (guide_spacer == g) & (proto_spacer == d)
        is_transition &= mismatch
        is_wobble &= mismatch

        count = mismatch.sum(axis=1)
        any_mm = count > 0
        proximal_grid = np.where(mismatch, positions[None, :] + 1, 0)
        distal_grid = np.where(mismatch, positions[None, :] + 1, GUIDE_LEN + 1)
        most_proximal = proximal_grid.max(axis=1)
        most_distal = np.where(any_mm, distal_grid.min(axis=1), 0)
        mean_position = np.where(
            any_mm, proximal_grid.sum(axis=1) / np.maximum(count, 1), 0
        )

        columns = [
            count,
            cfd,
            np.log(cfd + 1e-12),
            pam_value,
            mismatch[:, 0:5].sum(axis=1),
            mismatch[:, 5:10].sum(axis=1),
            mismatch[:, 10:15].sum(axis=1),
            mismatch[:, 15:20].sum(axis=1),
            ret.min(axis=1),
            product,
            (1.0 - ret).sum(axis=1),
            most_proximal,
            most_distal,
            proto_gc,
            guide_gc,
            np.abs(proto_gc - guide_gc),
            is_transition.sum(axis=1),
            (mismatch & ~is_transition).sum(axis=1),
            is_wobble.sum(axis=1),
            pam[:, 0],
            pam[:, 1],
            mean_position,
            _max_run(mismatch),
            mismatch[:, 12:20].sum(axis=1),
        ]
        block = np.column_stack(columns).astype(np.float32)
        if extended:
            pair_codes = _PAIR_INDEX[guide_spacer * 4 + proto_spacer]
            pair_codes = np.where(mismatch, pair_codes, 0)
            block = np.hstack([block, pair_codes.astype(np.float32)])
        out[start:stop] = block

    return out


def cfd_column(features: np.ndarray) -> np.ndarray:
    """The CFD score column, for use as the baseline ranking."""
    return features[:, FEATURE_NAMES.index("cfd")]


# Describing the gap itself, so a model can learn what a bulge costs instead of
# inheriting the placeholder constant in scoring.BULGE_PENALTY. That constant
# exists so a bulge site is not ranked as a perfect match; it is fitted to
# nothing, and there is no published multiplicative bulge weight to use
# instead. Given labelled bulge sites, the effect is better learned.
BULGE_FEATURE_NAMES = (
    "n_bulges",
    "bulge_is_dna",
    "bulge_is_rna",
    "bulge_position",
    "bulge_distance_from_pam",
)

_BASE_CODE = {"A": 0, "C": 1, "G": 2, "T": 3}


def alignment_feature_names(extended: bool = True) -> tuple[str, ...]:
    return feature_names(extended) + BULGE_FEATURE_NAMES


def features_from_alignments(
    records: list[tuple[str, list[tuple[int, str, str]], str, str | None, int, int]],
    model: MismatchModel,
    extended: bool = True,
) -> np.ndarray:
    """Features for alignments that may contain a bulge.

    Each record is (guide spacer, mismatches, PAM, bulge kind, first bulge
    position, bulge count), where `mismatches` holds (guide position 1-20,
    guide base, observed base). A bulge kind of None means a plain mismatch
    alignment; "both" means the alignment carries gaps on each side.

    The array path in `featurize` cannot express this, because a bulge
    alignment is not 20 positions wide. On mismatch-only input the two paths
    must agree exactly, which the tests assert rather than assume.

    The `cfd` column here carries no bulge penalty: a constant penalty only
    shifts every bulge row by the same amount, and the bulge columns describe
    the gap directly so the model can price it from data.
    """
    if not model.is_published:
        raise ValueError(
            "features_from_alignments needs the published CFD tables; "
            "run scripts/import_cfd.py"
        )
    retention = retention_lookup(model)
    pam_factors = pam_lookup(model)
    names = alignment_feature_names(extended)
    out = np.zeros((len(records), len(names)), dtype=np.float32)

    for row, (
        guide,
        mismatches,
        pam,
        kind,
        bulge_position,
        n_bulges,
    ) in enumerate(records):
        positions = np.array([m[0] for m in mismatches], dtype=np.int64)
        values = np.array(
            [retention[p - 1, _BASE_CODE[g], _BASE_CODE[d]] for p, g, d in mismatches],
            dtype=np.float64,
        )
        count = positions.size
        product = float(values.prod()) if count else 1.0
        pam_value = float(pam_factors[_BASE_CODE[pam[1]], _BASE_CODE[pam[2]]])
        cfd = product * pam_value

        guide_gc_count = sum(1 for b in guide if b in "GC")
        observed_gc_count = guide_gc_count
        transitions = wobble = 0
        for _, g, d in mismatches:
            observed_gc_count += (d in "GC") - (g in "GC")
            pair = (_BASE_CODE[g], _BASE_CODE[d])
            transitions += pair in _TRANSITION_PAIRS
            wobble += pair in _WOBBLE_PAIRS
        guide_gc = guide_gc_count / GUIDE_LEN
        observed_gc = observed_gc_count / GUIDE_LEN

        if count:
            ordered = np.sort(positions)
            run = best = 1
            for i in range(1, ordered.size):
                run = run + 1 if ordered[i] == ordered[i - 1] + 1 else 1
                best = max(best, run)
        else:
            best = 0

        columns = [
            count,
            cfd,
            np.log(cfd + 1e-12),
            pam_value,
            int(((positions >= 1) & (positions <= 5)).sum()),
            int(((positions >= 6) & (positions <= 10)).sum()),
            int(((positions >= 11) & (positions <= 15)).sum()),
            int(((positions >= 16) & (positions <= 20)).sum()),
            float(values.min()) if count else 1.0,
            product,
            float((1.0 - values).sum()) if count else 0.0,
            int(positions.max()) if count else 0,
            int(positions.min()) if count else 0,
            observed_gc,
            guide_gc,
            abs(observed_gc - guide_gc),
            transitions,
            count - transitions,
            wobble,
            _BASE_CODE[pam[1]],
            _BASE_CODE[pam[2]],
            float(positions.mean()) if count else 0.0,
            best,
            int((positions >= 13).sum()),
        ]
        if extended:
            per_position = np.zeros(GUIDE_LEN, dtype=np.float32)
            for p, g, d in mismatches:
                per_position[p - 1] = _PAIR_INDEX[_BASE_CODE[g] * 4 + _BASE_CODE[d]]
            columns.extend(per_position.tolist())

        columns.extend(
            [
                n_bulges,
                1 if kind in ("dna", "both") else 0,
                1 if kind in ("rna", "both") else 0,
                bulge_position if kind else 0,
                (GUIDE_LEN - bulge_position) if kind else 0,
            ]
        )
        out[row] = np.asarray(columns, dtype=np.float32)

    return out
