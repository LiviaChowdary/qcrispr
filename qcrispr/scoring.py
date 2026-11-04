"""Off-target and on-target scoring.

Read this before trusting a number out of this module.

The off-target model has the same functional form as the Cutting Frequency
Determination (CFD) score of Doench et al. 2016 -- a product of per-mismatch
retained-activity terms multiplied by a PAM factor -- but the weights built
into this file are a smooth parameterisation written here, not the published
supplementary table. They encode three facts that are well established in the
literature and nothing finer grained than that:

  1. mismatches close to the PAM cost far more activity than distal ones,
  2. some mismatch identities are tolerated better than others, with
     rG:dT and rU:dG the most forgiving,
  3. NGG is the strongly preferred PAM; NAG and NGA retain partial activity.

### How bad is the parameterisation? Measured, not guessed.

Against the published 240-cell table, the parameterisation below scores:

    mean absolute error   0.219
    max  absolute error   0.828
    Pearson correlation   0.485

That is not a usable substitute, and the reason is instructive. The published
weights are far more dependent on mismatch *identity* than on position: real
CFD gives retained activity 1.000 for guide C against an observed A at
position 16, and 0.938 for G against A at position 20, both PAM-proximal,
where the smooth positional curve below predicts 0.27 and 0.11. The familiar
summary "mismatches near the PAM are more disruptive" is true on average and
badly wrong cell by cell.

So: run `scripts/import_cfd.py` and use the real tables. The parameterisation
exists so the pipeline runs with no data files present, and `is_published`
reports which one is active. When a real table is loaded, a missing cell
raises rather than silently falling back.

Absolute scores are unitless and uncalibrated against measured cleavage. Only
differences between scores computed by the same model are meaningful.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

# Relative cleavage retained by non-canonical PAMs. Order-of-magnitude values
# consistent with reported alternative-PAM activity, not fitted constants.
PAM_FACTORS: dict[str, float] = {
    "AGG": 1.0, "CGG": 1.0, "GGG": 1.0, "TGG": 1.0,
    "AAG": 0.259, "CAG": 0.259, "GAG": 0.259, "TAG": 0.259,
    "AGA": 0.069, "CGA": 0.069, "GGA": 0.069, "TGA": 0.069,
    "ACG": 0.022, "CCG": 0.022, "GCG": 0.022, "TCG": 0.022,
    "AGC": 0.017, "CGC": 0.017, "GGC": 0.017, "TGC": 0.017,
    "AGT": 0.016, "CGT": 0.016, "GGT": 0.016, "TGT": 0.016,
}
DEFAULT_PAM_FACTOR = 0.004

# Multipliers on the positional retention, by (guide base, DNA base). The guide
# spacer is written in DNA letters, so T stands for U.
_PAIR_MODIFIER: dict[tuple[str, str], float] = {
    ("G", "T"): 1.35, ("T", "G"): 1.35,            # rG:dT, rU:dG wobble-like
    ("A", "G"): 1.10, ("G", "A"): 1.10,            # purine-purine
    ("C", "T"): 1.10, ("T", "C"): 1.10,            # pyrimidine-pyrimidine
}
_TRANSVERSION_MODIFIER = 0.80
_MAX_RETENTION = 0.98


def _positional_retention(position: int, guide_len: int = 20) -> float:
    """Activity retained by a mismatch at `position` (1-based, 5'->3').

    Position `guide_len` is PAM-proximal and most disruptive.
    """
    frac = (position - 1) / (guide_len - 1)
    return 0.95 - 0.85 * (frac**1.4)


@dataclass
class MismatchModel:
    """Per-(position, guide base, DNA base) retained activity, plus PAM factors.

    An empty `table` means every lookup falls through to the parameterisation
    above. A populated one means the published weights are in use, and
    `is_published` reports which.

    `dna_base` is always the base OBSERVED in the genomic protospacer. This
    matters: a CFD key names the complementary base, because it refers to the
    opposite strand. `scripts/import_cfd.py` complements on import so that
    everything here works in the observed-base convention.
    """

    guide_len: int = 20
    table: dict[tuple[int, str, str], float] = field(default_factory=dict)
    pam: dict[str, float] = field(default_factory=dict)
    source: str = "parameterised (not the published CFD table)"

    @property
    def is_published(self) -> bool:
        return bool(self.table)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        pam_path: str | Path | None = None,
        guide_len: int = 20,
    ) -> "MismatchModel":
        table: dict[tuple[int, str, str], float] = {}
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.lower().startswith("position"):
                    continue
                pos, rna, dna, value = line.split()[:4]
                key = (int(pos), rna.upper().replace("U", "T"), dna.upper())
                table[key] = float(value)

        pam_table: dict[str, float] = {}
        if pam_path is not None:
            with open(pam_path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.lower().startswith("pam"):
                        continue
                    key, value = line.split()[:2]
                    pam_table[key.upper()] = float(value)

        return cls(
            guide_len=guide_len, table=table, pam=pam_table, source=str(path)
        )

    def retention(self, position: int, guide_base: str, dna_base: str) -> float:
        """Fraction of activity retained by one mismatch."""
        key = (position, guide_base, dna_base)
        if key in self.table:
            return self.table[key]
        if self.table:
            raise KeyError(
                f"no published weight for guide {guide_base} vs observed "
                f"{dna_base} at position {position}; refusing to silently fall "
                f"back to the parameterisation while a real table is loaded"
            )
        base = _positional_retention(position, self.guide_len)
        modifier = _PAIR_MODIFIER.get((guide_base, dna_base), _TRANSVERSION_MODIFIER)
        return min(_MAX_RETENTION, base * modifier)

    def pam_factor(self, pam: str) -> float:
        """Relative cleavage retained by this PAM.

        A loaded CFD table is keyed by PAM positions 2-3 only, so the first
        base is ignored, which is what the reference implementation does.
        """
        if self.pam:
            return self.pam.get(pam[-2:].upper(), 0.0)
        return PAM_FACTORS.get(pam.upper(), DEFAULT_PAM_FACTOR)


def load_cfd_model(
    mismatch_path: str | Path, pam_path: str | Path, guide_len: int = 20
) -> MismatchModel:
    """Load the published CFD tables produced by scripts/import_cfd.py."""
    return MismatchModel.from_file(mismatch_path, pam_path, guide_len=guide_len)


_DEFAULT_MODEL = MismatchModel()


def pam_factor(pam: str) -> float:
    return PAM_FACTORS.get(pam.upper(), DEFAULT_PAM_FACTOR)


def cfd_score(
    spacer: str,
    protospacer: str,
    pam: str,
    model: MismatchModel | None = None,
) -> float:
    """CFD-style score in (0, 1] for a guide against one genomic protospacer.

    1.0 means a perfect match with a canonical PAM. Raises on length mismatch
    rather than silently truncating.
    """
    if len(spacer) != len(protospacer):
        raise ValueError(
            f"spacer length {len(spacer)} != protospacer length {len(protospacer)}"
        )
    model = model or _DEFAULT_MODEL
    score = model.pam_factor(pam)
    for idx, (g, d) in enumerate(zip(spacer.upper(), protospacer.upper()), start=1):
        if g != d:
            score *= model.retention(idx, g, d)
    return score


# CFD has no bulge term: it was fit on mismatch data only, and there is no
# published multiplicative weight for an insertion or deletion. This constant
# is a placeholder so that bulge sites carry *some* cost and are not ranked as
# if they were perfect matches. It is not fitted to anything.
#
# Treat any score with n_bulges > 0 as "flagged for review", not as a
# calibrated activity estimate. The literature backs this up: bulge-capable
# predictors reach only ~0.045 AUPR on bulge-only subsets, so nobody can score
# these well yet. What matters is that the site is *found*, since a site that
# is never enumerated cannot be reviewed at all.
BULGE_PENALTY = 0.3


def cfd_score_alignment(
    mismatches: list[tuple[int, str, str]],
    pam: str,
    model: MismatchModel | None = None,
    n_bulges: int = 0,
    bulge_penalty: float = BULGE_PENALTY,
) -> float:
    """Score an alignment given its mismatched positions.

    `mismatches` holds (guide position 1-20, guide base, observed base) for
    mismatched positions only. This is the general form behind `cfd_score`,
    and it is what bulge-containing alignments need, since their aligned
    length is not 20.
    """
    model = model or _DEFAULT_MODEL
    score = model.pam_factor(pam)
    for position, guide_base, observed_base in mismatches:
        score *= model.retention(position, guide_base, observed_base)
    if n_bulges:
        score *= bulge_penalty**n_bulges
    return score


def count_mismatches(spacer: str, protospacer: str) -> int:
    return sum(1 for g, d in zip(spacer.upper(), protospacer.upper()) if g != d)


def on_target_proxy(spacer: str) -> float:
    """Crude on-target efficiency proxy in [0, 1].

    This is NOT Rule Set 2 or any trained model. It captures only three coarse
    design rules: mid-range GC is best, a poly-T run terminates U6
    transcription, and a PAM-proximal G or C helps slightly. Replace it with a
    trained predictor before drawing any biological conclusion.
    """
    s = spacer.upper()
    if not s:
        return 0.0
    gc = sum(1 for b in s if b in "GC") / len(s)
    score = math.exp(-(((gc - 0.5) / 0.22) ** 2))
    if "TTTT" in s:
        score *= 0.4
    if s[-1] in "GC":
        score *= 1.1
    return max(0.0, min(1.0, score))
