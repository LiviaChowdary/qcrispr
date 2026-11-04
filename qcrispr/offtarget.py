"""Genome-wide off-target enumeration.

All candidate protospacers in the genome are encoded once into a single uint8
matrix. Finding the off-targets of a guide is then one vectorised row-wise
comparison, so scoring many guides against the same genome is cheap.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from .scoring import MismatchModel, cfd_score_alignment
from .sequence import GUIDE_LEN, Site, encode, find_sites, revcomp

# PAMs searched when looking for off-targets. NGG alone misses the partially
# active alternative PAMs, which is where real off-target cleavage hides.
OFFTARGET_PAMS = ("NGG", "NAG", "NGA")


@dataclass(frozen=True)
class OffTarget:
    site: Site
    mismatches: int
    cfd: float
    bulges: int = 0
    alignment: str = ""

    @property
    def locus(self) -> tuple[str, int]:
        return (self.site.chrom, self.site.cut)

    @property
    def has_bulge(self) -> bool:
        return self.bulges > 0


_BASES = "ACGTN"


class GenomeIndex:
    """All PAM-adjacent protospacers in a genome, encoded for fast comparison.

    With `max_bulges >= 1` the index also stores the 19nt and 21nt windows
    immediately 5' of each PAM, which is what makes bulge-containing
    alignments findable at all:

      * a 21nt window aligned to a 20nt guide means an extra base in the DNA
        (a DNA bulge),
      * a 19nt window means an extra base in the guide (an RNA bulge).

    This matters more than it sounds. Roughly a fifth of experimentally
    measured off-target sites involve a bulge, and a mismatch-only search
    cannot represent them at any mismatch threshold -- they are not
    low-scoring, they are absent. In the CHOPOFF comparison the two most
    widely used enumerators each missed thousands of bulge sites at edit
    distance 2-3 for exactly this reason.
    """

    def __init__(
        self,
        records: dict[str, str],
        patterns: tuple[str, ...] = OFFTARGET_PAMS,
        guide_len: int = GUIDE_LEN,
        max_bulges: int = 1,
    ) -> None:
        self.guide_len = guide_len
        self.patterns = patterns
        self.records = records
        self.max_bulges = max_bulges

        sites: list[Site] = []
        strands: dict[tuple[str, str], str] = {}
        for chrom, seq in records.items():
            sites.extend(find_sites(seq, chrom, patterns, guide_len=guide_len))
            strands[(chrom, "+")] = seq
            strands[(chrom, "-")] = revcomp(seq)
        self.sites = sites

        lengths = [guide_len]
        if max_bulges >= 1:
            lengths = [guide_len - 1, guide_len, guide_len + 1]
        self.lengths = tuple(lengths)

        # windows[L][i] is the L-nt window ending at site i's PAM, encoded.
        # Code 4 never matches a real base, so a window that would run off the
        # end of the sequence is padded and can never produce a spurious hit.
        self.windows: dict[int, np.ndarray] = {
            L: np.full((len(sites), L), 4, dtype=np.uint8) for L in lengths
        }
        for i, site in enumerate(sites):
            strand_seq = strands[(site.chrom, site.strand)]
            if site.strand == "+":
                offset = site.start
            else:
                offset = len(strand_seq) - (site.start + guide_len)
            pam_start = offset + guide_len
            for L in lengths:
                begin = pam_start - L
                if begin < 0:
                    continue
                window = strand_seq[begin:pam_start]
                if len(window) == L:
                    self.windows[L][i] = encode(window)

        self.codes = self.windows[guide_len]

    def __len__(self) -> int:
        return len(self.sites)

    def _align(
        self, query: np.ndarray, length: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Best alignment of `query` against every window of `length`.

        Returns (mismatch count per site, skipped index per site). With at
        most one bulge the alignment is exact and needs no dynamic
        programming: the only freedom is which single base to skip, so every
        choice is tried and the best kept.
        """
        windows = self.windows[length]
        if length == self.guide_len:
            return (windows != query[None, :]).sum(axis=1), np.full(len(windows), -1)
        if length == self.guide_len + 1:
            # DNA bulge: drop one genomic base, then compare straight across.
            counts = np.stack(
                [
                    (np.delete(windows, p, axis=1) != query[None, :]).sum(axis=1)
                    for p in range(length)
                ]
            )
        else:
            # RNA bulge: drop one guide base instead.
            counts = np.stack(
                [
                    (windows != np.delete(query, p)[None, :]).sum(axis=1)
                    for p in range(self.guide_len)
                ]
            )
        return counts.min(axis=0), counts.argmin(axis=0)

    def _mismatch_triples(
        self, spacer: str, idx: int, length: int, skip: int
    ) -> list[tuple[int, str, str]]:
        """(guide position, guide base, observed base) for each mismatch."""
        window = "".join(_BASES[c] for c in self.windows[length][idx])
        triples: list[tuple[int, str, str]] = []
        if length == self.guide_len:
            for i, (g, d) in enumerate(zip(spacer, window), start=1):
                if g != d:
                    triples.append((i, g, d))
        elif length == self.guide_len + 1:
            observed = window[:skip] + window[skip + 1 :]
            for i, (g, d) in enumerate(zip(spacer, observed), start=1):
                if g != d:
                    triples.append((i, g, d))
        else:
            guide = spacer[:skip] + spacer[skip + 1 :]
            for j, (g, d) in enumerate(zip(guide, window)):
                if g != d:
                    # Map back to the original guide numbering, which is what
                    # the CFD position weights are indexed by.
                    triples.append((j + 1 if j < skip else j + 2, g, d))
        return triples

    def find_offtargets(
        self,
        spacer: str,
        max_mismatches: int = 4,
        min_cfd: float = 0.0,
        exclude: set[tuple[str, int, str]] | None = None,
        model: MismatchModel | None = None,
        max_bulges: int | None = None,
    ) -> list[OffTarget]:
        """Every indexed site within the mismatch and bulge limits.

        `exclude` drops sites by key, which is how a guide's own intended
        on-target site is kept out of its off-target burden. Where a site
        admits several alignments, the one with fewest bulges wins, then
        fewest mismatches: an exact reading is always preferred to a
        bulge-containing interpretation of the same locus.
        """
        if len(spacer) != self.guide_len:
            raise ValueError(f"spacer must be {self.guide_len}nt, got {len(spacer)}")
        if not self.sites:
            return []
        budget = self.max_bulges if max_bulges is None else min(max_bulges, self.max_bulges)

        query = encode(spacer)
        best: dict[int, tuple[int, int, int, int]] = {}
        for length in self.lengths:
            n_bulges = abs(length - self.guide_len)
            if n_bulges > budget:
                continue
            counts, skips = self._align(query, length)
            for idx in np.flatnonzero(counts <= max_mismatches):
                idx = int(idx)
                candidate = (n_bulges, int(counts[idx]), length, int(skips[idx]))
                current = best.get(idx)
                if current is None or candidate[:2] < current[:2]:
                    best[idx] = candidate

        exclude = exclude or set()
        out: list[OffTarget] = []
        for idx, (n_bulges, mismatches, length, skip) in best.items():
            site = self.sites[idx]
            if site.key in exclude:
                continue
            triples = self._mismatch_triples(spacer, idx, length, skip)
            score = cfd_score_alignment(
                triples, site.pam, model=model, n_bulges=n_bulges
            )
            if score < min_cfd:
                continue
            if n_bulges == 0:
                alignment = "exact"
            elif length == self.guide_len + 1:
                alignment = f"DNA bulge @ protospacer {skip + 1}"
            else:
                alignment = f"RNA bulge @ guide {skip + 1}"
            out.append(OffTarget(site, mismatches, score, n_bulges, alignment))
        out.sort(key=lambda o: -o.cfd)
        return out


def burden(
    offtargets: list[OffTarget],
    locus_weights: dict[tuple[str, int], float] | None = None,
) -> float:
    """Summed CFD over off-targets: the predicted off-target load of a guide.

    Summing is a deliberate choice. The maximum single off-target score answers
    "what is the worst site", the sum answers "how much total mis-cutting", and
    the sum is what a guide-selection objective should minimise.
    """
    if locus_weights is None:
        return float(sum(o.cfd for o in offtargets))
    return float(sum(o.cfd * locus_weights.get(o.locus, 1.0) for o in offtargets))


@dataclass
class LocusRegistry:
    """Global identity for off-target cut sites.

    Two guides whose off-targets land in the same place risk cutting that place
    twice, which is what the quadratic term in the QUBO penalises. Detecting
    that requires locus identity to be *global*: if guide A cuts at 117 and
    guide B at 121, both must resolve to the same locus id or the collision is
    invisible.

    Bucketing by integer division cannot do this. It splits any pair straddling
    a boundary (117//20 = 5 but 121//20 = 6), and bucketing each guide's list
    separately would let the same position get different ids. So all cut
    positions are clustered once, up front, by single linkage at `window` bp,
    and every guide then looks its off-targets up in the same registry.
    """

    window: int = 20
    bounds: dict[str, list[int]] = field(default_factory=dict)

    @classmethod
    def from_cuts(
        cls, cuts: Iterable[tuple[str, int]], window: int = 20
    ) -> "LocusRegistry":
        by_chrom: dict[str, set[int]] = {}
        for chrom, cut in cuts:
            by_chrom.setdefault(chrom, set()).add(cut)

        bounds: dict[str, list[int]] = {}
        for chrom, positions in by_chrom.items():
            ordered = sorted(positions)
            starts = [ordered[0]]
            for prev, cur in zip(ordered, ordered[1:]):
                if cur - prev > window:
                    starts.append(cur)
            bounds[chrom] = starts
        return cls(window=window, bounds=bounds)

    def key(self, chrom: str, cut: int) -> tuple[str, int]:
        """Locus id for one cut position: the start of its cluster.

        A position the registry never saw becomes its own locus, so an
        unregistered cut is never silently merged into someone else's cluster.
        """
        starts = self.bounds.get(chrom)
        if not starts:
            return (chrom, cut)
        idx = bisect.bisect_right(starts, cut) - 1
        if idx < 0:
            return (chrom, cut)
        return (chrom, starts[idx])


def locus_keys(
    offtargets: list[OffTarget], registry: LocusRegistry
) -> frozenset[tuple[str, int]]:
    """The set of loci a guide's off-targets occupy, as registry ids."""
    return frozenset(registry.key(chrom, cut) for chrom, cut in (o.locus for o in offtargets))
