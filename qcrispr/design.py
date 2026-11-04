"""Assembling guide candidates for a target window."""

from __future__ import annotations

from dataclasses import dataclass

from .offtarget import GenomeIndex, LocusRegistry, OffTarget, burden, locus_keys
from .scoring import MismatchModel, on_target_proxy
from .sequence import Site, find_sites


@dataclass
class GuideCandidate:
    site: Site
    efficiency: float
    offtargets: list[OffTarget]
    burden: float
    loci: frozenset[tuple[str, int]]

    @property
    def spacer(self) -> str:
        return self.site.spacer

    @property
    def cut(self) -> int:
        return self.site.cut

    @property
    def label(self) -> str:
        return f"{self.site.chrom}:{self.site.start}{self.site.strand}"

    @property
    def worst_offtarget(self) -> float:
        return max((o.cfd for o in self.offtargets), default=0.0)


def collect_candidates(
    index: GenomeIndex,
    chrom: str,
    start: int,
    end: int,
    max_mismatches: int = 4,
    min_cfd: float = 1e-4,
    model: MismatchModel | None = None,
    locus_window: int = 30,
    max_bulges: int | None = None,
) -> list[GuideCandidate]:
    """Every NGG guide in the target window, with its genome-wide off-targets.

    On-target candidates are restricted to canonical NGG PAMs because that is
    what one designs against; off-target search still considers NAG and NGA.
    """
    seq = index.records[chrom]
    on_sites = find_sites(seq, chrom, ("NGG",), start=start, end=end)

    # Two passes: enumerate every candidate's off-targets, then cluster all the
    # cut positions together so that locus ids are comparable across guides.
    enumerated: list[tuple[Site, list[OffTarget]]] = []
    for site in on_sites:
        offs = index.find_offtargets(
            site.spacer,
            max_mismatches=max_mismatches,
            min_cfd=min_cfd,
            exclude={site.key},
            model=model,
            max_bulges=max_bulges,
        )
        enumerated.append((site, offs))

    registry = LocusRegistry.from_cuts(
        (o.locus for _, offs in enumerated for o in offs), window=locus_window
    )

    return [
        GuideCandidate(
            site=site,
            efficiency=on_target_proxy(site.spacer),
            offtargets=offs,
            burden=burden(offs),
            loci=locus_keys(offs, registry),
        )
        for site, offs in enumerated
    ]


def prefilter(
    candidates: list[GuideCandidate],
    limit: int,
    w_offtarget: float = 1.0,
    w_efficiency: float = 1.0,
) -> list[GuideCandidate]:
    """Keep the `limit` best candidates by a linear score.

    This exists only because exact statevector QAOA cannot hold more than ~18
    qubits. It is a real limitation of the quantum route, not a design choice:
    the classical solvers never need it, and any guide dropped here is
    invisible to the quantum solver no matter how good the optimisation is.
    """
    if limit >= len(candidates):
        return list(candidates)
    scale = max((c.burden for c in candidates), default=0.0) or 1.0
    ranked = sorted(
        candidates,
        key=lambda c: w_offtarget * (c.burden / scale) - w_efficiency * c.efficiency,
    )
    return ranked[:limit]
