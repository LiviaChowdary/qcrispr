"""Sequence handling: FASTA IO, PAM-site scanning, and a deterministic demo genome.

Coordinates are always 0-based and always expressed on the forward strand, so
that distances between sites on opposite strands are directly comparable.
For a site on the minus strand the 20nt spacer still occupies forward interval
[start, start+20); its PAM sits immediately to the left, at [start-3, start).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GUIDE_LEN = 20
PAM_LEN = 3
BASES = "ACGT"
_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")

# A/C/G/T -> 0/1/2/3; anything else -> 4, which never compares equal to a real
# base, so an N in the genome always counts as a mismatch.
_CODE = np.full(256, 4, dtype=np.uint8)
for _i, _b in enumerate(BASES):
    _CODE[ord(_b)] = _i
    _CODE[ord(_b.lower())] = _i

_IUPAC = {
    "A": "A", "C": "C", "G": "G", "T": "T", "N": "ACGT",
    "R": "AG", "Y": "CT", "S": "CG", "W": "AT", "K": "GT", "M": "AC",
    "B": "CGT", "D": "AGT", "H": "ACT", "V": "ACG",
}


def revcomp(seq: str) -> str:
    """Reverse complement, preserving case and passing N through."""
    return seq.translate(_COMPLEMENT)[::-1]


def encode(seq: str) -> np.ndarray:
    """Encode a sequence as uint8 base codes (non-ACGT becomes 4)."""
    return _CODE[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]


def pam_matches(pam: str, pattern: str) -> bool:
    """Test a concrete PAM against an IUPAC pattern such as NGG."""
    if len(pam) != len(pattern):
        return False
    return all(b.upper() in _IUPAC[p.upper()] for b, p in zip(pam, pattern))


def read_fasta(path: str | Path) -> dict[str, str]:
    records: dict[str, str] = {}
    name: str | None = None
    chunks: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records[name] = "".join(chunks)
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line.upper())
    if name is not None:
        records[name] = "".join(chunks)
    return records


def write_fasta(path: str | Path, records: dict[str, str], width: int = 70) -> None:
    with open(path, "w") as fh:
        for name, seq in records.items():
            fh.write(f">{name}\n")
            for i in range(0, len(seq), width):
                fh.write(seq[i : i + width] + "\n")


@dataclass(frozen=True)
class Site:
    """A candidate Cas9 protospacer: a 20nt spacer plus its 3nt PAM."""

    chrom: str
    start: int
    strand: str
    spacer: str
    pam: str

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.chrom, self.start, self.strand)

    @property
    def cut(self) -> int:
        """Approximate blunt-cut position, 3bp from the PAM-proximal end."""
        return self.start + 17 if self.strand == "+" else self.start + 3

    def __str__(self) -> str:
        return f"{self.chrom}:{self.start}{self.strand} {self.spacer}|{self.pam}"


def find_sites(
    seq: str,
    chrom: str = "chr",
    patterns: tuple[str, ...] = ("NGG",),
    guide_len: int = GUIDE_LEN,
    start: int = 0,
    end: int | None = None,
    both_strands: bool = True,
) -> list[Site]:
    """Scan for protospacers whose PAM matches any of `patterns`.

    `start`/`end` restrict the forward-strand interval the spacer must fall in.
    Sites containing a non-ACGT base are skipped.
    """
    length = len(seq)
    end = length if end is None else end
    window = guide_len + PAM_LEN
    out: list[Site] = []

    strands: list[tuple[str, str]] = [("+", seq)]
    if both_strands:
        strands.append(("-", revcomp(seq)))

    for strand, s in strands:
        for i in range(len(s) - window + 1):
            spacer = s[i : i + guide_len]
            pam = s[i + guide_len : i + window]
            fwd_start = i if strand == "+" else length - (i + guide_len)
            if fwd_start < start or fwd_start + guide_len > end:
                continue
            if not any(pam_matches(pam, p) for p in patterns):
                continue
            if any(b not in BASES for b in spacer) or any(b not in BASES for b in pam):
                continue
            out.append(Site(chrom, fwd_start, strand, spacer, pam))
    out.sort(key=lambda s: (s.start, s.strand))
    return out


BLOCK_LEN = GUIDE_LEN + PAM_LEN


def make_demo_genome(
    length: int = 120_000,
    seed: int = 7,
    target_start: int = 50_000,
    target_len: int = 300,
    offtargets_per_guide: int = 3,
    n_hotspots: int = 6,
    hotspot_size: int = 3,
    bulge_guides: int = 12,
    slot_step: int = 300,
) -> tuple[str, list[dict]]:
    """Build a deterministic synthetic genome with planted off-target sites.

    Random background already contains NGG PAMs about once every 16 positions
    per strand, so the target window needs no special construction. What random
    background does not contain is near-matches to a given spacer, so 1-4
    mismatch off-targets are written in for *every* guide in the window.

    Seeding every guide matters. In a 120kb random background almost no guide
    has a natural near-match, so a sparsely seeded genome produces a candidate
    pool where most guides have zero predicted burden -- an instance with no
    structure to optimise, and an artifact of the toy genome size rather than
    anything biological. In a real mammalian genome essentially every guide has
    off-targets.

    `n_hotspots` groups pack off-targets for several *different* guides 23bp
    apart, so their cut sites cluster into one locus. That is what a repeat
    family or a paralogous region looks like to a guide designer, and it is
    what creates the pairwise collisions the QUBO's quadratic term scores.

    The returned list is ground truth for testing the enumerator.
    """
    rng = random.Random(seed)
    genome = [rng.choice(BASES) for _ in range(length)]

    target_end = target_start + target_len
    window_seq = "".join(genome[target_start:target_end])
    spacers = sorted({s.spacer for s in find_sites(window_seq, "demo_chr", ("NGG",))})
    rng.shuffle(spacers)

    # Planted sites stay clear of the target window and of each other.
    slots = [
        s
        for s in range(5_000, length - 5_000, slot_step)
        if not (target_start - 2_000 <= s <= target_end + 2_000)
    ]
    rng.shuffle(slots)
    slot_iter = iter(slots)

    planted: list[dict] = []

    def plant(spacer: str, pos: int, bulge: str = "none") -> None:
        # Bulge sites get fewer mismatches so they stay inside a realistic
        # search budget; a bulge plus four mismatches is beyond what anyone
        # searches for.
        n_mm = rng.randint(1, 4) if bulge == "none" else rng.randint(0, 2)
        mutated = list(spacer)
        for p in rng.sample(range(GUIDE_LEN), n_mm):
            mutated[p] = rng.choice([b for b in BASES if b != spacer[p]])

        if bulge == "dna":
            # An extra base in the genome: the guide aligns to 21nt.
            mutated.insert(rng.randrange(1, GUIDE_LEN), rng.choice(BASES))
        elif bulge == "rna":
            # A missing base in the genome: the guide aligns to 19nt.
            del mutated[rng.randrange(1, GUIDE_LEN - 1)]

        protospacer = "".join(mutated)
        pam = rng.choice(["AGG", "CGG", "TGG", "GGG", "AAG", "TAG"])
        block = protospacer + pam
        genome[pos : pos + len(block)] = list(block)
        pam_start = pos + len(protospacer)
        planted.append(
            {
                "guide": spacer,
                "chrom": "demo_chr",
                # Where the block was written, and where a 20nt protospacer
                # ending at this PAM starts. They differ for bulge sites, and
                # the enumerator reports the latter.
                "start": pos,
                "site_start": pam_start - GUIDE_LEN,
                "strand": "+",
                "protospacer": protospacer,
                "pam": pam,
                "mismatches": n_mm,
                "bulge": bulge,
            }
        )

    for spacer in spacers:
        for _ in range(offtargets_per_guide):
            pos = next(slot_iter, None)
            if pos is None:
                break
            plant(spacer, pos)

    for _ in range(n_hotspots):
        pos = next(slot_iter, None)
        if pos is None:
            break
        group = rng.sample(spacers, min(hotspot_size, len(spacers)))
        for idx, spacer in enumerate(group):
            plant(spacer, pos + idx * BLOCK_LEN)

    # Bulge-containing sites, as ground truth for whether the search can find
    # them at all. A mismatch-only enumerator cannot, at any threshold.
    for spacer in spacers[:bulge_guides]:
        for kind in ("dna", "rna"):
            pos = next(slot_iter, None)
            if pos is None:
                break
            plant(spacer, pos, kind)

    return "".join(genome), planted
