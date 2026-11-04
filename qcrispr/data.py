"""Loading the measured CHANGE-seq off-target dataset.

Two files are needed, both from OrensteinLab/SysEvalOffTarget:

  output_file_pam_change.txt   3,567,646 candidate sites enumerated at up to
                               6 mismatches, with no activity information
  CHANGE-seq.xlsx              202,043 measured sites with read counts

A candidate that appears in the measured table gets its read count; every
other candidate is a measured zero. That asymmetry is the whole difficulty of
this dataset: about 2% of candidates were ever detected, and the rest are
negatives by absence rather than by observation.

### The 5% that cannot be represented

Joining on (target, chrom, chromStart, strand), 191,841 of the 202,043
measured sites are found among the candidates. The missing 10,202 are not a
join bug. The candidate file was enumerated with mismatches only, so a
measured site whose alignment needs an insertion or deletion has no candidate
row to attach to, at any mismatch threshold. They are dropped here because
there is nothing to score them against, and the loader reports the count
rather than hiding it. This is the same blind spot `offtarget.py` fixes for
the search path, measured on real data instead of planted sites.

### Activity threshold

Published work on this dataset calls a site active at read count > 100 and
treats lower counts as noise. That choice moves every downstream number, so
it is a parameter here and is reported with the results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .sequence import encode

SITE_LEN = 23  # 20nt protospacer plus 3nt PAM
ACTIVE_READ_THRESHOLD = 100.0
EXPECTED_CANDIDATES = 3_567_646


@dataclass
class OffTargetDataset:
    """Candidate sites with measured read counts, grouped by guide."""

    guides: np.ndarray       # (n_guides,) 23nt guide+PAM strings
    guide_idx: np.ndarray    # (n,) index into guides
    proto: np.ndarray        # (n, 23) encoded protospacer
    reads: np.ndarray        # (n,) measured reads, 0 where undetected
    distance: np.ndarray     # (n,) mismatch count as enumerated
    dropped_unmatched: int = 0

    def __len__(self) -> int:
        return int(self.reads.size)

    @property
    def n_guides(self) -> int:
        return int(self.guides.size)

    def labels(self, threshold: float = ACTIVE_READ_THRESHOLD) -> np.ndarray:
        return (self.reads > threshold).astype(np.int8)

    def guide_codes(self) -> np.ndarray:
        """(n_guides, 23) encoded guide sequences."""
        return np.vstack([encode(str(g)) for g in self.guides])

    def summary(self, threshold: float = ACTIVE_READ_THRESHOLD) -> str:
        y = self.labels(threshold)
        active = int(y.sum())
        inactive = len(self) - active
        return (
            f"{len(self):,} candidate sites across {self.n_guides} guides; "
            f"{active:,} active at reads>{threshold:g} "
            f"({active / len(self):.3%}), imbalance 1:{inactive / max(1, active):.0f}"
        )


@dataclass(frozen=True)
class BulgeSite:
    """A measured off-target whose alignment to the guide needs a gap.

    The measured table encodes the alignment directly, which is lucky: a 23nt
    sequence containing '-' marks positions where the DNA has no base, so a
    guide base aligns to nothing (an RNA bulge); a 24nt sequence carries one
    extra DNA base (a DNA bulge). Nothing has to be re-aligned or guessed for
    the first kind, and only the skip position has to be chosen for the second.
    """

    guide: str
    sequence: str
    reads: float
    distance: int
    kind: str  # "rna" or "dna"

    @property
    def n_bulges(self) -> int:
        return self.sequence.count("-") if self.kind == "rna" else 1


def classify_bulge(guide: str, sequence: str) -> str | None:
    """"rna", "dna", or None when the row is not a scoreable bulge."""
    if any(c not in "ACGT-" for c in sequence.upper()):
        return None
    if "-" in sequence:
        return "rna" if len(sequence) == len(guide) else None
    if len(sequence) == len(guide) + 1:
        return "dna"
    return None


def load_bulge_sites(xlsx_path: str | Path) -> tuple[list[BulgeSite], int]:
    """Measured sites whose alignment requires a bulge, plus a malformed count.

    These are the sites that mismatch-only enumeration cannot produce a
    candidate row for, so every model trained on the standard candidate set is
    blind to them. They are a fifth of measured off-targets in the literature
    and about 5% here.
    """
    import openpyxl

    workbook = openpyxl.load_workbook(xlsx_path, read_only=True)
    sheet = workbook[workbook.sheetnames[0]]
    rows = sheet.iter_rows(values_only=True)
    header = list(next(rows))
    index = {name: i for i, name in enumerate(header) if name}
    read_column = next(c for c in index if isinstance(c, str) and c.endswith("_reads"))

    sites: list[BulgeSite] = []
    malformed = 0
    for row in rows:
        target = row[index["target"]]
        sequence = row[index["offtarget_sequence"]]
        if target is None or sequence is None:
            continue
        guide = str(target)
        sequence = str(sequence).upper()
        if len(sequence) == len(guide) and "-" not in sequence:
            continue  # a plain mismatch site, handled by the main dataset
        kind = classify_bulge(guide, sequence)
        if kind is None:
            malformed += 1
            continue
        sites.append(
            BulgeSite(
                guide=guide,
                sequence=sequence,
                reads=float(row[index[read_column]] or 0.0),
                distance=int(row[index["distance"]] or 0),
                kind=kind,
            )
        )
    workbook.close()
    return sites, malformed


def parse_alignment(
    align_guide: str, align_offtarget: str
) -> tuple[str, list[tuple[int, str, str]], str, str | None, int, int] | None:
    """Turn a gapped alignment pair into an explicit alignment record.

    Both strings are the same length and already aligned, with '-' marking a
    position where one side has no base. A gap in the guide means an extra DNA
    base (a DNA bulge); a gap in the off-target means the DNA is missing a base
    the guide has (an RNA bulge).

    Returns (guide spacer, mismatches, PAM, bulge kind, first bulge position,
    bulge count), or None when the row cannot be read. Guide positions are
    1-based over the ungapped 20nt spacer, which is what the CFD weights are
    indexed by, so gaps must be skipped when counting positions.
    """
    if len(align_guide) != len(align_offtarget) or len(align_guide) < 6:
        return None
    guide_body, guide_pam = align_guide[:-3], align_guide[-3:]
    other_body, pam = align_offtarget[:-3], align_offtarget[-3:]
    if len(pam) != 3 or any(c not in "ACGT" for c in pam):
        return None
    if "-" in guide_pam or "-" in pam:
        return None

    spacer: list[str] = []
    mismatches: list[tuple[int, str, str]] = []
    n_dna = n_rna = 0
    first_bulge = 0
    position = 0

    for g, d in zip(guide_body, other_body):
        if g == "-":
            # Extra base in the DNA: no guide position is consumed.
            n_dna += 1
            first_bulge = first_bulge or max(position, 1)
            continue
        position += 1
        spacer.append(g)
        if d == "-":
            n_rna += 1
            first_bulge = first_bulge or position
            continue
        if g != d:
            if g not in "ACGT" or d not in "ACGT":
                return None
            mismatches.append((position, g, d))

    if position != 20 or len(spacer) != 20:
        return None
    n_bulges = n_dna + n_rna
    if n_bulges == 0:
        kind = None
    elif n_dna and n_rna:
        kind = "both"
    elif n_dna:
        kind = "dna"
    else:
        kind = "rna"
    return "".join(spacer), mismatches, pam, kind, first_bulge, n_bulges


def guides_in_csv(path: str | Path, column: str = "sgRNA") -> set[str]:
    """The distinct guides a gapped-alignment CSV covers."""
    import csv

    with open(path, newline="") as handle:
        return {row[column] for row in csv.DictReader(handle)}


def load_aligned_csv(
    path: str | Path,
    label_column: str,
    threshold: float | None = None,
    negative_rate: float = 1.0,
    seed: int = 0,
    max_rows: int | None = None,
    exclude_guides: set[str] | None = None,
) -> tuple[list, np.ndarray, np.ndarray, np.ndarray, int]:
    """Read a gapped-alignment CSV into alignment records plus labels.

    `threshold` converts a read count into a binary label; pass None when the
    file already carries 0/1 labels.

    `negative_rate` keeps that fraction of negatives, which is how a 4.9M-row
    training file is made tractable. Every positive is always kept. This
    changes the class balance the model trains on but not the ranking it is
    scored on, since the evaluation set is never subsampled. Report it anyway.

    `exclude_guides` drops every row for those guides. This is not an
    optimisation, it is the difference between a held-out result and a leaked
    one: six of the ten guides in refined TrueOT also appear in the CHANGE-seq
    training file, so training on all of it and testing on TrueOT scores a
    model partly on guides it has already seen.

    Returns (records, labels, bulge counts, guide ids, skipped rows).
    """
    import csv

    rng = np.random.default_rng(seed)
    records: list = []
    labels: list[int] = []
    bulges: list[int] = []
    guide_ids: list[int] = []
    guide_index: dict[str, int] = {}
    skipped = 0

    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if max_rows is not None and len(records) >= max_rows:
                break
            if exclude_guides and row["sgRNA"] in exclude_guides:
                continue
            raw = row.get(label_column)
            if raw is None or raw == "":
                skipped += 1
                continue
            value = float(raw)
            label = int(value > threshold) if threshold is not None else int(value)
            if label == 0 and negative_rate < 1.0 and rng.random() >= negative_rate:
                continue

            parsed = parse_alignment(row["Align.sgRNA"], row["Align.off-target"])
            if parsed is None:
                skipped += 1
                continue
            spacer, mismatches, pam, kind, first_bulge, n_bulges = parsed

            guide = row["sgRNA"]
            if guide not in guide_index:
                guide_index[guide] = len(guide_index)
            records.append((spacer, mismatches, pam, kind, first_bulge, n_bulges))
            labels.append(label)
            bulges.append(n_bulges)
            guide_ids.append(guide_index[guide])

    return (
        records,
        np.asarray(labels, dtype=np.int8),
        np.asarray(bulges, dtype=np.int16),
        np.asarray(guide_ids, dtype=np.int32),
        skipped,
    )


def _read_measured(xlsx_path: str | Path) -> dict[tuple[str, str, int, str], float]:
    """(target, chrom, chromStart, strand) -> read count."""
    import openpyxl

    workbook = openpyxl.load_workbook(xlsx_path, read_only=True)
    sheet = workbook[workbook.sheetnames[0]]
    rows = sheet.iter_rows(values_only=True)
    header = list(next(rows))
    index = {name: i for i, name in enumerate(header) if name}

    required = {"target", "chrom", "chromStart", "strand"}
    missing = required - set(index)
    if missing:
        raise ValueError(f"{xlsx_path}: missing columns {sorted(missing)}")
    read_column = next(
        (c for c in index if isinstance(c, str) and c.endswith("_reads")), None
    )
    if read_column is None:
        raise ValueError(f"{xlsx_path}: no *_reads column found in {header}")

    measured: dict[tuple[str, str, int, str], float] = {}
    for row in rows:
        target = row[index["target"]]
        if target is None:
            continue
        key = (
            str(target),
            str(row[index["chrom"]]),
            int(row[index["chromStart"]]),
            str(row[index["strand"]]),
        )
        measured[key] = float(row[index[read_column]] or 0.0)
    workbook.close()
    return measured


def build_dataset(
    candidates_path: str | Path,
    measured_path: str | Path,
    out_path: str | Path | None = None,
) -> OffTargetDataset:
    """Join enumerated candidates against measured read counts."""
    measured = _read_measured(measured_path)

    guide_ids: dict[str, int] = {}
    guide_list: list[str] = []
    idx_rows: list[int] = []
    proto_rows: list[np.ndarray] = []
    read_rows: list[float] = []
    dist_rows: list[int] = []
    seen_measured: set[tuple[str, str, int, str]] = set()

    with open(candidates_path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        columns = {name: i for i, name in enumerate(header)}
        for needed in (
            "target", "chrom", "chromStart", "offtarget_sequence", "strand",
            "distance",
        ):
            if needed not in columns:
                raise ValueError(f"{candidates_path}: missing column {needed!r}")

        for line in handle:
            parts = line.rstrip("\n").split("\t")
            target = parts[columns["target"]]
            sequence = parts[columns["offtarget_sequence"]]
            if len(sequence) != SITE_LEN:
                continue
            key = (
                target,
                parts[columns["chrom"]],
                int(parts[columns["chromStart"]]),
                parts[columns["strand"]],
            )
            reads = measured.get(key, 0.0)
            if reads:
                seen_measured.add(key)

            gid = guide_ids.get(target)
            if gid is None:
                gid = len(guide_list)
                guide_ids[target] = gid
                guide_list.append(target)

            idx_rows.append(gid)
            # Candidate sequences mark mismatches with lowercase; case carries
            # no information the alignment does not already give.
            proto_rows.append(encode(sequence.upper()))
            read_rows.append(reads)
            dist_rows.append(int(parts[columns["distance"]]))

    dataset = OffTargetDataset(
        guides=np.array(guide_list, dtype=object),
        guide_idx=np.asarray(idx_rows, dtype=np.int16),
        proto=(
            np.vstack(proto_rows) if proto_rows
            else np.zeros((0, SITE_LEN), dtype=np.uint8)
        ),
        reads=np.asarray(read_rows, dtype=np.float32),
        distance=np.asarray(dist_rows, dtype=np.uint8),
        dropped_unmatched=len(measured) - len(seen_measured),
    )
    if out_path is not None:
        save_dataset(dataset, out_path)
    return dataset


def save_dataset(dataset: OffTargetDataset, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        guides=np.array([str(g) for g in dataset.guides]),
        guide_idx=dataset.guide_idx,
        proto=dataset.proto,
        reads=dataset.reads,
        distance=dataset.distance,
        dropped_unmatched=np.array([dataset.dropped_unmatched]),
    )


def load_dataset(path: str | Path) -> OffTargetDataset:
    with np.load(path, allow_pickle=False) as blob:
        return OffTargetDataset(
            guides=blob["guides"],
            guide_idx=blob["guide_idx"],
            proto=blob["proto"],
            reads=blob["reads"],
            distance=blob["distance"],
            dropped_unmatched=int(blob["dropped_unmatched"][0]),
        )
