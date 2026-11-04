#!/usr/bin/env python3
"""Convert the published CFD lookup tables into plain TSV.

Run once:

    python scripts/import_cfd.py <mismatch.pkl> <pam.pkl>

The CFD weights ship as Python pickles in various third-party repositories
(the Doench 2016 supplement itself is a spreadsheet). Those pickles are
downloaded files, so this script never calls `pickle.load` on them --
unpickling runs arbitrary code by design, and no lookup table is worth that.
Instead it checks that the file is a pure-ASCII protocol-0 pickle containing
none of the opcodes that can construct objects or call functions, then
extracts the string/float pairs textually.

### The base-convention trap

The reference implementation (`cfd-score-calculator.py`) builds its lookup key
as:

    key = 'r' + guide_base + ':d' + revcom(offtarget_base) + ',' + position

so the `d` base in a CFD key is the *complement* of the base actually observed
in the genomic protospacer, because it names the base on the opposite strand.
Using the observed base directly would silently read the wrong weight for
every mismatch, and the resulting scores would still look plausible.

This script resolves it once, here: the emitted `dna_base` column is the
observed protospacer base, obtained by complementing the key's `d` base. Every
consumer downstream then works in one consistent convention.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MISMATCH_OUT = ROOT / "data" / "cfd_mismatch.tsv"
PAM_OUT = ROOT / "data" / "cfd_pam.tsv"

COMPLEMENT = str.maketrans("ACGT", "TGCA")

# Protocol-0 opcodes that build objects, import names, or call callables.
# A legitimate dict-of-str-to-float table needs none of them.
FORBIDDEN_OPCODES = {
    "c": "GLOBAL (imports a name)",
    "R": "REDUCE (calls a callable)",
    "i": "INST (instantiates a class)",
    "o": "OBJ (instantiates a class)",
    "b": "BUILD (calls __setstate__)",
}

KEY_VALUE = re.compile(r"S'([^']+)'\s*(?:\np\d+)?\s*\nF([-+0-9.eE]+)")
MISMATCH_KEY = re.compile(r"^r([ACGU])\:d([ACGT]),(\d{1,2})$")


def audit_and_read(path: Path) -> str:
    """Reject anything that is not a plain protocol-0 pickle of literals."""
    raw = path.read_bytes()
    if any(byte > 127 for byte in raw):
        raise SystemExit(
            f"{path.name}: contains non-ASCII bytes, so it is not a protocol-0 "
            f"pickle. Refusing to parse it."
        )
    text = raw.decode("ascii")
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line:
            continue
        opcode = line[0]
        if opcode in FORBIDDEN_OPCODES:
            raise SystemExit(
                f"{path.name}:{lineno}: opcode {opcode!r} "
                f"({FORBIDDEN_OPCODES[opcode]}) found. Refusing to parse it."
            )
    return text


def parse_pairs(path: Path) -> dict[str, float]:
    text = audit_and_read(path)
    pairs = KEY_VALUE.findall(text)
    if not pairs:
        raise SystemExit(f"{path.name}: no string/float pairs found")
    table: dict[str, float] = {}
    for key, value in pairs:
        table[key] = float(value)
    if len(table) != len(pairs):
        raise SystemExit(f"{path.name}: duplicate keys in source table")
    return table


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_mismatch(table: dict[str, float], source: Path) -> int:
    rows: list[tuple[int, str, str, float]] = []
    for key, value in table.items():
        match = MISMATCH_KEY.match(key)
        if not match:
            raise SystemExit(f"unrecognised mismatch key {key!r}")
        rna, dna_key, position = match.groups()
        guide_base = rna.replace("U", "T")
        observed_base = dna_key.translate(COMPLEMENT)
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"{key}: retained activity {value} outside [0, 1]")
        rows.append((int(position), guide_base, observed_base, value))
    rows.sort()

    MISMATCH_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MISMATCH_OUT, "w") as fh:
        fh.write(f"# CFD mismatch weights, imported from {source.name}\n")
        fh.write(f"# source sha256: {sha256(source)}\n")
        fh.write(
            "# dna_base is the base OBSERVED in the genomic protospacer. The CFD\n"
            "# key names the complementary base; it was complemented on import.\n"
        )
        fh.write("position\trna_base\tdna_base\tretained_activity\n")
        for position, guide_base, observed_base, value in rows:
            fh.write(f"{position}\t{guide_base}\t{observed_base}\t{value!r}\n")
    return len(rows)


def write_pam(table: dict[str, float], source: Path) -> int:
    rows = sorted(table.items())
    for key, value in rows:
        if not re.fullmatch(r"[ACGT]{2}", key):
            raise SystemExit(f"unrecognised PAM key {key!r}; expected a 2-mer")
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"{key}: PAM factor {value} outside [0, 1]")

    PAM_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(PAM_OUT, "w") as fh:
        fh.write(f"# CFD PAM factors, imported from {source.name}\n")
        fh.write(f"# source sha256: {sha256(source)}\n")
        fh.write(
            "# Keyed by the last two PAM bases: CFD scores NGG-family PAMs by\n"
            "# positions 2-3 only, so the first base is ignored.\n"
        )
        fh.write("pam2\tfactor\n")
        for key, value in rows:
            fh.write(f"{key}\t{value!r}\n")
    return len(rows)


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        raise SystemExit(__doc__)
    mismatch_src, pam_src = Path(argv[0]), Path(argv[1])

    mismatch = parse_pairs(mismatch_src)
    pam = parse_pairs(pam_src)

    n_mismatch = write_mismatch(mismatch, mismatch_src)
    n_pam = write_pam(pam, pam_src)

    print(f"{mismatch_src.name}: {n_mismatch} weights -> {MISMATCH_OUT}")
    print(f"{pam_src.name}: {n_pam} factors -> {PAM_OUT}")

    expected = 12 * 20
    if n_mismatch == expected:
        print(f"{n_mismatch} = 12 mismatch pairs x 20 positions, as published")
    else:
        print(
            f"note: {n_mismatch} weights, not the canonical {expected} "
            f"(12 pairs x 20 positions) -- check which variant this table is"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
