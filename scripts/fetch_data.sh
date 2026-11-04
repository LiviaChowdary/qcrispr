#!/usr/bin/env bash
# Download the third-party data this repository evaluates on.
#
# Nothing here is redistributed in the repository. Read NOTICE before running
# this: the CFD scoring tables are licensed for academic and non-profit use
# only, and this script fetches them on your behalf, not on its own authority.
#
# Total download is roughly 1.1 GB, expanding to about 2.2 GB on disk.
#
#   bash scripts/fetch_data.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT="$ROOT/external"
PY="${PYTHON:-$ROOT/.venv/bin/python}"

if [ ! -x "$PY" ]; then
  echo "No interpreter at $PY. Create one with:" >&2
  echo "  python3 -m venv .venv && ./.venv/bin/pip install -e '.[dev]'" >&2
  exit 1
fi

mkdir -p "$EXT/cfd" "$EXT/syseval" "$EXT/crispr_bulge"

echo "== 1/4 CFD scoring tables (academic and non-profit use only) =="
# Protocol-0 pickles. scripts/import_cfd.py parses them without calling
# pickle.load, since unpickling a downloaded file executes arbitrary code.
curl -fsSL -o "$EXT/cfd/mismatch_score.pkl" \
  "https://raw.githubusercontent.com/bm2-lab/iGWOS/master/CFD/mismatch_score.pkl"
curl -fsSL -o "$EXT/cfd/pam_scores.pkl" \
  "https://raw.githubusercontent.com/bm2-lab/iGWOS/master/CFD/pam_scores.pkl"
"$PY" -I "$ROOT/scripts/import_cfd.py" \
  "$EXT/cfd/mismatch_score.pkl" "$EXT/cfd/pam_scores.pkl"

echo
echo "== 2/4 CHANGE-seq candidates and measured reads (SysEvalOffTarget) =="
curl -fsSL -o "$EXT/syseval/output_file_pam_change.zip" \
  "https://raw.githubusercontent.com/OrensteinLab/SysEvalOffTarget/main/files/datasets/output_file_pam_change.zip"
curl -fsSL -o "$EXT/syseval/changeseq.xlsx" \
  "https://raw.githubusercontent.com/OrensteinLab/SysEvalOffTarget/main/files/datasets/CHANGE-seq.xlsx"
unzip -o -q "$EXT/syseval/output_file_pam_change.zip" -d "$EXT/syseval"

echo
echo "== 3/4 bulge-aligned datasets and refined TrueOT (CRISPR-Bulge, Git LFS) =="
# The file is a Git LFS object, so the raw URL returns a pointer rather than
# the archive. Resolve it through the LFS batch API instead.
OID="f892f70ba4ac3b05b03b2171b4ad38746630de08ad630e650f355dd61203eab0"
SIZE=524400344
curl -fsSL -X POST \
  -H "Accept: application/vnd.git-lfs+json" \
  -H "Content-Type: application/vnd.git-lfs+json" \
  -d "{\"operation\":\"download\",\"transfers\":[\"basic\"],\"objects\":[{\"oid\":\"$OID\",\"size\":$SIZE}]}" \
  "https://github.com/OrensteinLab/CRISPR-Bulge.git/info/lfs/objects/batch" \
  -o "$EXT/crispr_bulge/lfs.json"
HREF="$("$PY" -I -c "import json,sys; print(json.load(open(sys.argv[1]))['objects'][0]['actions']['download']['href'])" "$EXT/crispr_bulge/lfs.json")"
curl -fsSL -o "$EXT/crispr_bulge/datasets.zip" "$HREF"

echo "verifying checksum"
echo "$OID  $EXT/crispr_bulge/datasets.zip" | shasum -a 256 -c -

echo
echo "== 4/4 extracting the tables the benchmarks use =="
unzip -o -q "$EXT/crispr_bulge/datasets.zip" \
  "datasets/Refined_TrueOT.csv" \
  "datasets/CHANGEseq/include_on_targets/CHANGEseq_CR_Lazzarotto_2020_dataset.csv" \
  "datasets/FullGUIDEseq/include_on_targets/FullGUIDEseq_CR_Lazzarotto_2020_dataset.csv" \
  -d "$EXT/crispr_bulge"

echo
echo "done. next: bash scripts/reproduce.sh"
