#!/usr/bin/env bash
# Reproduce every number in the README, in the order it reports them.
#
#   bash scripts/fetch_data.sh      # once, downloads the data
#   bash scripts/reproduce.sh
#
# Expect about 45 minutes on a laptop. The two cross-validation steps and the
# TrueOT benchmark dominate that; everything before them takes seconds.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
CLI="$PY -m qcrispr.cli"
BULGE="$ROOT/external/crispr_bulge/datasets"

cd "$ROOT"

if [ ! -f data/cfd_mismatch.tsv ]; then
  echo "data/cfd_mismatch.tsv missing. Run scripts/fetch_data.sh first." >&2
  exit 1
fi

echo "===== tests ====="
$PY -m pytest -q

echo
echo "===== synthetic genome, enumeration recall ====="
$CLI demo-genome
$CLI validate

echo
echo "===== guide-set selection as a QUBO ====="
$CLI candidates --top 8
$CLI solve --n 14 --k 4 --p 3
$CLI benchmark --instances 8 --n 14 --k 4 --p-values 2,3 --qaoa-restarts 2

echo
echo "===== measured CHANGE-seq, leave-guides-out ====="
$CLI dataset
$CLI evaluate --max-iter 300 --out results/offtarget_eval.tsv
echo "-- same thing without per-position identity, for the ablation --"
$CLI evaluate --basic-features --out results/offtarget_eval_basic.tsv

echo
echo "===== measured bulge sites, rank correlation ====="
$CLI bulges

echo
echo "===== refined TrueOT, no transfer ====="
$CLI trueot --out results/trueot.tsv

echo
echo "===== refined TrueOT, with cellular fine-tuning ====="
$CLI trueot \
  --finetune "$BULGE/FullGUIDEseq/include_on_targets/FullGUIDEseq_CR_Lazzarotto_2020_dataset.csv" \
  --out results/trueot_transfer.tsv

echo
echo "===== leaked variant, to show the size of the leakage effect ====="
$CLI trueot --keep-overlap --out results/trueot_leaked.tsv

echo
echo "all results written to results/"
