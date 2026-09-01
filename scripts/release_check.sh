#!/usr/bin/env bash
# Reproducibility preflight for the public release.
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK_DIR=${1:-"$ROOT_DIR/build/release-check"}
BUILD_DIR="$WORK_DIR/cmake"
M2DGR_DIR="$WORK_DIR/m2dgr"

mkdir -p "$BUILD_DIR" "$M2DGR_DIR"

echo "[1/6] Python syntax and entry points"
PYTHONPYCACHEPREFIX="$WORK_DIR/pycache" python3 -m compileall -q "$ROOT_DIR/baselines" "$ROOT_DIR/experiments" "$ROOT_DIR/figures"
python3 "$ROOT_DIR/experiments/common/run_overlap_transformer.py" --help >/dev/null
python3 "$ROOT_DIR/experiments/common/run_minkloc3dv2.py" --help >/dev/null

echo "[2/6] Result-bundle checksums"
(cd "$ROOT_DIR/data/package" && sha256sum -c SHA256SUMS)

echo "[3/6] Standalone C++ build"
cmake -S "$ROOT_DIR/updown_sc" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j2

echo "[4/6] Safe archive paths"
for archive in "$ROOT_DIR"/data/package/*.tar.gz; do
  if tar -tzf "$archive" | grep -Eq '(^/|(^|/)\.\.(/|$))'; then
    echo "unsafe archive path in $archive" >&2
    exit 1
  fi
done

echo "[5/6] M2DGR deterministic retrieval replay"
tar -xzf "$ROOT_DIR/data/package/m2dgr_2m.tar.gz" -C "$M2DGR_DIR"
DATA_ROOT="$M2DGR_DIR/m2dgr_hall_eval_2m"
"$BUILD_DIR/scan_context_cross_sequence_evaluator" "$DATA_ROOT/scd/map.scd" "$DATA_ROOT/scd/query.scd" "$WORK_DIR/rerun_candidates.csv" 100 0.3 0.7 2 0.5 0.1 >/dev/null
cut -d, -f1-24 "$DATA_ROOT/results/updown_candidates.csv" > "$WORK_DIR/recorded_core.csv"
cut -d, -f1-24 "$WORK_DIR/rerun_candidates.csv" > "$WORK_DIR/rerun_core.csv"
cmp "$WORK_DIR/recorded_core.csv" "$WORK_DIR/rerun_core.csv"

echo "[6/6] M2DGR reported recall"
python3 "$ROOT_DIR/experiments/construction_hall_cross_sequence/evaluate_candidate_csv.py" --candidates "$WORK_DIR/rerun_candidates.csv" --metadata "$DATA_ROOT/derived/queries_seq2/metadata.csv" --output-per-query "$WORK_DIR/rerun_per_query.csv" --output-summary "$WORK_DIR/rerun_summary.csv" --algorithm UpDown-SC --correct-radius 2.0 >/dev/null
grep -Fq 'UpDown-SC,proposed_gravity_canonicalized_single_scan,28,0.9285714285714286,1.0' "$WORK_DIR/rerun_summary.csv"

echo "release check passed"
