#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

STAMP="${1:-$(date '+%Y%m%d_%H%M%S')}"
OUT_DIR="${EVIDENCE_DIR:-$ROOT_DIR/evidence/cfn_${STAMP}}"

if [[ -e "$OUT_DIR" ]]; then
  suffix=1
  while [[ -e "${OUT_DIR}_${suffix}" ]]; do
    suffix=$((suffix + 1))
  done
  OUT_DIR="${OUT_DIR}_${suffix}"
fi

mkdir -p "$OUT_DIR/models" "$OUT_DIR/meta"

# Copy model/result artifacts only (weights + reports + logs) to avoid huge raw assets.
rsync -a --prune-empty-dirs \
  --include="*/" \
  --include="*.json" \
  --include="*.jsonl" \
  --include="*.csv" \
  --include="*.pth" \
  --include="*.pkl" \
  --exclude="*" \
  "models/" "$OUT_DIR/models/"

copy_if_exists() {
  local src="$1"
  if [[ -f "$src" ]]; then
    mkdir -p "$OUT_DIR/$(dirname "$src")"
    cp "$src" "$OUT_DIR/$src"
  fi
}

copy_if_exists "notebooks/CFN_accuracy_colab_notes.ipynb"
copy_if_exists "notebooks/technique_runs.csv"
copy_if_exists "notebooks/technique_catalog.csv"
copy_if_exists "eval_manifest_mixed.tsv"
copy_if_exists "eval_manifest_mixed_val.tsv"
copy_if_exists "eval_manifest_mixed_test.tsv"
copy_if_exists "train_eval_preds.jsonl"
copy_if_exists "train_eval_preds_v2.jsonl"

{
  echo "snapshot_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "snapshot_local=$(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "cwd=$ROOT_DIR"
  echo "user=${USER:-unknown}"
  echo "host=$(hostname)"
  echo "command=$0 $*"
  git rev-parse HEAD 2>/dev/null | sed 's/^/git_head=/'
} > "$OUT_DIR/meta/snapshot_info.txt"

git status --short > "$OUT_DIR/meta/git_status_short.txt" 2>/dev/null || true
git diff --name-only > "$OUT_DIR/meta/git_changed_files.txt" 2>/dev/null || true

(
  cd "$OUT_DIR"
  find . -type f ! -name "sha256_manifest.txt" -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 shasum -a 256 > "meta/sha256_manifest.txt"
)

echo "Evidence snapshot created: $OUT_DIR"
echo "Manifest: $OUT_DIR/meta/sha256_manifest.txt"
