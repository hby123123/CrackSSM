#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
DATASET="${1:-TUT}"
DATA_DIR="${CRACKSSM_DATA_ROOT:-$PROJECT_ROOT/data}"
CKPT_DIR="${CRACKSSM_CKPT_DIR:-$PROJECT_ROOT/checkpoints}"
CKPT="${CRACKSSM_MODEL_PATH:-$CKPT_DIR/$DATASET/best_model.pth}"

if [ ! -d "$DATA_DIR/$DATASET" ]; then
    echo "Error: data dir not found: $DATA_DIR/$DATASET"
    echo "Put dataset under data/<name>/ with test_img, test_lab."
    exit 1
fi
if [ ! -f "$CKPT" ]; then
    echo "Error: checkpoint not found: $CKPT"
    echo "Train or put best_model.pth there, or set CRACKSSM_MODEL_PATH."
    exit 1
fi

echo "Dataset: $DATASET | Data: $DATA_DIR | Checkpoint: $CKPT"
python test/run_test.py --dataset "$DATASET" --ckpt "$CKPT" --data_root "$DATA_DIR"
echo "Done. Results in test/test_results/$DATASET"
