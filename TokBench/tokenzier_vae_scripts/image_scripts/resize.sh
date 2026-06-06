#!/bin/bash
# Reconstruction (resize baseline) adapted for a single machine.
# The "resize" baseline is pure PIL (CPU); it needs NO GPU. We still split the
# file list into CHUNKS parallel CPU processes purely to speed things up.
set -e

# ---- config (override via env, e.g. `DATA_ROOT=/data/tok bash resize.sh`) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"          # .../VisualTokenizer/TokBench

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/tokbench_data}"    # contains images/ and annotations/
RECON_ROOT="${RECON_ROOT:-$REPO_ROOT/image_reconstruction_results}"
MODEL_NAME="${MODEL_NAME:-resize}"
CHUNKS="${CHUNKS:-8}"                                 # parallel CPU workers
# Padding sizes to reconstruct. Default 256 for a quick run; add "512 1024" to reproduce the paper.
PADDING_SIZES=(${PADDING_SIZES:-256})

cd "$SCRIPT_DIR"

DATAS=("ic13" "ic15" "textocr" "tt" "cord" "docvqa" "infograph" "sroie")
for DATA in "${DATAS[@]}"; do
    for PADDING_SIZE in "${PADDING_SIZES[@]}"; do
        echo "[$MODEL_NAME] padding=$PADDING_SIZE dataset=$DATA (text)"
        for IDX in $(seq 0 $((CHUNKS-1))); do
            python resize_rec.py \
                --image_path "$DATA_ROOT/images/text_data/$DATA" \
                --save_path  "$RECON_ROOT/$MODEL_NAME/text_data/$DATA" \
                --padding_size $PADDING_SIZE \
                --num_chunks $CHUNKS \
                --chunk_idx $IDX &
        done
        wait
    done
done

DATAS=("wflw")
for DATA in "${DATAS[@]}"; do
    for PADDING_SIZE in "${PADDING_SIZES[@]}"; do
        echo "[$MODEL_NAME] padding=$PADDING_SIZE dataset=$DATA (face)"
        for IDX in $(seq 0 $((CHUNKS-1))); do
            python resize_rec.py \
                --image_path "$DATA_ROOT/images/face_data/$DATA" \
                --save_path  "$RECON_ROOT/$MODEL_NAME/face_data/$DATA" \
                --padding_size $PADDING_SIZE \
                --num_chunks $CHUNKS \
                --chunk_idx $IDX &
        done
        wait
    done
done

echo "Reconstruction done -> $RECON_ROOT/$MODEL_NAME"
