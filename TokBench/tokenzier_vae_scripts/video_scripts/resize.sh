#!/bin/bash
# Video reconstruction (resize baseline) adapted for a single machine.
# The "resize" baseline is pure imageio+PIL (CPU); no GPU needed. We split the
# file list into CHUNKS parallel CPU processes only for speed.
set -e

# ---- config (override via env) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"          # .../VisualTokenizer/TokBench

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/tokbench_data}"    # contains videos/ and video_annotations/
RECON_ROOT="${RECON_ROOT:-$REPO_ROOT/video_reconstruction_results}"
MODEL_NAME="${MODEL_NAME:-resize}"
CHUNKS="${CHUNKS:-8}"                                 # parallel CPU workers
# Video supports 256 and 480. Default 256; e.g. `SHORT_SIZES="256 480" bash resize.sh`.
SHORT_SIZES=(${SHORT_SIZES:-256})

cd "$SCRIPT_DIR"

# ---- text videos ----
DATAS=("ds" "ch3")
for DATA in "${DATAS[@]}"; do
    for SHORT_SIZE in "${SHORT_SIZES[@]}"; do
        echo "[$MODEL_NAME] short=$SHORT_SIZE dataset=$DATA (text)"
        for IDX in $(seq 0 $((CHUNKS-1))); do
            python resize_rec.py \
                --video_path "$DATA_ROOT/videos/text_data/$DATA" \
                --save_path  "$RECON_ROOT/$MODEL_NAME/text_data/$DATA" \
                --short_size $SHORT_SIZE \
                --num_chunks $CHUNKS \
                --chunk_idx $IDX &
        done
        wait
    done
done

# ---- face videos (upstream pointed these at text_data by mistake -> fixed to face_data) ----
DATAS=("face_clip_3s")
for DATA in "${DATAS[@]}"; do
    for SHORT_SIZE in "${SHORT_SIZES[@]}"; do
        echo "[$MODEL_NAME] short=$SHORT_SIZE dataset=$DATA (face)"
        for IDX in $(seq 0 $((CHUNKS-1))); do
            python resize_rec.py \
                --video_path "$DATA_ROOT/videos/face_data/$DATA" \
                --save_path  "$RECON_ROOT/$MODEL_NAME/face_data/$DATA" \
                --short_size $SHORT_SIZE \
                --num_chunks $CHUNKS \
                --chunk_idx $IDX &
        done
        wait
    done
done

echo "Video reconstruction done -> $RECON_ROOT/$MODEL_NAME"
