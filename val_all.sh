#!/usr/bin/env bash

# BASE="../test_exp/occrwkv/outputs/ms_v4/chkpt/best-metric/"
BASE="../SSC_out/LMSCNet_2d_MultiDataset_1125_165040/chkpt/last"
CFG="SSC_configs/examples/LMSCNet_2d_navio.yaml"

for WEIGHTS in "$BASE"/weights_epoch_*.pth; do
    # Извлечь только номер эпохи из имени файла (001, 002, 010, ...)
    EPOCH=$(basename "$WEIGHTS" | sed -E 's/.*weights_epoch_([0-9]+)\.pth/\1/')

    OUT="output/ms_time/epoch_${EPOCH}"

    echo "✅ Running epoch $EPOCH"
    echo "   weights: $WEIGHTS"
    echo "   out_path: $OUT"

    python3 LMSCNet/validate.py --weights "$WEIGHTS" --cfg "$CFG" --out_path "$OUT"
done
