#!/usr/bin/env bash
# Stage 07 grid: sizes x seeds x augmentations x models, two workers. Skips finished runs.
set -u
cd "$(dirname "$0")/.."
TRAIN=${TRAIN:-zoo/mnist_s32_train5k}; TEST=${TEST:-zoo/mnist_s32_test1k}; OUT=${OUT:-results/wsl}
EPOCHS=${EPOCHS:-60}
mkdir -p "$OUT"
jobs_file=$(mktemp)
for n in 500 1000 2000 5000; do for model in deepsets2 aligned_mlp; do for aug in none programs permute; do for seed in 0 1 2; do
  f="$OUT/${model}_${aug}_${n}_s${seed}.json"
  [ -f "$f" ] || echo "python3 scripts/train_wsl.py --train $TRAIN --test $TEST --n-train $n --model $model --aug $aug --seed $seed --epochs $EPOCHS --threads 2 --out $f > $OUT/${model}_${aug}_${n}_s${seed}.log 2>&1"
done; done; done; done > "$jobs_file"
echo "$(wc -l < "$jobs_file") runs queued"
xargs -P 2 -I {} sh -c "{}" < "$jobs_file"
echo GRID_DONE
