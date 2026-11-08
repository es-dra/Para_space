#!/usr/bin/env bash
# Stage 10 on one GPU. Restartable: eval_natural skips finished zoos, build_big_zoo resumes with --offset,
# train_wsl runs are skipped when their json exists.
#   KODAK=Data/kodak MNIST=Data/mnist DEVICE=cuda:0 bash scripts/gpu/run_scale.sh [A|B|all]
set -eu
cd "$(dirname "$0")/../.."
PART=${1:-all}
DEVICE=${DEVICE:-cuda:0}
KODAK=${KODAK:-Data/kodak}
MNIST=${MNIST:-Data/mnist}
OUT=studies/10-gpu-scale

if [ "$PART" = A ] || [ "$PART" = all ]; then
  # Part A: natural images at 256 x 256. Big shallow nets fit one crop at a time (memory).
  python3 scripts/eval_natural.py --images "$KODAK" --n 24 --side 256 --up 4 --device "$DEVICE" --chunk 4 \
      --zoos deep_h64 shallow_w1024 deep_h128 deep_h128x3 deep_h128_swing2 deep_h128_swing4 shallow_w2048 --out "$OUT/natural256"
  python3 scripts/eval_natural.py --images "$KODAK" --n 24 --side 256 --up 4 --device "$DEVICE" --chunk 1 \
      --zoos shallow_w4096 shallow_w8192 --out "$OUT/natural256"
  # sensitivity: more crops for the cheapest pair
  python3 scripts/eval_natural.py --images "$KODAK" --n 48 --side 256 --up 4 --device "$DEVICE" --chunk 4 \
      --zoos deep_h64 shallow_w1024 --out "$OUT/natural256_n48"
fi

if [ "$PART" = B ] || [ "$PART" = all ]; then
  # Part B: 10k zoo and the augmentation grid.
  [ -f zoo/mnist_s32_train10k/index.json ] && [ "$(ls zoo/mnist_s32_train10k/*.npz | wc -l)" -ge 10000 ] || \
    python3 scripts/build_big_zoo.py --mnist "$MNIST" --split train --n 10000 --chunk 1000 --device "$DEVICE" --out zoo/mnist_s32_train10k
  [ -f zoo/mnist_s32_test1k/index.json ] || \
    python3 scripts/build_big_zoo.py --mnist "$MNIST" --split test --n 1000 --chunk 1000 --device "$DEVICE" --out zoo/mnist_s32_test1k
  R=$OUT/wsl; mkdir -p "$R"
  for n in 1000 2000 5000 10000; do for model in nfn deepsets2 aligned_mlp; do for aug in none programs permute; do for seed in 0 1 2; do
    for width in 64 128; do
      [ "$model" = nfn ] || [ "$width" = 64 ] || continue   # width sweep only for the NFN
      f="$R/${model}_w${width}_${aug}_${n}_s${seed}.json"
      [ -f "$f" ] && continue
      python3 scripts/train_wsl.py --train zoo/mnist_s32_train10k --test zoo/mnist_s32_test1k --n-train "$n" --n-test 1000 \
        --model "$model" --aug "$aug" --seed "$seed" --epochs 60 --width "$width" --device "$DEVICE" --out "$f" > "${f%.json}.log" 2>&1
    done
  done; done; done; done
fi
echo SCALE_DONE
