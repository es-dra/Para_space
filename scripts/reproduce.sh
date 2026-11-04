#!/usr/bin/env bash
# Reproduce every stage from scratch (CPU, NumPy trainer). Data: the four MNIST idx .gz files in $MNIST and the two
# Fashion-MNIST test files in $FASHION. Total wall time on 4 CPU cores: roughly 6 hours.
set -euo pipefail
MNIST=${MNIST:-Data/mnist}; FASHION=${FASHION:-Data/fashion}
b() { python scripts/build_zoo.py --mnist "$1" --n "$2" --steps 2000 --backend np "${@:3}"; }

# stage 00 / 02 / 03 zoos
b "$MNIST" 60 --hidden 32 --hidden-layers 1 --out zoo/mnist_s32
b "$MNIST" 60 --hidden 64 --hidden-layers 0 --out zoo/mnist_shallow64
b "$MNIST" 60 --hidden 32 --hidden-layers 0 --out zoo/mnist_shallow32
b "$MNIST" 60 --hidden 32 --hidden-layers 2 --out zoo/mnist_d4_s32
for W in 128 256 384 1024; do b "$MNIST" 20 --hidden $W --hidden-layers 0 --out zoo/mnist_shallow$W; done
b "$MNIST" 20 --hidden 8 --hidden-layers 1 --out zoo/mnist_d3_s8
b "$MNIST" 20 --hidden 16 --hidden-layers 1 --out zoo/mnist_d3_s16
b "$MNIST" 20 --hidden 16 --hidden-layers 2 --out zoo/mnist_d4_s16
python scripts/build_zoo.py --mnist "$MNIST" --n 5 --steps 10000 --hidden 384 --hidden-layers 0 --backend np --out zoo/mnist_shallow384_long
b "$FASHION" 20 --hidden 384 --hidden-layers 0 --out zoo/fashion_shallow384
b "$FASHION" 20 --hidden 32 --hidden-layers 1 --out zoo/fashion_s32
b "$FASHION" 20 --hidden 16 --hidden-layers 1 --out zoo/fashion_d3_s16
python scripts/build_null_zoo.py --like zoo/mnist_s32 --n 20 --out zoo/null_s32

# stage 02: programs by layer and depth, per-unit spectra
for z in "mnist_shallow64 L2w64" "mnist_shallow32 L2w32" "mnist_s32 L3w32" "mnist_d4_s32 L4w32"; do set -- $z
  python scripts/eval_programs.py --zoo zoo/$1 --tag $2 --n 20 --out results/programs_$2.json; done
python scripts/analyze_programs.py --results results/programs_L2w64.json results/programs_L2w32.json results/programs_L3w32.json results/programs_L4w32.json --out studies/02-programs/outcomes.json
python scripts/unit_spectra.py --zoo zoo/mnist_s32 --n 20 --sigma 1 --out results/unit_spectra_L3w32.json
python scripts/unit_spectra.py --zoo zoo/mnist_d4_s32 --n 20 --sigma 1 --out results/unit_spectra_L4w32.json
for lam in 1e-3 1e-2; do b "$MNIST" 20 --hidden 32 --hidden-layers 1 --harmonic-penalty $lam --out zoo/mnist_s32_lam$lam
  python scripts/eval_programs.py --zoo zoo/mnist_s32_lam$lam --tag L3w32_lam$lam --n 20 --out results/programs_L3w32_lam$lam.json; done

# stage 03: width vs depth
python scripts/analyze_width_depth.py --out studies/03-width-vs-depth/outcomes.json

# stage 04: editable training (exploratory seed-0 series and the confirmatory v2 grid)
for B in 0.5 1 2 4; do b "$MNIST" 20 --hidden 32 --hidden-layers 1 --swing-bound $B --out zoo/mnist_s32_swing$B
  python scripts/eval_programs.py --zoo zoo/mnist_s32_swing$B --tag L3w32_swing$B --n 20 --sigmas 1 --steps 10 50 --out results/programs_L3w32_swing$B.json; done
for ds in "mnist $MNIST" "fashion $FASHION"; do set -- $ds; for seed in 0 1; do for B in 1 2 4 8; do z=${1}_s32_swing${B}_seed$seed
  b "$2" 20 --hidden 32 --hidden-layers 1 --swing-bound $B --seed $seed --out zoo/$z
  python scripts/eval_programs.py --zoo zoo/$z --tag $z --n 20 --sigmas 1 --steps 10 --out results/programs_$z.json; done; done; done
python scripts/analyze_editable.py --out studies/04-editable-training/outcomes_v2.json
python scripts/plot_tradeoff.py --out studies/04-editable-training/tradeoff.png
