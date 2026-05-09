# Stage B Run Manifest

正式结果目录：`results/FittingDynamics_StageB/`

Stage B 只使用 scratch runs，不使用 pretrained LIIF/LIIF-EQ。

## SIREN Scratch

Command template:

```bash
CUDA_VISIBLE_DEVICES=2 python experiments/Phase1_FittingDynamics/run_siren.py \
  --image Data/Set5/HR/{image}.png \
  --seed {seed} \
  --device cuda \
  --save_dir results/FittingDynamics_StageB/SIREN_{image}_seed{seed}
```

| Run dir | Image | Seed | Steps | Snapshots | Final PSNR | Status |
|---|---|---:|---:|---:|---:|---|
| `SIREN_baby_seed42` | baby.png | 42 | 5000 | 12 | 99.005 | official Stage B |
| `SIREN_baby_seed123` | baby.png | 123 | 5000 | 12 | 99.965 | official Stage B |
| `SIREN_baby_seed456` | baby.png | 456 | 5000 | 12 | 99.970 | official Stage B |
| `SIREN_bird_seed42` | bird.png | 42 | 5000 | 12 | 100.000 | official Stage B |
| `SIREN_butterfly_seed42` | butterfly.png | 42 | 5000 | 12 | 100.000 | official Stage B |

## LIIF Reduced Scratch SR x4

Command template:

```bash
CUDA_VISIBLE_DEVICES=2 python experiments/Phase1_FittingDynamics/run.py \
  --model liif \
  --image Data/Set5/HR/{image}.png \
  --sr 4 \
  --seed {seed} \
  --device cuda \
  --save_dir results/FittingDynamics_StageB/LIIF_reduced_{image}_sr4_seed{seed}
```

| Run dir | Image | Seed | Steps | Snapshots | Params | Final PSNR | Status |
|---|---|---:|---:|---:|---:|---:|---|
| `LIIF_reduced_baby_sr4_seed42` | baby.png | 42 | 25000 | 11 | 138531 | 28.526 | original Stage B |
| `LIIF_reduced_baby_sr4_seed123` | baby.png | 123 | 25000 | 11 | 138531 | 30.747 | original Stage B |
| `LIIF_reduced_baby_sr4_seed456` | baby.png | 456 | 25000 | 11 | 138531 | 30.581 | original Stage B |
| `LIIF_reduced_bird_sr4_seed42` | bird.png | 42 | 25000 | 11 | 138531 | 21.542 | original Stage B |
| `LIIF_reduced_butterfly_sr4_seed42` | butterfly.png | 42 | 25000 | 11 | 138531 | 26.863 | original Stage B |
| `LIIF_reduced_head_sr4_seed42` | head.png | 42 | 25000 | 11 | 138531 | 31.590 | Stage C extension |
| `LIIF_reduced_woman_sr4_seed42` | woman.png | 42 | 25000 | 11 | 138531 | 29.722 | Stage C extension |

## Validation

All official Stage B and Stage C extension LIIF result dirs passed:

- trajectory schema;
- trajectory shapes;
- summary schema;
- analysis readiness;
- summary consistency.

Validation command:

```bash
for d in results/FittingDynamics_StageB/*; do
  [ -d "$d" ] || continue
  [ "$(basename "$d")" = logs ] && continue
  python experiments/Phase1_FittingDynamics/check_outputs.py "$d"
done
```

## Interpretation Boundary

- Original Stage B PCA/control tables use the original 10 runs only.
- head/woman are formal scratch reduced LIIF outputs, but they are Stage C pilot extensions, not part of the original Stage B PCA/control matrix.
