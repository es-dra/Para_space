# Stage 10: GPU scale-up (protocol frozen 2026-11-08, not yet run)

`protocol.json` fixes the data, the matched zoo pairs and predictions S1-S7 (natural images at 256 x 256,
compression regime) and W1-W4 (weight-space augmentation on a 10k zoo) before any GPU run.
`scripts/gpu/run_scale.sh` executes parts A and B and is restartable (finished zoos and runs are skipped).
Outcomes go to `outcomes.json` after the runs, failures included.
