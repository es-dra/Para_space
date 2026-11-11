# Stage 10: GPU scale-up (protocol frozen 2026-11-08, not yet run)

`protocol.json` fixes the data, the matched zoo pairs and predictions S1-S7 (natural images at 256 x 256,
compression regime) and W1-W4 (weight-space augmentation on a 10k zoo) before any GPU run.
`scripts/gpu/run_scale.sh` executes parts A and B and is restartable (finished zoos and runs are skipped).
Outcomes go to `outcomes.json` after the runs, failures included.

## Amendment B2 (frozen 2026-11-11)

Part B failed W1 on the independent-initialisation zoo (classifier at chance; W2-W4 void). The diagnosis and the
re-run on a shared-initialisation zoo at lr 5e-4 are fixed in `protocol.json` (`B2_shared_init_amendment`) with the
unchanged W1-W4 thresholds. Run with `LANE=i NLANES=4 DEVICE=cuda:0 bash scripts/gpu/run_scale.sh B2` on four GPUs;
results in `wsl_shared/`.
