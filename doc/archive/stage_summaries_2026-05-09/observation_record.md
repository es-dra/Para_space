# Stage B Observation Record

This is the concise canonical Stage B record. The longer historical log remains at [../../archive/historical_docs/StageB_observation_log.md](../../archive/historical_docs/StageB_observation_log.md).

## Official Result Directory

```text
results/FittingDynamics_StageB/
```

## Protocol

- Scratch only.
- No `run_finetune.py`.
- No pretrained LIIF / LIIF-EQ.
- Reduced LIIF uses `LIIF_CONFIG_REDUCED`, `n_params=138531`, `sr_scale=4`.
- Outputs must pass `check_outputs.py`.
- PCA is diagnostic only.

## Official Matrix

Original Stage B:

- SIREN scratch：baby seeds 42/123/456，bird seed42，butterfly seed42。
- LIIF reduced scratch SR x4：baby seeds 42/123/456，bird seed42，butterfly seed42。

Stage C extension but same scratch protocol:

- LIIF reduced scratch SR x4：head seed42，woman seed42。

## Validation

All official outputs passed:

- trajectory schema;
- trajectory shapes;
- summary schema;
- analysis readiness;
- summary consistency.

## Boundary

head/woman are valid scratch LIIF outputs but not part of the original Stage B PCA/control table.
