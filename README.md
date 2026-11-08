# Para_space: weight-space programs for implicit neural representations

Which image operations can be written directly into the weights of an implicit neural representation (INR),
without training, and what does that programmability cost?

A SIREN `f(p) = A_L sin(... sin(A1 p + c1) ...) + c_L` separates into a coordinate layer, a dictionary of
sinusoidal units and a linear readout. Affine coordinate maps act on layer 1 exactly; affine value maps act on
the readout exactly; and for a one-sine-layer network the dictionary is a set of plane waves, so **any linear
shift-invariant filter** (Gaussian, box footprint, motion, derivative, Laplacian, unsharp, band-pass) is an
exact per-unit edit `A2_i *= |G(w_i)|, c1_i += arg G(w_i)`. Depth destroys this: after standard training the
units of a deep SIREN swing 9-18 radians through the sine and are broadband, no per-unit scalar blurs them,
and a weight-only harmonic index predicts which units fail. The price of keeping the dictionary programmable is
about 7-10 dB at matched parameters in the compression regime (tens of dB when networks interpolate small images) because the
compression that depth buys comes from the same harmonics; it should be small where the capacity lives in an
encoder, as in arbitrary-scale super-resolution decoders.

The study started from the question whether a frontier LLM can read and edit INR weights ("weight-space
programming", Takehi et al. 2026). That measurement is kept as an optional later stage (`studies/01-sandbox`,
harness in `wsinr/harness/`); the research line is the programs themselves.

| Stage | Question | Status |
| --- | --- | --- |
| `studies/00-pilot` | tool-free LLM probes: algebra yes, reading no | done |
| `studies/01-sandbox` | sandbox-protocol LLM measurement | frozen, deferred |
| `studies/02-programs` | programs by layer and depth; harmonic failure mechanism; readout ceilings | done |
| `studies/03-width-vs-depth` | the price of programmability; LSI completeness | done |
| `studies/04-editable-training` | swing-bound training: a monotone fit-vs-programmability curve (two datasets, two seeds) | done |
| `studies/06-theory` | readout-programmability characterisation; validated unit statistics; the error law in the harmonic index | done |
| `studies/07-weight-space-augmentation` | exact programs as data augmentation for weight-space classifiers | in progress |
| `studies/08-antialiasing` | prefiltered rendering as a weight edit vs supersampling | done |
| `studies/09-natural-images` | the same results on natural crops; interpolation vs compression regime | done |
| `studies/10-gpu-scale` | 256 x 256 Kodak, matched pairs up to 33k weights; 10k-zoo augmentation grid (`scripts/gpu/run_scale.sh`) | protocol frozen, awaiting GPU |

## Tasks

| Level | Tasks | Closed form in weight space |
| --- | --- | --- |
| 0 | flip_lr, flip_ud, transpose, rot90, integer shifts, negative, brightness, contrast | yes: layer 1 (`A1' = R A1`, `c1' = c1 + t A1`) or the output layer (`A3' = a A3`, `c3' = a c3 + b`) |
| 1 | rotate_30, zoom_1.5, zoom_0.7 | exact in function space, target sampled |
| 2 | dilate, erode, blur_1, threshold | none known |

Reading: name the digit stored in a network from its weights alone; draw it.

## Layout

| Path | Content |
| --- | --- |
| `wsinr/siren.py` | SIREN parameters, NumPy forward pass, effective form `h = sin(h A + c)` |
| `wsinr/oracle.py` | closed-form edits and function-preserving symmetries |
| `wsinr/tasks.py` | task registry: target transform, oracle, validity mask, instruction text |
| `wsinr/prompt.py`, `wsinr/edits.py` | prompts in the Sparks style; the `edits` mini-language and its interpreter |
| `wsinr/zoo.py`, `wsinr/zoo_np.py`, `wsinr/baselines/` | batched SIREN fitting (PyTorch or NumPy), Adam fine-tuning, frozen-dictionary least squares |
| `wsinr/align.py` | permutation and sign alignment of two SIRENs |
| `wsinr/harness/` | `run.py` tool-free one-reply runs; `kernel.py` the restricted Python kernel; `forward.py` the budgeted forward-pass tool; `session.py` the multi-turn driver and backends |
| `scripts/` | `build_zoo.py`, `eval_baselines.py`, `make_prompts.py`, `score.py` |
| `studies/` | frozen protocols and predictions, one folder per stage |

## Usage

```bash
pip install -r requirements.txt
python scripts/build_zoo.py --mnist Data/mnist --n 200 --steps 2000 --out zoo/mnist_s32
python scripts/eval_baselines.py --zoo zoo/mnist_s32 --n 50 --out results/baselines.json
python scripts/make_prompts.py --zoo zoo/mnist_s32 --n 3 --tasks flip_lr negative dilate --out prompts/pilot
ANTHROPIC_API_KEY=... python -m wsinr.harness.run --prompts prompts/pilot --backend anthropic --model claude-opus-5-5 --effort high
python scripts/score.py --prompts prompts/pilot --out results/pilot_scores.json
```

`Data/mnist` holds the four original idx `.gz` files.

## Protocols

| Protocol | What the model has | Script |
| --- | --- | --- |
| tool-free | the weights as text, one reply, an `edits` block | `wsinr/harness/run.py` |
| sandbox-noforward | a persistent Python kernel over the weights (no autograd, optimizers, least squares or fitting; `sin`/`cos` refused so the network cannot be evaluated) | `scripts/run_sandbox.py --protocol sandbox-noforward` |
| sandbox-forward | the kernel plus a forward-pass tool with a budget of `--passes` requests (render the grid, or probe up to 256 points with hidden activations) | `scripts/run_sandbox.py --protocol sandbox-forward --passes k` |

Every session is logged turn by turn (`turns/tNNN/{prompt.txt,answer.md,run.json}`, `tool_calls.jsonl`, `state.json`, `final_answer.md`, `work/edited/*.npy`), after the audit format of the Sparks logs. Backends: the Anthropic API (`ANTHROPIC_API_KEY`) and any OpenAI-compatible endpoint (`OPENAI_API_KEY`, `OPENAI_BASE_URL`). Reference rows for every task come from `scripts/eval_baselines.py`: no edit, closed-form oracle where one exists, the frozen-dictionary bound (least squares on the readout only), and Adam fine-tuning at 10/50/200 steps, which is the budget-matched comparison for `--passes`.

```bash
python scripts/run_sandbox.py --zoo zoo/mnist_s32 --n 3 --tasks dilate blur_1 --protocol sandbox-noforward \
    --backend anthropic --model claude-sonnet-5-5 --max-runs 30 --out runs/sb0
python scripts/score.py --prompts runs/sb0 --out results/sb0_scores.json
```

## Conventions

Coordinates are `p = (y, x)` with `y` the row axis (top -1, bottom +1) and `x` the column axis; images are rendered on `linspace(-1, 1, 28)` as in the SIREN and NFN code. The model sees the effective weights `A_k = 30 W_k`, `c_k = 30 b_k` for sine layers, so that the first layer is a bank of plane-wave frequencies. Edits apply to the full-precision network; the model sees 4 decimals.

## History

This repository previously held an exploratory study of INR fitting dynamics (whiteboard reset, May 2026); its controls are summarised in `docs/lessons_from_para_space.md` and its code remains in the history before this study began.

## Acknowledgements

Prompt structure, the `edits` language and the no-tool protocol follow the released experiment logs of Takehi et al. (CC BY 4.0). The SIREN configuration (2-32-32-1, omega 30) is that of the NFN SIREN zoo (Zhou et al., 2023), whose dilate/contrast editing tasks this benchmark includes. BSD 3-Clause licence.
