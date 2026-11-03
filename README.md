# Para_space: weight-space programming for INRs

Can a language model read and edit the weights of an implicit neural representation (INR)?

[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi, Teoh, Zhang, Khattab, 2026) shows a frontier LLM naming the game a policy network plays from its raw weights, repairing a stalled policy with a one-shot weight edit, and removing a repetition loop from Qwen3.5-0.8B by projecting a direction out of 131,072 weights. Those networks read spatially indexed inputs (grid cells, physics state) whose weights carry meaning one can see. An INR has none of that: a SIREN maps two coordinates to a grey value through sinusoids, its weight space is highly symmetric (unit permutations, sign flips), and learned weight-space networks still read it poorly. INRs are therefore a sharp test of whether weight-space programming is more than reading spatial layouts.

This repository is the study code: a SIREN zoo, a graded set of edit tasks with closed-form oracles where they exist, a tool-free one-reply LLM harness after the Sparks protocol, scoring, and gradient baselines.

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
| `wsinr/zoo.py`, `wsinr/baselines/finetune.py` | batched SIREN fitting and Adam fine-tuning (PyTorch) |
| `wsinr/harness/run.py` | tool-free one-reply runs through Claude Code (`--tools ""`), the Anthropic API or an OpenAI-compatible API, with a run record per case |
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

## Conventions

Coordinates are `p = (y, x)` with `y` the row axis (top -1, bottom +1) and `x` the column axis; images are rendered on `linspace(-1, 1, 28)` as in the SIREN and NFN code. The model sees the effective weights `A_k = 30 W_k`, `c_k = 30 b_k` for sine layers, so that the first layer is a bank of plane-wave frequencies. Edits apply to the full-precision network; the model sees 4 decimals.

## History

This repository previously held an exploratory study of INR fitting dynamics (whiteboard reset, May 2026); its controls are summarised in `docs/lessons_from_para_space.md` and its code remains in the history before this study began.

## Acknowledgements

Prompt structure, the `edits` language and the no-tool protocol follow the released experiment logs of Takehi et al. (CC BY 4.0). The SIREN configuration (2-32-32-1, omega 30) is that of the NFN SIREN zoo (Zhou et al., 2023), whose dilate/contrast editing tasks this benchmark includes. BSD 3-Clause licence.
