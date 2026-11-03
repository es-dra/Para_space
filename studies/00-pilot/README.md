# Stage 00: feasibility probe (internal, 2026-11-03)

Protocol and frozen predictions: `protocol.json`. Editor: the model serving the Claude Code session, as 20 fresh
subagents told to use no tools beyond reading the prompt file and writing the answer (self-reported; not enforced
by a harness, so these numbers are a feasibility signal, not paper data). Zoo: 60 SIRENs (2-32-32-1, omega 30,
1,185 weights) on MNIST test images 0..59, 2,000 Adam steps, mean fit PSNR 64 dB (`zoo_index.json`).

## Outcomes

| Prediction | Outcome |
| --- | --- |
| P1 (level 0 within 2x oracle in >= 7/9; shifts partly) | Passed, stronger than predicted: 12/12 cases at oracle precision, including all three 4-pixel shifts (the editor computed `c1 -= (8/27) A1[1, :]` for 32 units by hand). |
| P2 (dilate beats no-edit in <= 1/3; never reaches 200-step fine-tune) | Half failed: 3/3 beat no-edit (MSE 0.036 / 0.059 / 0.069 against 0.052 / 0.081 / 0.105), but every answer used the same global knob (output gain x3 plus a bias), which the Sparks protocol itself rules out as "not what we are looking for". No answer attempted a spatial operation. The 200-step fine-tune bound is far below (see `../../results/baselines_n20.json`). |
| P3 (digit reading at or near chance, <= 2/5) | Held: 1/5 correct (true 7, 2, 1, 0, 4; guessed 3, 2, 2, 1, 8), self-reported confidence 12-18%, drawings worse than a blank frame on 4/5. The answers read first-layer frequencies (pure-y, pure-x, oblique plane waves) but cannot turn them into a shape. |

Scores: `pilot_edit_scores.json`, `pilot_read_scores.json`; the answers are in `answers/`.

## Reading

Level-0 edits are solved by knowing the closed form, not by reading the weights: the only numbers the editor needs
are the ones it copies from `A1` or `c3`. The tool-free one-reply setting therefore separates two abilities that the
Sparks post presents together: applying known weight-space algebra (the editor does this perfectly) and reading
what a network computes from its weights (at chance for a SIREN). Per the decision rule, the next stage is the
sandbox protocol (Python over the weights, no forward pass, then a small forward-pass budget), where reading and
non-local edits become possible in principle and the comparison with gradient fine-tuning is the one that matters.
