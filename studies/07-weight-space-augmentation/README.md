# Stage 07: exact programs as weight-space data augmentation (status 2026-11-07: blocked on the classifier)

Protocol (frozen): `protocol.json`. Data: `zoo/mnist_s32_train5k` (5,000 SIRENs, 2-32-32-1, MNIST train 0..4999)
and `zoo/mnist_s32_test1k` (MNIST test 1000..1999), built with `scripts/build_big_zoo.py`. Code:
`wsinr/wsl.py` (exact program augmentation in weight space, verified to render the transformed image; a
function-preserving permutation/sign control; two permutation-invariant classifiers written for this study;
an aligned-MLP baseline) and `scripts/train_wsl.py`, `scripts/run_wsl_grid.sh`, `scripts/analyze_wsl.py`.

**The grid has not been run, because no classifier available here learns the task well enough to measure an
augmentation effect.** Sanity runs on the first 1,000-1,500 networks (test set containing the training set,
so the numbers are upper bounds on generalisation):

| model | training nets | epochs | result |
| --- | --- | --- | --- |
| DeepSets2 (two-level invariant, d = 64) | 64 | 300 | memorises to 100% (the pipeline is correct) |
| DeepSets2 (d = 64) | 1,000 | 200 | 27.7% on 1,500 incl. the training set (underfits) |
| DeepSets2 (d = 128, lr 3e-3) | 1,000 | 300 | 31.7% (underfits) |
| NeuralGraph (edge-conditioned message passing, 3 rounds) | 1,200 | 40 | 12.6% (does not train) |
| aligned MLP | 1,000 | 200 | 100% on its training set, chance on the 500 unseen networks (memorises) |
| NP/HNP-equivariant NFN (own implementation, `wsinr/nfn_layers.py`, equivariance tested), c = 32 | 1,000 | 60 | 16.6% on the disjoint 1,000-network test zoo |
| same, no augmentation | 5,000 | 60 | 24.7% (train loss 1.72, still falling) |
| same, program augmentation | 5,000 | 60 | 21.3% (train loss 2.03): the augmented distribution is harder and the under-trained model gets worse, not better |

Published weight-space classifiers reach about 92% on MNIST INRs with NP/HNP-equivariant layers trained on
10,000 networks for tens of thousands of steps (NFN); our runs (at most 4,700 steps on CPU) stay in the
under-fitting regime where augmentation cannot help and in fact hurts (G1 would be scored as failed on this
evidence: -3.4 points). The right reading is not that the augmentation is useless but that the test is
underpowered: it needs the published recipe, GPU budget and a zoo of at least 10,000 networks. The experiment is kept, with its protocol, for the GPU phase using the
NFN library (`pip install nfn`, which needs a permission this session did not have) or a faithful
re-implementation of the NP layers. The exact augmentation itself is finished and tested; the question whether it
helps is open.
