# Stage 06: theory (2026-11-05)

Protocol: `protocol.json` (frozen before running); scoring: `outcomes.json`; per-unit data: `results/`
(4,480 units from seven zoos: standard, swing bounds 0.5/1/2/4/8, L1 penalty 1e-3).

## T1. Which dictionaries admit readout-only LSI programs

**Proposition.** Let $f=\sum_j a_j\phi_j+b$ with fixed dictionary functions $\phi_j$. Readout-only edits
$a_j\to s_j a_j$ implement every isotropic Gaussian blur exactly, for every readout, iff every $\phi_j$ has its
spectrum on a single radius; they implement every real LSI filter exactly iff every $\phi_j$ is a single sinusoid.
*Proof.* Plane waves are the eigenfunctions of LSI operators. A $\phi_j$ with spectral components at two gains
cannot be reproduced by one scalar; conversely a single-shell function is an eigenfunction of every isotropic
filter, and a single sinusoid of every LSI filter. $\square$

Consequences. (i) The one-sine-layer SIREN is the only depth at which the readout is LSI-programmable, and there
the phase edit $c_{1,i}\mathrel{+}=\arg G(\mathbf w_i)$ extends exactness to complex responses (stage 03).
(ii) A deeper unit $\sin(c_j+\sum_i a_{ij}\sin\theta_i)$ has spectrum on all integer combinations
$\sum_i n_i\mathbf w_i$; in the linear regime $a_{ij}\to0$ only the fundamentals survive, which is why the
*input-attenuation* program (an edit of $\mathbf A_2$, per input wave) is exact there while a readout-only edit
still is not: even a linear-regime unit is a sum of many plane waves, not one. Numerically: on the standard zoo no
unit has more than 33% of its energy within a 20% band of its mean frequency (median 23%); on the swing-0.5 zoo
the median is 25% (many waves, but linear), and on the one-sine-layer zoo it is 100% by construction.

## T2. Predicting the first-order program's error from the weights

The independent-phase model (inputs $\theta_i$ uniform and independent) gives closed forms for every unit of a
deep SIREN by the Jacobi--Anger expansion: mean $\sin c\prod_k J_0(a_k)$, second moment
$\tfrac12-\tfrac12\cos 2c\prod_k J_0(2a_k)$, fundamental amplitude along $\sin\theta_i$ equal to
$2J_1(a_i)\prod_{k\ne i}J_0(a_k)\cos c$. **The model is accurate**: the predicted variance is within a factor 1.5
of the measured variance for 92% of the 3,149 units with $H\ge1$ (T2d, passed), and Monte Carlo confirms the
fundamental fraction $F_j$ to 0.004.

**The predictor I registered was the wrong one.** I predicted that $1-F_j$ (the share of a unit's energy outside
its fundamentals) would rank the first-order program's per-unit error better than the harmonic index
$H_j=\sum_i|a_{ij}|$. It does not: pooled Spearman 0.54 for $1-F$ against 0.90 for $H$ (T2a, T2c failed). Within a
swing-bounded zoo, where every unit has the same $H$, $F$ mostly tracks the unit's phase $c_j$ through $\cos c_j$
and the program's error does not follow it. T2b half-held: units with $F\ge0.9$ have median error $7\times10^{-4}$
(passed), units with $F\le0.3$ have $0.26$ (predicted $>0.5$, failed).

What survives is an empirical law in $H$ alone, pooled over all 4,480 units:

| $H$ range | units | median relative error of the first-order blur program |
| --- | --- | --- |
| $[0,0.75)$ | 980 | $7\times10^{-4}$ |
| $[0.75,1.5)$ | 666 | $6\times10^{-5}$ |
| $[1.5,3)$ | 714 | $1.2\times10^{-3}$ |
| $[3,6)$ | 796 | $0.026$ |
| $[6,10)$ | 702 | $0.29$ |
| $[10,30)$ | 622 | $1.5$ (worse than no edit) |

with a pointwise guarantee that is loose but free: for a unit with swing $|\delta|\le H$,
$|\sin(c+\delta)-\sin c-\delta\cos c|\le H^2/2$, so the program's pointwise error is at most $H^2$ in units of the
sine's amplitude. The practical threshold is $H\approx3$: below it the program is essentially exact, above $6$ it
is useless. This is the statement the swing bound of stage 04 should be read against: $B=2$ sits in the exact
regime, $B=4$ on the shoulder, $B=8$ past it.

## Reading

The characterisation (T1) and the validated unit statistics are the theory the paper can carry. The failed
prediction is recorded as such; the lesson is that the obstacle to layer-local programs is the *amplitude* of the
nonlinearity, not the spectral shape of its output, and the right weight-only diagnostic is the simplest one.
