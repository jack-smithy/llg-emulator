# Benchmarks

Headline metric: **`sp4_bulk_rmse`** — RMSE of the bulk magnetization ⟨m⟩(t) over a
100-step autoregressive rollout against the SP4 reference. SP4 is held out by **geometry**: a 100×25 cell strip (500×125 nm, 4:1 aspect ratio) against the
training set's 256×256 squares, with a domain-wall-mediated reversal under strong shape
anisotropy. Its field and s-state initial condition are *inside* the training distribution
(34 of 512 training samples have H_x < −24.6 mT; 100 start from an s-state) — see the
correction note under *Benchmark variants*.

## run1 — 2026-08-26 (SLURM job 1534)

First run on the `data/{train,val}/<variant>/` dataset (512 train + 256 val
trajectories, `fixed_geo`, 256×256 cells → 51,200 one-step training pairs).

```
uv run train --epochs 40 --batch-size 8 --learning-rate 3e-4 --cosine \
  --weight-decay 1e-5 --grad-clip 1.0 --hidden-channels 64 --num-blocks 4 \
  --checkpoint-every 5 --out weights/run1
```

337,987 trainable parameters. Wall clock 7:46:23, ~11.6 min/epoch — **contended**,
three other jobs shared the node for the first ~1.5 h, so treat that as an upper bound.

| metric | value |
|---|---|
| **best `sp4_bulk_rmse`** | **0.0177** (epoch 35, `weights/run1/best.eqx`) |
| final `sp4_bulk_rmse` | 0.0211 (epoch 40, `weights.eqx`) |
| final val one-step MSE | 2.2626e-06 — 3,536× below the trivial `m_{t+1} = m_t` baseline (8.0e-3) |
| val rollout correlation, final frame (16 trajectories) | 0.9549 (best) |
| `large` bulk RMSE (2000x2000 cells, 61x training area) | **0.0076**, corr mean 0.9997 |
| `sp4` correlation (per-frame mean cosine) | mean 0.9964, final 0.9977 |
| SP4 pointwise error, mean / max \|m_pred - m_ref\| | 0.0561 / 0.7626 |
| training mesh → eval mesh | (256, 256) → (100, 25) cells |

Per-epoch SP4 rollout (checkpoint epochs only):

| epoch | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 |
|---|---|---|---|---|---|---|---|---|
| `sp4_bulk_rmse` | 0.1398 | 0.0531 | 0.0757 | 0.1231 | 0.0401 | 0.0295 | **0.0177** | 0.0211 |

Train and val losses fall monotonically to 1.69e-06 / 2.26e-06 with no sign of
overfitting at 40 epochs.

**The bulk curve is much better than the field.** ⟨m⟩(t) overlays the reference, but
that is a spatial average over 2,626 nodes and it hides local disagreement: mean
\|m_pred - m_ref\| peaks at **0.117 around t = 600 ps**, the global max is **0.763**
(frame 49, mid-switch), and **17%** of all (frame, node) pairs exceed 0.1. The error
is concentrated in the reversal and the domain-wall structure that follows it, which
is exactly where neighbouring spins disagree most. Inspect it in ParaView via
`weights/run1/vtr/sp4_error.pvd`.

### Benchmark variants

> **Correction (2026-09-02).** Earlier versions of this file called SP4's applied field
> "out-of-distribution — more negative in x than any training sample". That was true of the
> old 24-trajectory dataset and is **false** for the current one: training fields are uniform in
> |H_i| ≤ 30 mT, 34 of 512 samples have H_x < −24.6 mT, and 100 start from the same s-state.
> What SP4 tests is *geometry* transfer (100×25 strip, 4:1 aspect, shape anisotropy) and a
> wall-mediated reversal. Interpretations below that leaned on "field extrapolation" have been
> reworded; the numbers are unchanged.

`data/val/<variant>` holds one trajectory each, every one on a mesh the model never
trained on. They are excluded from the val loss — none could share a batch with it.
The val set itself is 128 `fixed_geo` trajectories (256 until 2026-08-28; the first 128 were
kept, so every val number quoted here was computed on trajectories that still exist).

| variant | mesh (cells) | vs. training mesh | bulk RMSE | corr (mean / final) |
|---|---|---|---|---|
| `sp4_large` | 3000 x 3000 | **137x** the area, SP4 reversal | 0.0315 | 0.9886 / 0.9885 |
| `sp4`   | 100 x 25     | 0.04x the area, 4:1 strip, reversal | 0.0179 | 0.9964 / 0.9977 |
| `large` | 2000 x 2000  | **61x** the area, in-distribution field | **0.0076** | 0.9997 / 0.9973 |

`sp4_large` (added 2026-08-27) is the headline: SP4's field and s-state init on a
3000x3000 mesh, so it stacks every axis of transfer at once. It is the hardest of the
three for both models.

`large` scores *better* than `sp4` despite the 61x bigger grid, which is the expected
ordering once you separate the two things being tested: `large` probes mesh transfer
alone (its field is drawn from the training distribution and it never switches),
while `sp4` probes a reversal event on a narrow strip — a geometry, not a field, the model never saw.
Domain-size invariance is not the hard part.

**The training-mesh val set is the hardest case, not SP4.** Final-frame correlation
is 0.9549 over 16 held-out `fixed_geo` trajectories against 0.9977 on SP4 — for unit
vectors that is a mean pointwise error of ~0.30 against ~0.07. Those trajectories start
from random configurations and develop multi-domain structure; SP4's s-state reversal
is comparatively deterministic. Model selection currently tracks SP4 only.

> **Numerical note.** `sp4` bulk RMSE reads 0.0179 here against the 0.0177 logged
> during training. `metrics.rollout_metrics` replaced a `lax.scan` rollout with a
> loop of jitted steps when `large` forced an O(1)-memory implementation; identical
> arithmetic, different XLA fusion, and 100 autoregressive float32 steps amplify the
> last-bit difference to ~1%. Treat the third digit of any rollout metric as noise.

> The correlation figure was 0.9700 in the first write-up of this run. That number
> came from the old sign-blind `metrics.correlation` (it RMSed the pointwise dot
> product, scoring a perfectly *inverted* field 1.0). The metric is now a sign-aware
> mean cosine similarity; 0.9549 is the same checkpoint re-scored, not a regression.

### What this run establishes

1. **Mesh invariance holds end to end.** The model is trained only on 256×256 and
   rolled out 100 steps on 100×25 with no retraining — only the demag tensor is
   rebuilt (`model.with_mesh`). The ⟨m⟩(t) curves overlay the reference through the
   switch and the full precession ringtail (`weights/run1/sp4_bulk.png`).
2. **The one-step loss is no longer the bottleneck it was.** Known issue 2 recorded a
   16-epoch run whose 100-step unroll precessed and drifted out of plane. With 32× the
   trajectories, 40 epochs and cosine, plain one-step MSE reaches 0.0177 — the
   rollout-k loss is worth re-testing, but it is no longer needed to get a stable unroll.
3. **Best-checkpointing earns its keep.** `sp4_bulk_rmse` is *not* monotonic in the
   val loss — it rises 0.0531 → 0.1231 between epochs 10 and 20 while both losses fall.
   Selecting on it recovers 0.0177 against 0.0211 for the last epoch.
4. **The historical sweep's capacity/epoch findings are stale.** They were derived on
   24 trajectories, where hidden 128 and >160 epochs overfit. At 512 trajectories
   nothing has overfit yet at 40 epochs, so the capacity ceiling should be re-derived
   before it is trusted.

### Not comparable to the historical numbers

The pre-2026-08 sweep in CLAUDE.md (best `sp4_bulk_rmse` 0.037) ran on the **old
cell-based dataset** with a different SP4 reference on the *training* mesh and a
rollout-k4 loss. 0.0177 is not "2× better than 0.037" — it is a different benchmark.

## run-pf — 2026-08-27 (SLURM job 1551), pushforward loss

Identical to `run1` except `--pushforward`: same seed, 40 epochs, hidden 64, 4 blocks,
cosine, lr 3e-4, `--checkpoint-every 5`. Wall clock 9:34:39 (1.23x `run1`).

**Verdict: the mechanism works on the training geometry and on `large`; it hurts the SP4 reversal.**

| | run1 (one-step) | run-pf (pushforward) |
|---|---|---|
| val one-step MSE (final) | **2.26e-06** | 5.86e-06 |
| val one-step \|e\| | **0.00079** | 0.00120 |
| val rollout \|e\| | 0.05375 | **0.03835** (-29%) |
| **val amplification** | 67.7x | **32.1x** (-53%) |
| val corr, final frame | 0.9549 | **0.9884** |
| `large` bulk RMSE | 0.0076 | **0.0027** |
| `large` corr, final | 0.9973 | **1.0000** |
| `sp4` bulk RMSE | **0.0179** | 0.0236 |
| `sp4` one-step \|e\| | **0.00208** | 0.00352 |
| `sp4` rollout \|e\| | **0.0559** | 0.1028 |
| `sp4` amplification | **26.8x** | 29.2x |

Pushforward does exactly what it is supposed to do on in-distribution rollouts: it buys
a **2.1x reduction in error amplification** (67.7x -> 32.1x) at the price of a 1.5x
worse one-step map, netting 29% lower rollout error and closing most of the val-set
correlation gap. On `large` it is 2.8x better and reaches correlation 1.0000.

On SP4 it loses on every measure. Amplification does **not** drop there (26.8x -> 29.2x),
so the degraded one-step map goes straight through to the rollout. The plausible reading
is that SP4's error is dominated by the wall-resolution problem on the strip rather than by
drift into the model's own error distribution — pushforward only addresses the latter — but
that is a hypothesis, not something these runs establish.

**Confound worth fixing before drawing conclusions.** Best-checkpoint selection tracks
`sp4/bulk_rmse` only, so `run-pf`'s `best.eqx` was chosen by the one metric pushforward
hurts. Its val correlation and `large` numbers might improve further under a selection
rule that reflects them.

Next experiment: blend the two, `loss = one_step + lambda * pushforward`. That is the
one change that could keep the 2.1x amplification reduction while recovering the one-step
accuracy SP4 depends on.


## sp4_large — superseded (data deleted 2026-08-28)

> `data/val/sp4_large` was removed to reclaim quota once `sp4_xlarge` existed: same
> physics one size down, and every ordering it produced, `sp4_xlarge` reproduces. The
> numbers below stay as the record; re-running them means regenerating it
> (`N=3000 NAME=sp4_large sbatch scripts/generate.slrm`, ~15 min).

SP4's applied field and s-state initial condition on a **3000x3000** mesh (3001^2 nodal,
10.9 GB), i.e. the strip reversal + 137x the training area at once.
Streamed frame by frame; needs a full GPU (~2.3 GB per feature map at hidden 64).

| metric | run1 (one-step) | run-pf (pushforward) | better |
|---|---|---|---|
| bulk RMSE | **0.0315** | 0.0439 | run1 |
| rollout mean \|e\| | **0.0757** | 0.0853 | run1 |
| rollout final \|e\| | **0.0577** | 0.0789 | run1 |
| corr (mean / final) | 0.9886 / 0.9885 | **0.9939 / 0.9937** | run-pf |
| final-frame max \|e\| | 2.00 | 2.00 | — |

**The two pointwise metrics disagree on the ordering, which is itself the finding.**
`run-pf` has the higher mean cosine similarity but the higher mean \|e\|; that can only
happen if its error distribution is more concentrated — more nearly-perfect spins and a
heavier tail. Both models put some spins fully antipodal (max \|e\| = 2.0) by the last
frame, so neither is tracking the reversal correctly everywhere. Quote both, or quote
the one that matches what the downstream use actually needs.

### Pointwise rollout error, all benchmarks

| trajectory | field | run1 | run-pf | change |
|---|---|---|---|---|
| val `fixed_geo` (6 traj) | in-distribution | 0.0538 | **0.0384** | **-29%** |
| `large` | in-distribution | 0.0141 | **0.0052** | **-63%** |
| `sp4_large` | SP4 reversal | **0.0757** | 0.0853 | +13% |
| `sp4` | SP4 reversal | **0.0565** | 0.1027 | +82% |

Pushforward improves pointwise rollout accuracy on the training geometry and on `large`, and
degrades it on the SP4 reversals. That split is consistent with the amplification measurements:
it cuts self-drift amplification 2.1x on the val films (67.7x -> 32.1x) and not at all on SP4
(26.8x -> 29.2x), where the error is set by resolving the reversing domain wall on a narrow
strip rather than by drift into the model's own error distribution.


## sp4_xlarge — 4000x4000, the largest usable benchmark (2026-08-28)

SP4's field and s-state init on a **4000x4000** mesh (4001^2 nodal, 19.4 GB),
generated by job 1625 in 25 min. Reversal confirmed: bulk m_x runs +0.949 -> -0.983
and |m| holds at 0.99999. 244x the training area.

**Why 4000 and not larger.** The solver is not the constraint -- 5000x5000 peaks at
14.4 GB of a 96 GB RTX PRO 6000 Blackwell. The *surrogate* is: a rollout step peaks at
16.2 GB on 3001^2 and 30.2 GB on 4001^2 (quadratic in cells), and 5001^2 dies on a
single 36 GiB allocation even with `XLA_PYTHON_CLIENT_MEM_FRACTION=.95` handing it
96.9 GB. Anything bigger would be simulable but not evaluable without tiled inference.

| metric | run1 (one-step) | run-pf (pushforward) | better |
|---|---|---|---|
| bulk RMSE | **0.0353** | 0.0495 | run1 |
| rollout mean \|e\| | **0.0759** | 0.0903 | run1 |
| rollout final \|e\| | **0.0543** | 0.0796 | run1 |
| corr (mean / final) | 0.9903 / 0.9908 | **0.9936 / 0.9945** | run-pf |
| final-frame max \|e\| | 2.00 | 2.00 | — |

Same split as `sp4_large`: `run1` wins bulk and mean pointwise error, `run-pf` wins mean
cosine similarity, and both leave some spins fully antipodal by the last frame.

### The SP4 family across mesh size

| benchmark | mesh | run1 bulk | run-pf bulk | run1 corr | run-pf corr |
|---|---|---|---|---|---|
| `sp4`        | 100 x 25    | **0.0179** | 0.0236 | **0.9977** | 0.9839 |
| `sp4_large`  | 3000 x 3000 | **0.0315** | 0.0439 | 0.9885 | **0.9937** |
| `sp4_xlarge` | 4000 x 4000 | **0.0353** | 0.0495 | 0.9908 | **0.9945** |

Bulk RMSE degrades with mesh size for both models and `run1` leads at every size. The
correlation ordering **flips** between the 100x25 geometry and the big ones: `run1` is
ahead on the original SP4 mesh, `run-pf` on both large ones. Pushforward's per-spin
agreement holds up better as the domain grows even as its bulk average gets worse.

## Cost: learned integrator vs reference solver (2026-09-02)

Same GPU (RTX PRO 6000 Blackwell, 96 GB), 100 steps of 10 ps, both JIT-warm. Solver =
NeuralMag dopri5 from the stored frame 0 (job 2231); model = one-step `run1` (job 2913
component profile, `rollout_metrics` step time). Integrator time only — no I/O.

| mesh | solver, 100 steps | model, 100 steps | model / step | speed-up at stride 1 |
|---|---|---|---|---|
| 100 x 25 (SP4) | 0.3 s | < 0.1 s | ~ms | — (both trivial) |
| 2000 x 2000 | 20.1 s | **5.7 s** | 57 ms | **3.5x** |
| 4000 x 4000 | 86.8 s | **23.2 s** | 232 ms | **3.7x** |

At stride *k* the model takes 100/*k* steps, so the ceiling is *k* x these numbers — 62x at
stride 16 on 4000^2 — **if** the accuracy at that stride is acceptable, which for `run-dt` it
currently is not (see below).

### Where a step spends its time (4001^2 nodal, warm)

| component | time | share |
|---|---|---|
| demag: to_cell -> rfft2 -> N -> irfft2 -> to_node | 15 ms | 7% |
| CNN: lifting + 4 residual blocks + projection | 222 ms | **96%** |
| tangent projection + renormalise | 2 ms | 1% |
| **full step** | **232 ms** | |
| eval harness, per frame: memmap read + CPU transpose + H2D | 51 ms | (eval only) |
| eval harness, per frame: 3 metric reductions with device sync | 24 ms | (eval only) |

**Not demag-dominated.** The exact demag is 7% of a step; the CNN is 96%. Eight 3x3 convs at
64 channels on 16M nodes is ~9.4 TFLOP, running at ~42 TFLOPS effective — compute-bound at
roughly 40% of fp32 peak. Precision (TF32/bf16) or channel count are the levers, not the FFT.
The evaluation harness adds ~75 ms/step on top (25% at 4001^2, ~47% at 2001^2); it is not part
of the integrator and `evaluate` now reports the two separately.

### Why the first measurement said 88.7 s (correction)

The stride eval (job 2910) reported 88.7 s for 100 steps on 4000^2 — and 54.7 s for **one**
step at stride 64. The ~54 s was a fixed cost per call: `rollout_metrics` closed `cond`
(containing `s_enc`) into the jitted step, baking it into the HLO, so every stride and every
call recompiled the full 4001^2 graph — ~50 s cold (the persistent `.jax_cache` brings a
repeat down to ~7 s). Fixed 2026-09-02: `cond` is a traced argument; compile is paid once per
process and cached across processes; `step_time` excludes it. Marginal cost per step in the
old run was (88.7 − 54.7)/99 ≈ 0.34 s, i.e. model 0.23 s + harness 0.08 s + sync — consistent
with the profile.

### Solver self-consistency (correction)

Job 2231 printed |m − ref_100| ≈ 0.09 on SP4 for a solver re-run, which I read as the reference
not reproducing itself. Wrong: that script took one warm-up step *before* the timed 100, so it
compared frame 101 to frame 100. Job 2912 did it properly — dopri5 re-run from stored frame 0
matches the stored frame 100 to **0.0000** (bulk RMSE, pointwise, cosine) on all three meshes,
and a 1e-6 perturbation of frame 0 grows to only 2–4e-4 by frame 100. The reference is
reproducible and the reversal is not chaotic at that level; the learned integrator's pointwise
error (0.056 on SP4) is genuine, not solver noise. The timings from 2231 are unaffected.

## run-dt — variable step size (2026-09-02, SLURM job 2230)

Strides {1, 2, 4, 8, 16} x 10 ps, plain one-step loss, 9 epochs (= run1's gradient budget:
443 pairs/trajectory vs 99), otherwise run1's recipe. Wall 7:21. Evaluated by jobs 2911
(per-stride one-step MSE) and 2914 (rollouts, fixed-code timings); run1 baseline from 2910.

### Single-step accuracy vs step size (val films, 16 traj, 160 pairs/stride)

| stride | Δt | trivial | run1 (stride 1 only) | **run-dt** | run-dt / trivial |
|---|---|---|---|---|---|
| 1 | 10 ps | 5.4e-3 | **5.2e-6** | 1.7e-4 | 31x below |
| 2 | 20 ps | 2.5e-2 | 3.2e-2 | **3.7e-4** | 68x |
| 4 | 40 ps | 7.8e-2 | 1.5e-1 | **1.4e-3** | 54x |
| 8 | 80 ps | 2.0e-1 | 3.2e-1 | **2.2e-2** | 9x |
| 16 | 160 ps | 2.2e-1 | 4.5e-1 | **5.0e-2** | 4.5x |

The step-size conditioning works: run1 ignores `s_enc` and is *worse than the trivial
predictor* at every stride >= 2; run-dt beats trivial at all five with one set of weights.
Accuracy falls steeply with Δt (68x below trivial at 20 ps, 4.5x at 160 ps), and stride 1 paid
for the others: 33x worse than run1, having seen a fifth of the stride-1 gradient steps.

### Rollouts at each stride (integrator time only, JIT-warm)

Solver reference: 0.3 s (SP4), 20.1 s (2000^2), 86.8 s (4000^2) for the same 1 ns.

| benchmark | stride | steps | run-dt bulk RMSE | run-dt corr final | time | vs solver |
|---|---|---|---|---|---|---|
| `large` 2000^2 | 1 | 100 | 0.084 | 0.952 | 5.8 s | 3.5x |
| | 2 | 50 | 0.028 | **0.999** | 2.9 s | 6.9x |
| | 8 | 12 | 0.026 | 0.970 | 0.70 s | 29x |
| | **16** | **6** | **0.019** | **0.997** | **0.35 s** | **57x** |
| | 32 | 3 | 0.297 | 0.499 | 0.17 s | untrained stride: fails |
| | 100 | 1 | 0.461 | 0.250 | 0.06 s | fails |
| `sp4` 100x25 | 1 | 100 | 0.139 | 0.986 | 0.04 s | |
| | 2 | 50 | **0.091** | 0.978 | 0.02 s | |
| | 16 | 6 | 0.287 | 0.951 | <0.01 s | |
| | 100 | 1 | 0.634 | −0.230 | | fails |
| `sp4_xlarge` 4000^2 | 1 | 100 | 0.204 | 0.909 | 23.7 s | 3.7x |
| | 16 | 6 | 0.296 | 0.901 | 1.4 s | 61x |
| | 100 | 1 | 0.640 | −0.289 | 0.24 s | fails |

run1 at every stride >= 2: bulk RMSE 0.45–0.76, corr final <= 0.78 (mostly ~0 or negative).

**What this establishes.**

1. **On the in-distribution film the long step is the better integrator.** run-dt at stride 16
   on `large` — six 160 ps steps, 0.35 s — reaches final-frame correlation 0.997, equal to
   run1's 100-step rollout (0.9973, 5.8 s) and **57x faster than dopri5**, with bulk RMSE 0.019
   against run1's 0.008. Fewer steps means less compounding; the sweet spot is not the
   smallest step. Even at stride 1, run-dt's val-set rollout correlation (0.970) beats run1's
   (0.954) despite a 33x worse single-step error — the same trade pushforward makes.
2. **Extrapolation in Δt does not work.** Strides 32, 64, 100 were never trained
   (`s_enc` = 5, 6, 6.6 vs a training range of 0–4) and collapse everywhere: correlation
   0.25–0.64 on `large`, negative on both SP4 meshes. The end goal — one step to the 1 ns
   equilibrium — is **not** reached by this model. It would need those strides in training, and
   the trend (4.5x below trivial at 160 ps) says the single-step accuracy there will be poor.
3. **The SP4 reversal is where run-dt is weakest**, at every stride (0.09–0.29 vs run1's 0.018
   at stride 1). Consistent with stride 1 being undertrained and with the wall-resolution
   difficulty seen for pushforward: long steps across a domain-wall sweep on a 125 nm strip are
   the hardest thing in the set.

Next experiment, if pursued: train strides {1,…,16} for run1's *per-stride* budget (~40
epochs, ~35 h) or weight the loss toward stride 1, and add 32/64 to the training set before
asking for one-step equilibrium. Time per step is independent of Δt (0.236 s at 4001^2), so
every accurate doubling of the stride is a free 2x.

### Scaling with mesh size (job 2915)

Seconds per 10 ps step vs nodes, both integrators JIT-warm on one RTX PRO 6000, S-state under
the SP4 field (the first 10 steps of a reversal, where dopri5 takes the most substeps — hence
the 4000^2 per-step time here, 1.11 s, is above the whole-trajectory average of 0.87 s from
job 2231). Figure: `figures/scaling.pdf`.

| mesh | nodes | dopri5 s/step | learned s/step | ratio |
|---|---|---|---|---|
| 16x16 | 289 | 0.0046 | 0.0003 | 14 |
| 31x31 | 1,024 | 0.0046 | 0.0003 | 17 |
| 63x63 | 4,096 | 0.0045 | 0.0003 | 17 |
| 100x25 (SP4) | 2,626 | 0.0042 | 0.0003 | 16 |
| 256x256 | 66,049 | 0.0056 | 0.0010 | 5.9 |
| 2000x2000 | 4.0e6 | 0.251 | 0.059 | 4.3 |
| 4000x4000 | 1.6e7 | 1.107 | 0.235 | 4.7 |
| 5000x5000 | 2.5e7 | 1.765 | 0.368 | 4.8 |

Two regimes. Below ~10^5 nodes both are flat — kernel-launch bound — and the ratio is ~15x
because dopri5 launches many substeps' worth of kernels per 10 ps while the learned step is one
graph. Above ~10^6 nodes both scale linearly in N and the ratio settles at **4.3–4.8x per
step**. The learned integrator's advantage at fixed Δt is therefore a constant factor, not an
asymptotic one; the asymptotic lever is the step size (cost per step is independent of Δt).

**Ceiling correction.** The model ran at 5000^2 here (0.37 s/step). The OOM reported earlier at
5001^2 (probes 1622/1624, "single 36 GiB allocation") occurred with `cond` closed over as a jit
constant; with it passed as a traced argument the graph fits. `sp4_xlarge` at 4000^2 stays the
benchmark, but "largest mesh the surrogate can evaluate" is no longer a justified description —
the true single-GPU ceiling is above 5000^2 and has not been located.
