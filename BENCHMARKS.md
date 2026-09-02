# Benchmarks

Headline metric: **`sp4_bulk_rmse`** — RMSE of the bulk magnetization ⟨m⟩(t) over a
100-step autoregressive rollout against the SP4 reference. SP4 is doubly held out:
its applied field is more negative in x than any training sample, **and its mesh is
100×25 cells against the training set's 256×256**, so the number is a joint test of
field extrapolation and mesh invariance.

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

`data/val/<variant>` holds one trajectory each, every one on a mesh the model never
trained on. They are excluded from the val loss — none could share a batch with it.
The val set itself is 128 `fixed_geo` trajectories (256 until 2026-08-28; the first 128 were
kept, so every val number quoted here was computed on trajectories that still exist).

| variant | mesh (cells) | vs. training mesh | bulk RMSE | corr (mean / final) |
|---|---|---|---|---|
| `sp4_large` | 3000 x 3000 | **137x** the area, OOD field **and** reversal | 0.0315 | 0.9886 / 0.9885 |
| `sp4`   | 100 x 25     | 0.4x the area, OOD field | 0.0179 | 0.9964 / 0.9977 |
| `large` | 2000 x 2000  | **61x** the area, in-distribution field | **0.0076** | 0.9997 / 0.9973 |

`sp4_large` (added 2026-08-27) is the headline: SP4's field and s-state init on a
3000x3000 mesh, so it stacks every axis of transfer at once. It is the hardest of the
three for both models.

`large` scores *better* than `sp4` despite the 61x bigger grid, which is the expected
ordering once you separate the two things being tested: `large` probes mesh transfer
alone (its field is drawn from the training distribution and it never switches),
while `sp4` probes field extrapolation *and* a reversal event on top of a mesh change.
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

**Verdict: the mechanism works in-distribution and fails out-of-distribution.**

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
is that SP4's error is dominated by *field* extrapolation rather than by drift into the
model's own error distribution — pushforward only addresses the latter — but that is a
hypothesis, not something these runs establish.

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
10.9 GB), i.e. field extrapolation + a reversal event + 137x the training area at once.
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
| `sp4_large` | OOD | **0.0757** | 0.0853 | +13% |
| `sp4` | OOD | **0.0565** | 0.1027 | +82% |

Pushforward improves pointwise rollout accuracy wherever the applied field resembles
training and degrades it where the field is out of distribution. That split is consistent
with the amplification measurements: it cuts self-drift amplification 2.1x in-distribution
(67.7x -> 32.1x) and not at all on SP4 (26.8x -> 29.2x), where the error is driven by
field extrapolation instead.


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
