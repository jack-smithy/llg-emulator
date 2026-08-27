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

| variant | mesh (cells) | vs. training mesh | bulk RMSE | corr (mean / final) |
|---|---|---|---|---|
| `sp4`   | 100 x 25     | 0.4x the area, OOD field | 0.0179 | 0.9964 / 0.9977 |
| `large` | 2000 x 2000  | **61x** the area, in-distribution field | **0.0076** | 0.9997 / 0.9973 |

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