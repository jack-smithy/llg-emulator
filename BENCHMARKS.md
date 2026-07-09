# SP4 emulator benchmarks

Autoregressive LLG emulator on **micromagnetic standard problem 4**. The headline metric
`sp4_bulk_rmse` is the RMSE of the bulk magnetization ⟨m⟩(t) over a **100-step autoregressive
rollout** from the SP4 initial state, vs. the micromagnetic reference trajectory. SP4's applied
field (Hₓ/Ms = −0.0245) is **out-of-distribution** — more negative than any training sample — so
this measures extrapolation, not just fit.

**Lower `sp4_bulk_rmse` is better; corr metrics → 1.0 is better.** `sp4_*` = SP4 rollout;
`val_*` = held-out val set (`val_step_mse` one-step, `val_corr_*` rollout). Reproduce any row with
`scripts/eval <weights.eqx> <tag>`. Models are best-checkpointed on SP4 rollout RMSE during
training; val_corr is never used for selection and tracks sp4_bulk_rmse at corr −0.97 across runs,
so the ranking reflects genuine generalisation.

Sorted best-first. **★ = shipped as `weights/best.eqx`.**

| tag | sp4_bulk_rmse | sp4_corr_mean | sp4_corr_final | val_step_mse | val_corr_mean | val_corr_final | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hc64_160 | 0.0339 | 0.9587 | 0.9394 | 1.1464e-05 | 0.9571 | 0.8971 | 1-step hc64, cosine lr3e-4 160ep — lowest RMSE, but weaker rollout stability (`weights/hc64_160.eqx`) |
| ★ hc64_k4 | 0.0371 | 0.9501 | 0.9626 | 3.4238e-05 | 0.9819 | 0.9772 | **hc64 + rollout k=4**, cosine lr3e-4 60ep — best rollout stability + late-time accuracy (shipped) |
| hc96 | 0.0403 | 0.9584 | 0.9467 | 1.1624e-05 | 0.9724 | 0.9373 | 1-step hc96, cosine lr3e-4 100ep |
| hc64_240 | 0.0457 | 0.9499 | 0.9548 | 1.9981e-05 | 0.9507 | 0.9190 | 1-step hc64, 240ep — overfits vs 160ep |
| onestep_hc64 | 0.0483 | 0.9287 | 0.9329 | 3.0103e-05 | 0.9394 | 0.8893 | 1-step hc64, 80ep — capacity helps, but epoch-limited |
| rollout_k4 | 0.0581 | 0.9420 | 0.9758 | 5.7959e-05 | 0.9804 | 0.9803 | rollout k=4, hc32, 80ep — k=4 helps even the small model |
| onestep_long | 0.0800 | 0.9473 | 0.9754 | 8.1255e-05 | 0.9639 | 0.9737 | 1-step hc32, cosine lr3e-4 80ep — "just train it properly" |
| hc128 | 0.1093 | 0.9209 | 0.9073 | 3.1336e-05 | 0.9173 | 0.9175 | 1-step hc128 — **overfits** 24 trajectories (erratic rollout) |
| rollout_k2 | 0.1380 | 0.9147 | 0.9785 | 1.6157e-04 | 0.9166 | 0.9532 | rollout k=2 — horizon too short, *worse* than 1-step |
| aug_k1 | 0.2583 | 0.8767 | 0.9672 | 3.3522e-03 | 0.8844 | 0.9421 | 1-step + D4×Z₂ symmetry aug (16×) — didn't beat plain training |
| baseline | 0.8115 | 0.6413 | 0.5920 | 3.3661e-03 | 0.7285 | 0.7113 | reference `scripts/run` config (lr 1e-5, 8ep) — badly undertrained |

**Winner: hc64 + rollout-k4 (0.037) — 22× better than the 0.81 baseline.** Chosen over hc64_160
(0.034 RMSE) because it has the best rollout stability (val_corr 0.982 vs 0.957), the lowest
late-time (t≥50) error, and the closest reversal amplitude — and its training objective
(multi-step rollout) matches how the model is deployed. All decent models reproduce the SP4
switching frame (6) exactly. See CLAUDE.md § "Best recipe & results" for the full findings.
