"""Scaling: seconds per 10 ps step vs number of nodes, learned integrator vs dopri5.
Reads analysis/scaling.json (written by analysis/scaling.py); writes figures/scaling.{pdf,png}."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "lines.linewidth": 1.1,
                     "savefig.bbox": "tight", "pdf.fonttype": 42})
d = json.load(open(Path(__file__).resolve().parent.parent / "analysis" / "scaling.json"))
N = np.array([r["nodes"] for r in d]); ts = np.array([r["solver"] for r in d]); tm = np.array([r["model"] for r in d])
ok = np.isfinite(tm)

fig, ax = plt.subplots(figsize=(3.4, 2.9))
ax.loglog(N, ts, "ks-", label="reference solver (dopri5)")
ax.loglog(N[ok], tm[ok], "C3o-", label="learned integrator")
# linear-in-N guide through the largest common point
i = np.where(ok)[0][-1]
g = np.array([N[ok][0], N[i]]); ax.loglog(g, tm[i] * g / N[i], "C3:", lw=0.9, alpha=0.7)
ax.loglog(g, ts[i] * g / N[i], "k:", lw=0.9, alpha=0.7)
ax.text(2e5, tm[i] * 2e5 / N[i] * 0.35, r"$\propto N$", fontsize=7, color="0.3", ha="center")
for n, a, b in zip(N[ok], ts[ok], tm[ok]):
    if n >= 6e4:
        ax.annotate(f"{a/b:.1f}×", (n, b), textcoords="offset points", xytext=(0, -11), ha="center", fontsize=7, color="C3")
if (~ok).any():
    ax.plot(N[~ok], ts[~ok] * 0 + tm[ok][-1] * N[~ok] / N[ok][-1], "C3x", ms=6, label="learned: out of memory")
ax.set_xlabel("nodes $N$"); ax.set_ylabel("wall time per 10 ps step (s)")
ax.set_xticks([1e3, 1e4, 1e5, 1e6, 1e7]); ax.grid(alpha=.3, which="both", lw=0.5)
ax.legend(frameon=False, loc="upper left")
fig.savefig(Path(__file__).parent / "scaling.pdf"); fig.savefig(Path(__file__).parent / "scaling.png", dpi=130)
print("nodes / solver / model / ratio:")
for n, a, b in zip(N, ts, tm): print(f"  {n:9d}  {a:8.4f}  {b:8.4f}  {a/b:6.1f}")
