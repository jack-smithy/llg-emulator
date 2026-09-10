"""Paper figures for main.tex. CPU only (SP4 is 101x26 nodal). Run: uv run python figures/make_figures.py"""
import json, numpy as np, jax.random as jr, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from llg_emulator.config import sp4_path
from llg_emulator.data import open_trajectory
from llg_emulator.io import load_model
from llg_emulator.metrics import rollout_metrics, bulk_rmse

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "lines.linewidth": 1.1,
                     "savefig.bbox": "tight", "pdf.fonttype": 42})
W = 3.4  # \columnwidth in inches
OUT = Path(__file__).parent
ANALYSIS = Path(__file__).resolve().parent.parent / "analysis"
ERR = r"$|\mathbf{m}_\mathrm{learned}-\mathbf{m}_\mathrm{LLG}|$"

trj = open_trajectory(sp4_path()); dt_ps = trj.dt * 1e12

# Fig 2: SP4 <m>(t), reference vs learned (one-step model)
model = load_model(Path("weights/run1"), key=jr.PRNGKey(0), tag="best")
ref, pred, corr, _ = rollout_metrics(model, sp4_path(), stride=1)
t = np.arange(len(ref)) * dt_ps
fig, ax = plt.subplots(3, 1, figsize=(W, 3.6), sharex=True, constrained_layout=True)
for i, lab in enumerate((r"$\langle m_x\rangle$", r"$\langle m_y\rangle$", r"$\langle m_z\rangle$")):
    ax[i].plot(t, ref[:, i], "k-", label="LLG reference")
    ax[i].plot(t, pred[:, i], "C3--", label="learned integrator")
    ax[i].set_ylabel(lab); ax[i].set_ylim(-1.05, 1.05); ax[i].grid(alpha=.25)
ax[0].legend(loc="upper right", frameon=False); ax[-1].set_xlabel("t (ps)")
fig.savefig(OUT / "sp4_bulk.pdf"); plt.close(fig)
print(f"fig2: sp4 bulk rmse {bulk_rmse(ref, pred):.4f}")

# Fig 3: spatial error at mid-reversal, true 4:1 aspect, from the ParaView export
from vtk import vtkXMLRectilinearGridReader
from vtk.util.numpy_support import vtk_to_numpy
def read(fn):
    r = vtkXMLRectilinearGridReader(); r.SetFileName(str(fn)); r.Update()
    return vtk_to_numpy(r.GetOutput().GetCellData().GetArray("f")).reshape(26, 101, 3)
fr = 50; ext = [0, 500, 0, 125]
fig, ax = plt.subplots(3, 1, figsize=(W, 3.0), constrained_layout=True)
im0 = ax[0].imshow(read(f"weights/run1/vtr/sp4_reference_{fr:04d}.vtr")[..., 0],
                   cmap="RdBu_r", vmin=-1, vmax=1, origin="lower", extent=ext)
ax[0].set_ylabel("reference", fontsize=7); fig.colorbar(im0, ax=ax[0], label="$m_x$", pad=0.02)
for k, (run, lab) in enumerate((("run1", "one-step loss"), ("run-pf", "pushforward"))):
    e = np.linalg.norm(read(f"weights/{run}/vtr/sp4_error_{fr:04d}.vtr"), axis=-1)
    im = ax[k + 1].imshow(e, cmap="magma", vmin=0, vmax=1, origin="lower", extent=ext)
    ax[k + 1].set_ylabel(lab, fontsize=7)
    ax[k + 1].text(10, 95, f"mean {e.mean():.3f}", color="w", fontsize=6)
fig.colorbar(im, ax=ax[1:], label=ERR, pad=0.02)
for a in ax: a.set_yticks([0, 125]); a.set_xticks([])
ax[-1].set_xticks([0, 250, 500]); ax[-1].set_xlabel("x (nm)")
fig.savefig(OUT / "sp4_error_map.pdf"); plt.close(fig); print("fig3 ok")

# Fig 4: error growth, teacher-forced vs autoregressive
d1 = json.load(open(ANALYSIS / "diagnose_run1.json")); dp = json.load(open(ANALYSIS / "diagnose_run-pf.json"))
fig, ax = plt.subplots(figsize=(W, 2.4), constrained_layout=True)
tt = np.arange(len(d1["fr_mean"])) * dt_ps
ax.semilogy(tt[1:], d1["tf_mean"], "C0:", label="one-step loss, single step from LLG state")
ax.semilogy(tt, d1["fr_mean"], "C0-", label="one-step loss, autoregressive")
ax.semilogy(tt[1:], dp["tf_mean"], "C3:", label="pushforward, single step from LLG state")
ax.semilogy(tt, dp["fr_mean"], "C3-", label="pushforward, autoregressive")
ax.set_xlabel("t (ps)"); ax.set_ylabel("mean " + ERR); ax.grid(alpha=.25, which="both")
ax.legend(frameon=False, loc="upper left"); ax.set_ylim(top=1.0)
fig.savefig(OUT / "error_growth.pdf"); plt.close(fig); print("fig4 ok")

# Fig 5: transfer across mesh size (numbers from BENCHMARKS.md), stacked panels
cells = np.array([100 * 25, 2000**2, 4000**2])
bulk = {"one-step loss": [0.0179, 0.0076, 0.0353], "pushforward": [0.0236, 0.0027, 0.0495]}
cos = {"one-step loss": [0.9977, 0.9973, 0.9908], "pushforward": [0.9839, 1.0000, 0.9945]}
fig, ax = plt.subplots(2, 1, figsize=(W, 3.0), sharex=True, constrained_layout=True)
for (lab, v), c in zip(bulk.items(), ("C0", "C3")):
    ax[0].semilogx(cells, v, c + "o-", label=lab); ax[1].semilogx(cells, cos[lab], c + "o-")
for a in ax: a.axvline(256**2, color="k", ls=":", lw=.8); a.grid(alpha=.25)
ax[0].text(256**2 * 1.2, 0.049, "training mesh", fontsize=6, va="top")
ax[0].set_ylabel(r"RMSE of $\langle\mathbf{m}\rangle(t)$"); ax[1].set_ylabel("final-frame cosine")
ax[1].set_xlabel("cells"); ax[1].set_xticks(cells); ax[1].minorticks_off()
ax[1].set_xticklabels([r"SP4 $100\times25$", r"$2000^2$", r"$4000^2$"], fontsize=6)
ax[0].legend(frameon=False, loc="upper left")
fig.savefig(OUT / "mesh_transfer.pdf"); plt.close(fig); print("fig5 ok")
