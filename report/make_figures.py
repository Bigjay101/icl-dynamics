"""Report figures for Project 1 (Q1/Q2), built only from the CSV/JSON result files.

    python make_figures.py <results_folder> <out_folder>

fig1_normalised_change: per condition, how far each measure has moved from gen 0 towards
    its final value (0 = gen 0, 1 = final generation). Whichever curve rises first moved first.
fig2_within_training: matching score (best head) and accuracy over training, T=2 generations 0-4.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RES, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({
  "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
  "legend.fontsize": 6, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
  "ytick.color": INK2, "axes.linewidth": 0.6, "font.family": "DejaVu Sans", "pdf.fonttype": 42,
})


def read(path):
  with open(path) as f:
    return list(csv.DictReader(f))


def series(rows, run, key):
  rs = sorted((r for r in rows if r["run"] == run), key=lambda r: int(r["gen"]))
  return np.array([int(r["gen"]) for r in rs]), np.array([float(r[key]) for r in rs])


def style(ax):
  ax.spines[["top", "right"]].set_visible(False)
  ax.grid(axis="y", color=GRID, lw=0.5)
  ax.set_axisbelow(True)


# ---------------------------------------------------------------- figure 1
base = read(os.path.join(RES, "interp/tracking_base/tracking.csv"))
lr3 = read(os.path.join(RES, "interp/tracking_lr3e5/tracking.csv"))
MEASURES = [  # key, label, colour, marker, sign (+1: higher = worse)
  ("loss", "loss", ORANGE, "s", +1),
  ("use_context_prob", "use-context prob.", AQUA, "^", -1),
  ("in_context_acc", "accuracy", YELLOW, "D", -1),
  ("matching_max", "matching score (circuit)", BLUE, "o", -1),
]
PANELS = [(base, "sample_T2", "(a) T = 2, lr 1e-5"), (lr3, "T2_lr3e5", "(b) T = 2, lr 3e-5"),
          (base, "sample_T1", "(c) T = 1, lr 1e-5")]
fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.45), sharey=True)
for ax, (rows, run, title) in zip(axes, PANELS):
  for key, label, c, m, sign in MEASURES:
    g, x = series(rows, run, key)
    frac = (x - x[0]) / (x[-1] - x[0])
    lw = 1.6 if key == "matching_max" else 1.0
    ax.plot(g, frac, color=c, marker=m, ms=3.2, lw=lw, label=label, zorder=3 if key == "matching_max" else 2,
            markeredgecolor="white", markeredgewidth=0.4)
  ax.set_title(title, loc="left", color=INK)
  ax.set_xlabel("generation")
  ax.set_xticks(g)
  ax.set_ylim(-0.08, 1.08)
  style(ax)
axes[0].set_ylabel("fraction of\ntotal change")
h, lab = axes[0].get_legend_handles_labels()
fig.legend(h, lab, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.04))
fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.6)
for ext in ("pdf", "png"):
  fig.savefig(os.path.join(OUT, f"fig1_normalised_change.{ext}"), dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- figure 2
within = read(os.path.join(RES, "interp/tracking_base/within_training.csv"))
gens = [0, 1, 2, 3, 4]
ramp = ["#0b3a75", "#1a5cb0", "#2a78d6", "#6ea4e6", "#a9c9f0"]  # one hue, dark (gen 0) to light
fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.3), sharex=True)
for gi, c in zip(gens, ramp):
  rs = sorted((r for r in within if r["run"] == "sample_T2" and int(r["gen"]) == gi), key=lambda r: int(r["iter"]))
  it = np.array([int(r["iter"]) for r in rs]) / 1e6
  for ax, key in zip(axes, ("matching_max", "in_context_acc")):
    y = np.array([float(r[key]) for r in rs])
    ax.plot(it, y, color=c, lw=1.1, label=f"gen {gi}")
axes[0].set_title("(a) matching score, best head", loc="left")
axes[1].set_title("(b) in-context accuracy", loc="left")
axes[1].axhline(0.5, color=INK2, lw=0.6, ls=":")
axes[1].text(1.0, 0.47, "chance", color=INK2, fontsize=5.5, ha="right", va="top")
for ax in axes:
  ax.set_xlabel("training sequences seen (millions)")
  style(ax)
axes[1].set_ylim(0.44, 1.02)
h, lab = axes[0].get_legend_handles_labels()
fig.legend(h, lab, loc="center left", bbox_to_anchor=(0.99, 0.55), frameon=False, handlelength=1.2,
           title="T = 2", title_fontsize=6)
fig.tight_layout(w_pad=1.2)
for ext in ("pdf", "png"):
  fig.savefig(os.path.join(OUT, f"fig2_within_training_T2.{ext}"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("wrote", sorted(os.listdir(OUT)))

# ---------------------------------------------------------------- figure for the 2-page body: (a-c) + within training
fig, axes = plt.subplots(1, 4, figsize=(5.5, 1.5), gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
for ax, (rows, run, title) in zip(axes[:3], PANELS):
  for key, label, c, m, sign in MEASURES:
    g, x = series(rows, run, key)
    frac = (x - x[0]) / (x[-1] - x[0])
    ax.plot(g, frac, color=c, marker=m, ms=2.6, lw=1.5 if key == "matching_max" else 0.9, label=label,
            zorder=3 if key == "matching_max" else 2, markeredgecolor="white", markeredgewidth=0.3)
  ax.set_title(title, loc="left", fontsize=6.5)
  ax.set_xlabel("generation")
  ax.set_xticks(g[::2])
  ax.set_ylim(-0.08, 1.08)
  style(ax)
for ax in axes[1:3]:
  ax.set_yticklabels([])
axes[0].set_ylabel("fraction of total change")
ax = axes[3]
for gi, c in zip(gens, ramp):
  rs = sorted((r for r in within if r["run"] == "sample_T2" and int(r["gen"]) == gi), key=lambda r: int(r["iter"]))
  it = np.array([int(r["iter"]) for r in rs]) / 1e6
  y = np.array([float(r["matching_max"]) for r in rs])
  ax.plot(it, y, color=c, lw=1.0)
  lab = {0: ("gen 0", 0.05), 2: ("gen 1-2", -0.06), 3: ("gen 3", 0), 4: ("gen 4", 0)}.get(gi)
  if lab:
    ax.text(1.03, y[-1] + lab[1], lab[0], color=INK2, fontsize=5, va="center")
ax.set_title("(d) T = 2, within training", loc="left", fontsize=6.5)
ax.set_xlabel("sequences seen (M)")
ax.set_xlim(0, 1.3)
style(ax)
h, lab = axes[0].get_legend_handles_labels()
fig.legend(h, lab, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.45, 1.06))
fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.4)
for ext in ("pdf", "png"):
  fig.savefig(os.path.join(OUT, f"fig_main.{ext}"), dpi=300, bbox_inches="tight")
plt.close(fig)
