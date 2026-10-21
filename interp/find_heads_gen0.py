"""Q1 step 1: find generation 0's induction head(s) and previous-token head(s).

Three kinds of evidence (attention scores alone are only correlational):
  1. attention scores for all 2 x 8 heads on the real-data probe (fsl_val_rl)
  2. single-head zero-ablation: accuracy / loss change for every head
  3. circuit ablations: the picked IH, the picked PT head, both, and whole layers

Usage (repo root):
  python -m interp.find_heads_gen0 collapse/gen0_pool --out results/interp/gen0_base
  python -m interp.find_heads_gen0 collapse_ext/gen0_p50 --out results/interp/gen0_ext
Optional: --ckpt <path.eqx> (default: the run's last checkpoint).
"""
from __future__ import annotations

import argparse
import json
import os

import h5py
import numpy as np

from interp import jax_backend as jb
from interp.head_scores import all_scores, make_reader, pick_heads


def readers_for(probe):
  c = probe["class_idxs"]
  rs = [make_reader(c, jb.QUERY, "query")]
  if c.shape[1] > jb.NEXT:
    rs.append(make_reader(c, jb.NEXT, "next"))      # extended: the next symbol reads its label
  return rs


def logged_metric(run_folder, it, key="fsl_val_rl/acc"):
  """The value main.py logged at the eval closest to `it` (sanity check for the probe)."""
  path = os.path.join(run_folder, "log.h5")
  if not os.path.exists(path):
    return None
  with h5py.File(path, "r") as f:
    if key not in f:
      return None
    iters = f["eval_iter"][:]
    n = min(len(iters), len(f[key]))
    d = np.abs(iters[:n] - it)
    j = int(np.nonzero(d == d.min())[0][-1])     # last match: a restarted run can log an iteration twice
    return {"iter": int(iters[j]), "value": float(np.mean(f[key][j]))}


def analyse(model, probe, ablate_each=True):
  fwd = jb.forward(model, probe)
  n_items = probe["labels"].shape[1]
  sc = all_scores(fwd["attn"], readers_for(probe), n_items)
  base = jb.metrics(fwd, probe)
  L, H = sc["prev_token"].shape
  drop = {"acc": np.zeros((L, H)), "loss": np.zeros((L, H))}
  if ablate_each:
    for l in range(L):
      for h in range(H):
        m = jb.metrics(jb.forward(model, probe, ablate=[(l, h)]), probe)
        drop["acc"][l, h] = base["acc"] - m["acc"]
        drop["loss"][l, h] = m["loss"] - base["loss"]
  return fwd, sc, base, drop


def heatmaps(mats, picks, path):
  os.environ["MPLBACKEND"] = "Agg"
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig, axes = plt.subplots(len(mats), 1, figsize=(5.2, 1.25 * len(mats) + 0.4), constrained_layout=True)
  for ax, (name, M) in zip(np.atleast_1d(axes), mats.items()):
    lim = max(abs(M).max(), 1e-6)
    im = ax.imshow(M, cmap="RdBu_r" if M.min() < 0 else "viridis",
                   vmin=-lim if M.min() < 0 else 0, vmax=lim, aspect="auto")
    ax.set_title(name, fontsize=8, loc="left")
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels([f"L{l}" for l in range(M.shape[0])], fontsize=7)
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels([f"H{h}" for h in range(M.shape[1])], fontsize=7)
    for r in range(M.shape[0]):
      for c in range(M.shape[1]):
        ax.text(c, r, f"{M[r, c]:.2f}", ha="center", va="center", fontsize=6,
                color="w" if abs(M[r, c]) > 0.5 * lim else "k")
    for key, mk in (("induction", "s"), ("prev_token", "o")):
      l, h = picks[key]
      ax.scatter([h], [l], s=170, facecolors="none", edgecolors="red", marker=mk, linewidths=1.4)
    fig.colorbar(im, ax=ax, shrink=0.9)
  fig.savefig(path, dpi=200)
  plt.close(fig)


def patterns(attn, picks, path, b=0):
  os.environ["MPLBACKEND"] = "Agg"
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  T = attn.shape[-1]
  names = ["x0", "y0", "x1", "y1", "xq", "yq", "xn"][:T]
  fig, axes = plt.subplots(1, 2, figsize=(6.4, 3), constrained_layout=True)
  for ax, key in zip(axes, ("prev_token", "induction")):
    l, h = picks[key]
    ax.imshow(attn[l, :, h].mean(0), cmap="Blues", vmin=0, vmax=1)   # mean over probe sequences
    ax.set_title(f"{key} head L{l}H{h} (mean pattern)", fontsize=8)
    ax.set_xticks(range(T)); ax.set_xticklabels(names, fontsize=7)
    ax.set_yticks(range(T)); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("attends to (source)"); ax.set_ylabel("from (destination)")
  fig.savefig(path, dpi=200)
  plt.close(fig)


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument("run_folder")
  ap.add_argument("--ckpt", default=None)
  ap.add_argument("--evaluator", default="fsl_val_rl")
  ap.add_argument("--out", default="results/interp/gen0")
  ap.add_argument("--ind-min-layer", type=int, default=1)
  args = ap.parse_args(argv)
  os.makedirs(args.out, exist_ok=True)

  model, opts, it = jb.load_run(args.run_folder, args.ckpt)
  probe = jb.build_probe(opts, args.evaluator)
  fwd, sc, base, drop = analyse(model, probe)
  picks = pick_heads(sc, "matching", args.ind_min_layer)
  ind, pt = tuple(picks["induction"]), tuple(picks["prev_token"])
  L, H = sc["prev_token"].shape

  circuit = {}
  pt_set = [tuple(x) for x in picks["prev_token_set"]]
  sets = {"none": [], "IH": [ind], "PT": [pt], "IH+PT": [ind, pt], "all PT-set heads": pt_set,
          f"all L{ind[0]} except IH": [(ind[0], h) for h in range(H) if h != ind[1]],
          f"all L{ind[0]}": [(ind[0], h) for h in range(H)],
          f"all L{pt[0]}": [(pt[0], h) for h in range(H)]}
  for name, heads in sets.items():
    circuit[name] = jb.metrics(jb.forward(model, probe, ablate=heads), probe)

  argmax_abl = [int(v) for v in np.unravel_index(drop["acc"].argmax(), drop["acc"].shape)]
  summary = {
    "run_folder": args.run_folder, "iteration": it, "evaluator": args.evaluator,
    "n_probe": int(len(probe["labels"])), "extended": bool(probe["labels"].shape[1] > 3),
    "metrics": base, "logged_in_log_h5": logged_metric(args.run_folder, it),
    "picks": picks,
    "scores_of_picks": {"matching": float(sc["matching"][ind]), "induction": float(sc["induction"][ind]),
                        "prev_token": float(sc["prev_token"][pt]), "chance_induction": sc["chance"],
                        "prev_token_set": {f"{a}:{b}": float(sc["prev_token"][a, b]) for a, b in pt_set}},
    "layer_numbering": "0-indexed (L0 = the team guide's 'layer 1', L1 = 'layer 2')",
    "single_ablation_acc_drop_of_picks": {"IH": float(drop["acc"][ind]), "PT": float(drop["acc"][pt])},
    "most_important_head_by_ablation": argmax_abl,
    "ablation_agrees_with_attention_pick": argmax_abl == list(ind),
    "circuit_ablation": circuit,
    "paper_reference": "Singh et al. used IH 1:3 and PT 0:2 for init seed 5 (ih_paper_runs.sh)",
  }
  if "next_matching" in sc:
    summary["next_symbol_reader"] = {"best_head": pick_heads(sc, "next_matching", args.ind_min_layer)["induction"],
                                     "matching_of_query_IH": float(sc["next_matching"][ind])}

  np.savez(os.path.join(args.out, "gen0_scores.npz"),
           **{k: v for k, v in sc.items() if isinstance(v, np.ndarray)},
           ablation_acc_drop=drop["acc"], ablation_loss_increase=drop["loss"],
           mean_attn=fwd["attn"].mean(1))
  with open(os.path.join(args.out, "gen0_heads.json"), "w") as f:
    json.dump(summary, f, indent=2)
  mats = {"previous-token score": sc["prev_token"], "induction score (query -> correct label)": sc["induction"],
          "matching score (correct label - distractor label)": sc["matching"],
          "exemplar attention (query -> matching exemplar)": sc["exemplar_attn"],
          "accuracy drop when ablated": drop["acc"]}
  if "next_matching" in sc:
    mats["matching score, next symbol (extended)"] = sc["next_matching"]
  heatmaps(mats, picks, os.path.join(args.out, "gen0_head_heatmaps.png"))
  patterns(fwd["attn"], picks, os.path.join(args.out, "gen0_patterns.png"))
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()
