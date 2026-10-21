"""Q1 step 2: measure the generation-0 circuit at every generation (and, with
--within-training, at every checkpoint of every generation) and test whether the circuit
measures move BEFORE the behavioural measures do (team guide, section 3.1).

Folder layout written by run_generations.sh:
  <gen0_run>/                            generation 0 (e.g. collapse/gen0_pool)
  <condition>/gen{n}/                    generation n (trained from scratch, init seed 5)
  <condition>/pool_gen{n+1}_summary.json what model n wrote into the next pool

Every model is probed on the SAME real-data sequences: generation 0's fsl_val_rl evaluator
(1,000 sequences, held-out label pairs, real labels), rebuilt exactly as main.py builds it.

Across generations (each generation's last checkpoint) -> tracking.csv
  behaviour : acc, in_context_acc, loss, use_context_prob (+ next_label_*, symbol_* if extended)
  circuit   : matching_g0IH, induction_g0IH, exemplar_attn_g0IH   (the gen-0 induction head)
              prev_token_g0PT (gen-0 PT head), prev_token_g0PTset (mean over all gen-0 PT heads)
              matching_max / ind_head_best, prev_token_max / pt_head_best (strongest head now)
  causal    : abl_drop_g0IH, abl_drop_best, abl_drop_IH+PT (accuracy lost when ablated)
  pool      : pool_error_rate (+ pool_next_* if extended) of the pool this model wrote
-> early_warning.csv : for each circuit measure and each behavioural measure, the FIRST
  generation at which it is worse than generation 0 by more than --z paired standard errors
  (paired over the same 1,000 sequences), and lead = behaviour_gen - circuit_gen
  (positive = the circuit moved first = an early warning). Also the first generation with a
  relative drop of more than 10% (a cruder, scale-dependent check).

Within training (--within-training K: every K-th checkpoint, K=1 = all) -> within_training.csv
  and formation.csv: per generation, the iterations at which the matching score (best head)
  and in_context_acc complete 10/50/90% of their rise (same definition as find_phases.py),
  and lead_50 = acc_50 - circuit_50 (positive = circuit forms first).

IMPORTANT: every generation is trained from scratch with the same init seed, so the gen-0 head
index is a reasonable but unproven identity across generations; report ind_head_best too.

Usage (repo root):
  python -m interp.track_generations --run sample_T1=collapse/gen0_pool:collapse/sample_T1 \
      --gen0-json results/interp/gen0_base/gen0_heads.json --out results/interp/tracking_base
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re

import numpy as np

from interp import jax_backend as jb
from interp.find_heads_gen0 import readers_for
from interp.head_scores import all_scores, first_move, per_sequence, pick_heads, rise_points

CIRCUIT = ["matching_g0IH", "induction_g0IH", "prev_token_g0PT", "prev_token_g0PTset"]
BEHAVIOUR = {"in_context_acc": "down", "acc": "down", "loss": "up", "use_context_prob": "down"}
CIRCUIT_DIR = "down"


# ------------------------------------------------------------------ small helpers
def write_csv(rows, path):
  cols = []
  for r in rows:
    cols += [k for k in r if k not in cols]
  with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
      w.writerow({k: r.get(k, "") for k in cols})


def group(rows, *keys):
  out = {}
  for r in rows:
    out.setdefault(tuple(r[k] for k in keys), []).append(r)
  return {k: sorted(v, key=lambda r: (r.get("gen", 0), r.get("iter", 0))) for k, v in sorted(out.items())}


def col(rows, k):
  return np.array([np.nan if r.get(k) in (None, "") else float(r[k]) for r in rows])


def generations(gen0_run, condition):
  gens = [(0, gen0_run)]
  if os.path.isdir(condition):
    for d in os.listdir(condition):
      m = re.fullmatch(r"gen(\d+)", d)
      if m and os.path.isdir(os.path.join(condition, d, "checkpoints")):
        gens.append((int(m.group(1)), os.path.join(condition, d)))
  return sorted(gens)


def pool_outcomes(condition, gen):
  p = os.path.join(condition, f"pool_gen{gen + 1}_summary.json")
  if not os.path.exists(p):
    return {}
  with open(p) as f:
    s = json.load(f)
  out = {"pool_error_rate": s.get("error_rate"), "pool_error_distractor": s.get("error_distractor_label"),
         "pool_error_out_of_context": s.get("error_out_of_context")}
  for k in ("next_repeat_share_in_context", "next_out_of_context", "next_label_matches_symbol"):
    if k in s:
      out["pool_" + k] = s[k]
  return out


# ------------------------------------------------------------------ scoring one checkpoint
def score_checkpoint(model, probe, g0, with_ablation):
  """Returns (row of means, per-sequence dict, score matrices)."""
  fwd = jb.forward(model, probe)
  readers = readers_for(probe)
  n_items = probe["labels"].shape[1]
  sc = all_scores(fwd["attn"], readers, n_items)
  best = pick_heads(sc, "matching")
  ih0, pt0 = tuple(g0["induction"]), tuple(g0["prev_token"])
  pt_set = [tuple(x) for x in g0.get("prev_token_set", [g0["prev_token"]])]
  ihb, ptb = tuple(best["induction"]), tuple(best["prev_token"])
  seq = per_sequence(fwd["attn"], readers[0], n_items, ih0, [pt0])
  seq_set = per_sequence(fwd["attn"], readers[0], n_items, ih0, pt_set)
  beh = jb.per_sequence_behaviour(fwd, probe)
  per_seq = {"matching_g0IH": seq["matching"], "induction_g0IH": seq["induction"],
             "prev_token_g0PT": seq["prev_token"], "prev_token_g0PTset": seq_set["prev_token"], **beh}
  row = {**jb.metrics(fwd, probe),
         "matching_g0IH": float(sc["matching"][ih0]), "induction_g0IH": float(sc["induction"][ih0]),
         "exemplar_attn_g0IH": float(sc["exemplar_attn"][ih0]),
         "prev_token_g0PT": float(sc["prev_token"][pt0]),
         "prev_token_g0PTset": float(np.mean([sc["prev_token"][h] for h in pt_set])),
         "matching_max": float(sc["matching"][ihb]), "ind_head_best": "%d:%d" % ihb,
         "prev_token_max": float(sc["prev_token"][ptb]), "pt_head_best": "%d:%d" % ptb,
         "n_strong_IH": int((sc["matching"][1:] > 0.5).sum())}
  if "next_matching" in sc:
    row["next_matching_g0IH"] = float(sc["next_matching"][ih0])
    row["next_matching_max"] = float(sc["next_matching"][1:].max())
  if with_ablation:
    for tag, heads in {"g0IH": [ih0], "best": [ihb], "IH+PT": [ih0, pt0]}.items():
      row[f"abl_drop_{tag}"] = row["acc"] - jb.metrics(jb.forward(model, probe, ablate=heads), probe)["acc"]
  return row, per_seq, sc


# ------------------------------------------------------------------ early warning
def first_rel_drop(x, worse, frac=0.10):
  x = np.asarray(x, float)
  for g in range(1, len(x)):
    rel = (x[g] - x[0]) / abs(x[0]) if x[0] != 0 else np.inf * np.sign(x[g] - x[0])
    if (worse == "down" and rel < -frac) or (worse == "up" and rel > frac):
      return g
  return None


def early_warning(rows, per_seq, z):
  """per_seq[run][gen] -> dict of per-sequence arrays."""
  out = []
  for (run,), d in group(rows, "run").items():
    gens = [r["gen"] for r in d]
    seqs = [per_seq[run][g] for g in gens]
    as_gen = lambda i: None if i is None else gens[i]  # noqa: E731  (list index -> generation number)
    beh_first = {b: as_gen(first_move(seqs[0][b], [s[b] for s in seqs], w, z)[0]) for b, w in BEHAVIOUR.items()}
    beh_rel = {b: as_gen(first_rel_drop(col(d, b), w)) for b, w in BEHAVIOUR.items()}
    for c in CIRCUIT:
      ci, zs = first_move(seqs[0][c], [s[c] for s in seqs], CIRCUIT_DIR, z)
      cg = as_gen(ci)
      crel = as_gen(first_rel_drop(col(d, c), CIRCUIT_DIR))
      for b in BEHAVIOUR:
        bg = beh_first[b]
        out.append({"run": run, "circuit_measure": c, "behaviour": b,
                    "circuit_first_gen": cg, "behaviour_first_gen": bg,
                    "lead_generations": (None if cg is None or bg is None else bg - cg),
                    "verdict": verdict(cg, bg),
                    "circuit_first_gen_rel10": crel, "behaviour_first_gen_rel10": beh_rel[b],
                    "z_threshold": z,
                    "circuit_z_by_gen": " ".join(f"{gens[i]}:{v:.1f}" for i, v in enumerate(zs))})
  return out


def verdict(cg, bg):
  if cg is None and bg is None:
    return "neither moved"
  if cg is None:
    return "behaviour moved, circuit did not"
  if bg is None:
    return "circuit moved, behaviour did not (within these generations)"
  if cg < bg:
    return "circuit first (early warning)"
  if cg == bg:
    return "same generation"
  return "behaviour first"


# ------------------------------------------------------------------ plots
def plot_generations(rows, path):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  panels = [("in_context_acc", "in-context accuracy"), ("loss", "loss (query)"),
            ("use_context_prob", "use_context_prob"), ("matching_g0IH", "matching score, gen-0 IH"),
            ("prev_token_g0PTset", "previous-token score, gen-0 PT heads")]
  if any(r.get("pool_error_rate") is not None for r in rows):
    panels.append(("pool_error_rate", "error rate in the pool it wrote"))
  fig, axes = plt.subplots(1, len(panels), figsize=(2.9 * len(panels), 2.7), constrained_layout=True)
  for (run,), d in group(rows, "run").items():
    for ax, (k, t) in zip(axes, panels):
      ax.plot(col(d, "gen"), col(d, k), marker="o", ms=3, label=run)
  for ax, (k, t) in zip(axes, panels):
    ax.set_title(t, fontsize=8)
    ax.set_xlabel("generation")
    ax.grid(alpha=0.3)
  axes[0].legend(fontsize=6)
  fig.savefig(path, dpi=200)
  plt.close(fig)


def plot_within(within, path):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  runs = sorted({r["run"] for r in within})
  fig, axes = plt.subplots(2, len(runs), figsize=(4 * len(runs), 5), squeeze=False, constrained_layout=True)
  cmap = matplotlib.colormaps["viridis"]
  groups = group(within, "run", "gen")
  for j, run in enumerate(runs):
    gens = sorted(g for (r, g) in groups if r == run)
    for g in gens:
      d = groups[(run, g)]
      c = cmap(g / max(max(gens), 1))
      axes[0, j].plot(col(d, "iter"), col(d, "matching_max"), color=c, lw=1, label=f"gen {g}")
      axes[1, j].plot(col(d, "iter"), col(d, "in_context_acc"), color=c, lw=1)
    axes[0, j].set_title(f"{run}: matching score (best head)", fontsize=8)
    axes[1, j].set_title(f"{run}: in-context accuracy", fontsize=8)
    for ax in axes[:, j]:
      ax.set_xlabel("training sequences seen")
      ax.grid(alpha=0.3)
    axes[0, j].legend(fontsize=5, ncol=2)
  fig.savefig(path, dpi=200)
  plt.close(fig)


# ------------------------------------------------------------------ main
def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument("--run", action="append", required=True, help="NAME=GEN0_RUN:CONDITION_FOLDER (repeatable)")
  ap.add_argument("--gen0-json", default=None, help="gen0_heads.json from find_heads_gen0 (fixes the picks)")
  ap.add_argument("--evaluator", default="fsl_val_rl")
  ap.add_argument("--out", default="results/interp/tracking")
  ap.add_argument("--within-training", type=int, default=0, metavar="K",
                  help="also score every K-th checkpoint of every generation (1 = every checkpoint; 0 = off)")
  ap.add_argument("--z", type=float, default=3.0, help="paired z threshold for 'moved' (default 3)")
  ap.add_argument("--no-cache", action="store_true",
                  help="recompute everything (default: finished generations are cached in OUT/cache, so a "
                       "rerun after a disconnect continues where it stopped)")
  args = ap.parse_args(argv)
  os.makedirs(args.out, exist_ok=True)

  rows, within, mats, per_seq = [], [], {}, {}
  for spec in args.run:
    name, paths = spec.split("=", 1)
    gen0_run, condition = paths.split(":", 1)
    gens = generations(gen0_run, condition)
    model0, opts0, _ = jb.load_run(gen0_run)
    probe = jb.build_probe(opts0, args.evaluator)          # fixed real-data probe for this run
    if args.gen0_json:
      with open(args.gen0_json) as f:
        g0 = json.load(f)["picks"]
    else:
      fwd0 = jb.forward(model0, probe)
      g0 = pick_heads(all_scores(fwd0["attn"], readers_for(probe), probe["labels"].shape[1]), "matching")
    print(f"[{name}] {len(gens)} generations; gen-0 IH {g0['induction']}, PT {g0['prev_token']}, "
          f"PT set {g0.get('prev_token_set')}", flush=True)
    per_seq[name] = {}
    for gen, folder in gens:
      # keyed by the run folder, so generation 0 (shared by every condition) is scored once
      key = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.normpath(folder))
      cache = os.path.join(args.out, "cache", f"{key}__w{args.within_training}.pkl")
      if os.path.exists(cache) and not args.no_cache:
        with open(cache, "rb") as f:
          c = pickle.load(f)
        c["row"] = {**c["row"], "run": name, "gen": gen, **pool_outcomes(condition, gen)}
        c["within"] = [{**w, "run": name, "gen": gen} for w in c["within"]]
        print(f"  gen {gen:2d}: cached", flush=True)
      else:
        model, opts, it = jb.load_run(folder)
        if bool(getattr(opts, "extended", False)) != bool(getattr(opts0, "extended", False)):
          raise SystemExit(f"{folder} and {gen0_run} mix base and extended runs")
        row, ps, sc = score_checkpoint(model, probe, g0, with_ablation=True)
        row.update(run=name, gen=gen, iter=it, folder=folder, **pool_outcomes(condition, gen))
        print(f"  gen {gen:2d}: in-ctx acc {row['in_context_acc']:.3f}  loss {row['loss']:.3f}  "
              f"use_ctx {row['use_context_prob']:.3f}  matching(g0 IH) {row['matching_g0IH']:.3f}  "
              f"best {row['ind_head_best']}  PT(g0) {row['prev_token_g0PT']:.3f}  pool err {row.get('pool_error_rate')}",
              flush=True)
        wrows = []
        if args.within_training:
          ckpts = jb.list_checkpoints(folder)
          for k, (ci, cpath) in enumerate(ckpts):
            if k % args.within_training and k != len(ckpts) - 1:
              continue
            m, _, _ = jb.load_run(folder, cpath)
            w, _, _ = score_checkpoint(m, probe, g0, with_ablation=False)
            w.update(run=name, gen=gen, iter=ci)
            wrows.append(w)
          print(f"         within training: {len(wrows)} checkpoints scored", flush=True)
        c = {"row": row, "per_seq": ps, "matching": sc["matching"], "prev_token": sc["prev_token"], "within": wrows}
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache + ".tmp", "wb") as f:
          pickle.dump(c, f)
        os.replace(cache + ".tmp", cache)
      rows.append(c["row"])
      per_seq[name][gen] = c["per_seq"]
      mats[f"{name}_gen{gen}_matching"] = c["matching"]
      mats[f"{name}_gen{gen}_prev_token"] = c["prev_token"]
      within.extend(c["within"])

  write_csv(rows, os.path.join(args.out, "tracking.csv"))
  np.savez(os.path.join(args.out, "score_matrices.npz"), **mats)
  ew = early_warning(rows, per_seq, args.z)
  write_csv(ew, os.path.join(args.out, "early_warning.csv"))
  plot_generations(rows, os.path.join(args.out, "tracking.png"))
  if within:
    write_csv(within, os.path.join(args.out, "within_training.csv"))
    form = []
    for (run, gen), d in group(within, "run", "gen").items():
      it = col(d, "iter")
      c = rise_points(it, col(d, "matching_max"))
      a = rise_points(it, col(d, "in_context_acc"))
      lead = None if c["iter_50pct"] is None or a["iter_50pct"] is None else a["iter_50pct"] - c["iter_50pct"]
      form.append({"run": run, "gen": gen,
                   "circuit_10": c["iter_10pct"], "circuit_50": c["iter_50pct"], "circuit_90": c["iter_90pct"],
                   "acc_10": a["iter_10pct"], "acc_50": a["iter_50pct"], "acc_90": a["iter_90pct"],
                   "lead_50 (acc_50 - circuit_50)": lead,
                   "final_matching": float(col(d, "matching_max")[-1]),
                   "final_in_context_acc": float(col(d, "in_context_acc")[-1])})
    write_csv(form, os.path.join(args.out, "formation.csv"))
    plot_within(within, os.path.join(args.out, "within_training.png"))
  print(f"\n{'run':<16}{'circuit':<20}{'behaviour':<18}{'circ@':>6}{'beh@':>6}{'lead':>6}  verdict")
  for r in ew:
    print(f"{r['run']:<16}{r['circuit_measure']:<20}{r['behaviour']:<18}{str(r['circuit_first_gen']):>6}"
          f"{str(r['behaviour_first_gen']):>6}{str(r['lead_generations']):>6}  {r['verdict']}")


if __name__ == "__main__":
  main()
