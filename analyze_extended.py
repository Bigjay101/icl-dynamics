"""Analysis of an extended-version collapse condition, generation by generation.

    python analyze_extended.py <condition_folder> <gen0_run_folder> [--out <folder>] [--sims 200]

e.g. python analyze_extended.py collapse_ext/T1 collapse_ext/gen0_p50

Reads pool_gen{n}.h5 (written by model n-1) and each model's log.h5, and writes
<out>/summary.csv, <out>/summary.json and <out>/drift.pdf (+ .png).

The drift test. Each distinct training sequence (context + query) appears about
m = 13 times in the pool, every copy with its own next symbol. Let s be the
share of copies of one sequence whose next symbol repeats the query (among
copies whose next symbol is in context). Two hypotheses:

  generalise: the model learns one shared repeat probability p for all
      sequences. Every pool then resamples each copy from p, so across
      sequences s has the binomial spread Var = p(1-p)/m in every generation.
  memorise: the model learns each sequence's own share s. Every pool then
      resamples each copy from that sequence's s: a Wright-Fisher process
      (the discrete case of Shumailov et al.'s Markov chain), in which the
      spread grows, Var_t = p(1-p)[1 - (1-1/m)^(t+1)], and sequences fix at
      s = 0 or 1 (always repeat or never).

The script computes both predictions for the actual copy counts and compares
them with the observed spread and fixation.

A third source of spread is the model itself: even without memorising, a model
can prefer repeating more for some sequences than others (approximation error,
Shumailov et al.'s other error source). Pool n is sampled from model n-1's
per-sequence probabilities q, so its spread is E[q(1-q)/m] + Var(q). The
script therefore also reports, for each model:
  model_q_var        Var(q) across sequences: how unevenly the model prefers
                     repeating (0 for a model that learned one shared p)
  memorisation_corr  corr(q, share in its own training pool): does the model
                     copy each sequence's share (sampling-error route)?
  persistence_corr   corr(q of model n-1, q of model n-2): do the same
                     sequences keep the same bias across generations
                     (approximation errors compounding)?
"""
import argparse
import csv
import json
import os

import h5py
import numpy as np


def context_ids(pool):
  """Integer id per row, shared by rows with the same context and query."""
  key = np.concatenate([pool['class_idxs'][:, :3], pool['exemplar_inds'][:, :3],
                        pool['labels'][:, :2], pool['true_query_label'][:, None]], axis=1)
  _, ids = np.unique(key, axis=0, return_inverse=True)
  return ids.ravel()


def load_pool(path):
  with h5py.File(path, 'r') as f:
    keys = ['class_idxs', 'exemplar_inds', 'labels', 'true_query_label', 'true_next_class',
            'symbol_repeat_prob', 'symbol_other_prob', 'kept_real']
    pool = {k: f[k][()] for k in keys if k in f}
    pool['attrs'] = dict(f.attrs)
  return pool


def pool_stats(pool, ids, n_ctx):
  c, l = pool['class_idxs'], pool['labels']
  repeat = c[:, 3] == c[:, 2]
  other_class = np.where(c[:, 0] == c[:, 2], c[:, 1], c[:, 0])
  other = c[:, 3] == other_class
  in_ctx = repeat | other
  # Per-sequence share of repeats among in-context copies
  n_in = np.bincount(ids, weights=in_ctx, minlength=n_ctx)
  n_rep = np.bincount(ids, weights=repeat, minlength=n_ctx)
  has = n_in > 0
  share = np.where(has, n_rep / np.maximum(n_in, 1), np.nan)
  return {
    'repeat_share': float(repeat.mean()),
    'other_share': float(other.mean()),
    'out_of_context': float((~in_ctx).mean()),
    'repeat_share_in_context': float(repeat.sum() / max(in_ctx.sum(), 1)),
    'query_error': float((l[:, 2] != pool['true_query_label']).mean()),
    'share_var': float(np.nanvar(share)),
    'fixed_frac': float(np.mean((share[has] == 0) | (share[has] == 1))),
    'share': share, 'n_in': n_in,
  }


def model_metrics(run):
  path = os.path.join(run, 'log.h5')
  if not os.path.exists(path):
    return {}
  out = {}
  with h5py.File(path, 'r') as f:
    for m in ['in_context_acc', 'loss', 'next_label_in_context_acc', 'symbol_loss',
              'symbol_use_context_prob', 'symbol_repeat_prob', 'symbol_other_prob', 'total_loss']:
      k = 'fsl_val_rl/' + m
      if k in f:
        out['val_' + m] = float(np.nanmean(f[k][-1]))
  if 'val_symbol_repeat_prob' in out:
    out['val_symbol_repeat_share'] = out['val_symbol_repeat_prob'] / (
      out['val_symbol_repeat_prob'] + out['val_symbol_other_prob'])
  return out


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('condition')
  ap.add_argument('gen0_run')
  ap.add_argument('--out', default=None, help='Output folder (default <condition>/analysis)')
  ap.add_argument('--sims', type=int, default=200, help='Simulated Wright-Fisher runs for the fixation prediction')
  ap.add_argument('--seed', type=int, default=0)
  args = ap.parse_args()
  out_dir = args.out or os.path.join(args.condition, 'analysis')
  os.makedirs(out_dir, exist_ok=True)

  pools = [os.path.join(args.gen0_run, 'pool_gen0.h5')]
  n = 1
  while os.path.exists(os.path.join(args.condition, f'pool_gen{n}.h5')):
    pools.append(os.path.join(args.condition, f'pool_gen{n}.h5'))
    n += 1
  runs = [args.gen0_run] + [os.path.join(args.condition, f'gen{g}') for g in range(1, len(pools))]
  print(f'{len(pools)} pools (generation 0 to {len(pools) - 1})')

  p0 = load_pool(pools[0])
  ids = context_ids(p0)
  n_ctx = int(ids.max()) + 1
  copies = np.bincount(ids, minlength=n_ctx)
  print(f'{n_ctx} distinct sequences, {copies.mean():.2f} copies each on average')

  rows, prev_share, prev_q = [], None, None
  for g, (path, run) in enumerate(zip(pools, runs)):
    pool = load_pool(path)
    # Same rows in the same order in every generation; check the contexts match
    if g > 0 and not np.array_equal(pool['class_idxs'][:, :3], p0['class_idxs'][:, :3]):
      raise ValueError(f'{path} does not have the same sequences as generation 0')
    st = pool_stats(pool, ids, n_ctx)
    row = {'generation': g, **{k: v for k, v in st.items() if k not in ('share', 'n_in')}}
    # Generalise: every in-context copy drawn from one shared p (this generation's)
    has, nin, pt = st['n_in'] > 0, st['n_in'][st['n_in'] > 0], row['repeat_share_in_context']
    row['pred_generalise_var'] = float(np.mean(pt * (1 - pt) / nin))
    row['pred_generalise_fixed'] = float(np.mean(pt ** nin + (1 - pt) ** nin))
    # Memorisation: does model g-1's prediction for a sequence (stored in pool g)
    # follow that sequence's share in its own training pool (pool g-1)?
    if g > 0 and 'symbol_repeat_prob' in pool and prev_share is not None:
      q = pool['symbol_repeat_prob'] / np.maximum(pool['symbol_repeat_prob'] + pool['symbol_other_prob'], 1e-12)
      q_ctx = np.bincount(ids, weights=q, minlength=n_ctx) / copies
      ok = ~np.isnan(prev_share)
      x, y = prev_share[ok], q_ctx[ok]
      row['model_q_mean'] = float(q_ctx.mean())
      row['model_q_var'] = float(q_ctx.var())
      row['memorisation_corr'] = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float('nan')
      row['memorisation_slope'] = float(np.polyfit(x, y, 1)[0]) if x.std() > 0 else float('nan')
      if prev_q is not None:
        row['persistence_corr'] = float(np.corrcoef(prev_q, q_ctx)[0, 1])
      # Spread expected if pool g is binomial around model g-1's own q per sequence
      nin = st['n_in']
      has = nin > 0
      row['pred_given_model_var'] = float(np.mean(q_ctx[has] * (1 - q_ctx[has]) / nin[has]) + q_ctx[has].var())
      prev_q = q_ctx
    row.update(model_metrics(run))
    rows.append(row)
    prev_share = st['share']

  # Predictions, using the real copy counts (in-context copies of generation 0)
  m_c = np.maximum(copies, 1).astype(float)
  p = rows[0]['repeat_share_in_context']
  rng = np.random.default_rng(args.seed)
  G = len(rows)
  # Wright-Fisher (memorise): simulate per sequence from the real-data start
  wf_var = np.zeros(G)
  wf_fix = np.zeros(G)
  for _ in range(args.sims):
    s = rng.binomial(copies, p) / m_c
    for t in range(G):
      if t > 0:
        s = rng.binomial(copies, s) / m_c
      wf_var[t] += s.var() / args.sims
      wf_fix[t] += np.mean((s == 0) | (s == 1)) / args.sims
  for t, row in enumerate(rows):
    row['pred_memorise_var'] = float(np.mean(p * (1 - p) * (1 - (1 - 1 / m_c) ** (t + 1))))
    row['pred_memorise_var_sim'] = float(wf_var[t])
    row['pred_memorise_fixed_sim'] = float(wf_fix[t])

  cols = list(rows[-1].keys() | rows[0].keys())
  order = ['generation', 'query_error', 'repeat_share', 'other_share', 'out_of_context',
           'repeat_share_in_context', 'share_var', 'pred_generalise_var', 'pred_memorise_var',
           'pred_given_model_var', 'fixed_frac', 'pred_generalise_fixed', 'pred_memorise_fixed_sim',
           'model_q_mean', 'model_q_var', 'memorisation_corr', 'memorisation_slope', 'persistence_corr']
  cols = [c for c in order if c in cols] + sorted(c for c in cols if c not in order)
  with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
      w.writerow({k: r.get(k, '') for k in cols})
  with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
    json.dump({'condition': args.condition, 'gen0_run': args.gen0_run, 'n_sequences': n_ctx,
               'mean_copies': float(copies.mean()), 'rows': rows}, f, indent=2)

  nan = float('nan')
  print('\nPool g (written by model g-1). var: spread of the repeat share across sequences;'
        ' gen/mem/model = predicted if the model generalises / memorises / from model g-1\'s own q')
  print(f"{'gen':>3} {'q.err':>7} {'out-ctx':>8} {'share|ctx':>9} {'var obs':>8} {'var gen':>8} {'var mem':>8} "
        f"{'var model':>9} {'fix obs':>8} {'fix gen':>8} {'fix mem':>8} {'Var(q)':>7} {'mem.r':>6} {'pers.r':>6}")
  for r in rows:
    print(f"{r['generation']:>3} {r['query_error']:>7.4f} {r['out_of_context']:>8.4f} "
          f"{r['repeat_share_in_context']:>9.4f} {r['share_var']:>8.4f} {r['pred_generalise_var']:>8.4f} "
          f"{r['pred_memorise_var']:>8.4f} {r.get('pred_given_model_var', nan):>9.4f} {r['fixed_frac']:>8.4f} "
          f"{r['pred_generalise_fixed']:>8.4f} {r['pred_memorise_fixed_sim']:>8.4f} {r.get('model_q_var', nan):>7.4f} "
          f"{r.get('memorisation_corr', nan):>6.3f} {r.get('persistence_corr', nan):>6.3f}")

  # Figure
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  gens = [r['generation'] for r in rows]
  obs_c, gen_c, mem_c = '#2a78d6', '#7f7f7f', '#eb6834'
  fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
  ax[0].plot(gens, [r['repeat_share_in_context'] for r in rows], 'o-', color=obs_c, label='pool g (from model g-1)')
  vs = [(r['generation'], r['val_symbol_repeat_share']) for r in rows if 'val_symbol_repeat_share' in r]
  if vs:
    ax[0].plot(*zip(*vs), 's--', color='#1baf7a', label='model g on real data')
  ax[0].axhline(p, color=gen_c, lw=0.8, ls=':')
  ax[0].set(title='(a) Repeat share (in-context symbols)', xlabel='Generation', ylabel='Share')
  ax[0].legend(fontsize=7, frameon=False)
  ax[1].plot(gens, [r['share_var'] for r in rows], 'o-', color=obs_c, label='observed')
  ax[1].plot(gens, [r['pred_generalise_var'] for r in rows], '-', color=gen_c, label='generalise (no drift)')
  ax[1].plot(gens, [r['pred_memorise_var'] for r in rows], '-', color=mem_c, label='memorise (Wright-Fisher)')
  gm = [(r['generation'], r['pred_given_model_var']) for r in rows if 'pred_given_model_var' in r]
  if gm:
    ax[1].plot(*zip(*gm), 'x', color='black', label="from model g-1's q")
  ax[1].set(title='(b) Spread of share across sequences', xlabel='Generation', ylabel='Variance')
  ax[1].legend(fontsize=7, frameon=False)
  mq = [(r['generation'] - 1, r['model_q_var']) for r in rows if 'model_q_var' in r]
  if mq:
    ax[2].plot(*zip(*mq), 'o-', color='#9b59b6')
  ax[2].set(title='(c) Uneven preference of model g: Var(q)', xlabel='Model generation', ylabel='Variance')
  mc = [(r['generation'] - 1, r['memorisation_corr']) for r in rows if 'memorisation_corr' in r]
  pc = [(r['generation'] - 1, r['persistence_corr']) for r in rows if 'persistence_corr' in r]
  if mc:
    ax[3].plot(*zip(*mc), 'o-', color=mem_c, label='memorisation: q vs own training share')
  if pc:
    ax[3].plot(*zip(*pc), 's--', color='#7f7f7f', label='persistence: q vs previous model q')
  ax[3].axhline(0, color='black', lw=0.6)
  ax[3].set(title='(d) Memorisation and persistence', xlabel='Model generation', ylabel='Correlation',
            ylim=(-0.1, 1.0))
  ax[3].legend(fontsize=7, frameon=False)
  for i, a in enumerate(ax):
    a.spines[['top', 'right']].set_visible(False)
    a.set_xticks(gens if i < 2 else (gens[:-1] or [0]))
  fig.tight_layout()
  for ext in ['pdf', 'png']:
    fig.savefig(os.path.join(out_dir, f'drift.{ext}'), dpi=150)
  print(f'\nWrote {out_dir}/summary.csv, summary.json, drift.pdf/png')


if __name__ == '__main__':
  main()
