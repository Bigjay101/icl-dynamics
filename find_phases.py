'''
Finds (1) around which iterations induction heads form and (2) where the
training loss plateaus, from the log.h5 files written by main.py.

Induction head formation is read off an in-context evaluator. By default this
is fsl_val_rl/acc: labels are relabeled per sequence with held-out label
assignments, so the only way to beat chance is to copy the label from context
(i.e. use an induction head). We report the iterations at which the metric has
completed 10% / 50% / 90% of its rise from its starting value to its peak,
plus the eval interval with the steepest rise.

Loss plateaus are found on the (dense) train loss. The loss is smoothed with a
rolling mean, and an iteration is "flat" if the smoothed loss changes by less
than --plateau_tol (relative) over the next --plateau_window iterations.
Contiguous flat stretches longer than --min_plateau_len are reported.

Usage:
  python find_phases.py ih_paper_reprod/main_paper_is5_ih3_pt2/omniglot50_rl5
  python find_phases.py ih_paper_reprod/main_paper_is5_ih3_pt2 --plot
  (a folder is searched recursively for runs, i.e. folders containing log.h5)
'''

import argparse
import os
import json

import numpy as np
import h5py as h5


def find_runs(paths):
  runs = []
  for p in paths:
    if os.path.isfile(os.path.join(p, 'log.h5')):
      runs.append(p)
    else:
      for root, _, files in os.walk(p):
        if 'log.h5' in files:
          runs.append(root)
  return sorted(set(runs))


def dedupe_keep_last(iters):
  '''
  Restarted runs append to the same log.h5, so the same iteration can be logged
  more than once. Returns indices of the last occurrence of each iteration,
  sorted by iteration.
  '''
  iters = np.asarray(iters)
  rev_idx = len(iters) - 1 - np.unique(iters[::-1], return_index=True)[1]
  return np.sort(rev_idx)


def load_run(run_dir, metric):
  with h5.File(os.path.join(run_dir, 'log.h5'), 'r') as f:
    train_iter = f['train_iter'][:]
    train_loss = f['train_loss'][:]
    eval_iter = f['eval_iter'][:]
    if metric not in f:
      raise KeyError("metric '{}' not in log. Available evaluators: {}".format(
        metric, [k for k in f if isinstance(f[k], h5.Group)]))
    eval_metric = f[metric][:].mean(axis=-1)

  keep = dedupe_keep_last(train_iter)
  train_iter, train_loss = train_iter[keep], train_loss[keep]
  keep = dedupe_keep_last(eval_iter)
  eval_iter, eval_metric = eval_iter[keep], eval_metric[keep]
  return train_iter, train_loss, eval_iter, eval_metric


def first_crossing(x, y, level):
  '''First x at which y reaches level, linearly interpolated between points.'''
  above = np.nonzero(y >= level)[0]
  if len(above) == 0:
    return None
  j = above[0]
  if j == 0:
    return float(x[0])
  x0, x1, y0, y1 = x[j-1], x[j], y[j-1], y[j]
  return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))


def ih_formation(eval_iter, eval_metric, fracs=(0.1, 0.5, 0.9)):
  start = float(eval_metric[0])
  peak_ind = int(np.argmax(eval_metric))
  peak = float(eval_metric[peak_ind])
  rise = peak - start
  out = {'start': start,
         'peak': peak,
         'peak_iter': int(eval_iter[peak_ind]),
         'final': float(eval_metric[-1]),
         'rise': rise}
  for frac in fracs:
    level = start + frac * rise
    out['iter_{}pct'.format(int(frac * 100))] = first_crossing(eval_iter, eval_metric, level)

  if len(eval_iter) > 1:
    slopes = np.diff(eval_metric) / np.maximum(np.diff(eval_iter), 1)
    j = int(np.argmax(slopes))
    out['steepest_rise_interval'] = [int(eval_iter[j]), int(eval_iter[j+1])]
    out['steepest_rise_per_1k_iters'] = float(slopes[j] * 1000)

  # Coarse eval schedules make the crossing iterations imprecise, so we report
  # the eval spacing around the 50% point as a resolution estimate.
  mid = out.get('iter_50pct')
  if mid is not None:
    j = min(np.searchsorted(eval_iter, mid), len(eval_iter) - 1)
    out['eval_resolution_at_50pct'] = int(eval_iter[j] - eval_iter[max(j-1, 0)])
  return out


def smooth(y, window_pts):
  window_pts = max(1, min(window_pts, len(y)))
  kernel = np.ones(window_pts) / window_pts
  # 'valid' then pad the edges, so the smoothed curve isn't biased at the ends
  sm = np.convolve(y, kernel, mode='valid')
  pad_l = (len(y) - len(sm)) // 2
  pad_r = len(y) - len(sm) - pad_l
  return np.concatenate([np.full(pad_l, sm[0]), sm, np.full(pad_r, sm[-1])])


def loss_plateaus(train_iter, train_loss, smooth_iters, window_iters, tol, min_len_iters):
  step = float(np.median(np.diff(train_iter)))
  sm = smooth(train_loss, int(round(smooth_iters / step)))

  # Relative change in smoothed loss over the next window_iters iterations
  ahead = np.searchsorted(train_iter, train_iter + window_iters)
  ahead = np.minimum(ahead, len(train_iter) - 1)
  rel_change = np.abs(sm[ahead] - sm) / np.maximum(np.abs(sm), 1e-8)
  # Points too close to the end to look ahead a full window inherit the last
  # full-window value
  full = train_iter + window_iters <= train_iter[-1]
  if np.any(full):
    rel_change[~full] = rel_change[full][-1]
  flat = rel_change < tol

  plateaus = []
  i = 0
  while i < len(flat):
    if not flat[i]:
      i += 1
      continue
    j = i
    while j + 1 < len(flat) and flat[j+1]:
      j += 1
    # A point is flat if the loss doesn't move over the *next* window, so the
    # plateau extends to the end of the last flat point's window
    end_ind = ahead[j]
    if train_iter[end_ind] - train_iter[i] >= min_len_iters:
      plateaus.append({'start_iter': int(train_iter[i]),
                       'end_iter': int(train_iter[end_ind]),
                       'mean_loss': float(np.mean(train_loss[i:end_ind+1])),
                       'reaches_end_of_training': bool(end_ind == len(train_iter) - 1)})
    i = j + 1

  # Merge plateaus that overlap because of the window extension, or that are
  # separated by a gap shorter than the window (a noise blip, not a new phase)
  merged = []
  for p in plateaus:
    if merged and p['start_iter'] - merged[-1]['end_iter'] <= window_iters:
      prev = merged[-1]
      lo = np.searchsorted(train_iter, prev['start_iter'])
      hi = np.searchsorted(train_iter, p['end_iter'])
      prev['end_iter'] = p['end_iter']
      prev['mean_loss'] = float(np.mean(train_loss[lo:hi+1]))
      prev['reaches_end_of_training'] = p['reaches_end_of_training']
    else:
      merged.append(p)
  return merged, sm


def analyze(run_dir, opts):
  train_iter, train_loss, eval_iter, eval_metric = load_run(run_dir, opts.metric)
  result = {'metric': opts.metric}

  if len(eval_iter) < 2 or eval_iter[-1] == eval_iter[0]:
    result['ih_formation'] = None
    result['note'] = 'not enough distinct eval iterations'
  else:
    result['ih_formation'] = ih_formation(eval_iter, eval_metric)
    rise = result['ih_formation']['rise']
    if rise < opts.min_rise:
      result['note'] = ('metric only rose by {:.3f} (< --min_rise {}); '
                        'induction heads may not have formed').format(rise, opts.min_rise)

  smoothed = None
  if len(train_iter) < 2:
    result['loss_plateaus'] = None
  else:
    plateaus, smoothed = loss_plateaus(train_iter, train_loss,
                                       smooth_iters=opts.smooth_iters,
                                       window_iters=opts.plateau_window,
                                       tol=opts.plateau_tol,
                                       min_len_iters=opts.min_plateau_len)
    result['loss_plateaus'] = plateaus
    result['initial_train_loss'] = float(smoothed[0])
    result['final_train_loss'] = float(smoothed[-1])
    final = [p for p in plateaus if p['reaches_end_of_training']]
    result['converged_iter'] = final[0]['start_iter'] if final else None

  curves = (train_iter, train_loss, smoothed, eval_iter, eval_metric)
  return result, curves


def plot_run(name, result, curves, opts, fname):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  train_iter, train_loss, smoothed, eval_iter, eval_metric = curves
  fig, (ax_m, ax_l) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

  ax_m.plot(eval_iter, eval_metric, 'o-', color='C0')
  ax_m.set_ylabel(opts.metric)
  ax_m.set_title(name)

  if len(train_iter) > 0:
    ax_l.plot(train_iter, train_loss, color='C1', alpha=0.25, lw=0.5, label='train loss')
    ax_l.plot(train_iter, smoothed, color='C1', label='smoothed')
    for k, p in enumerate(result['loss_plateaus'] or []):
      for ax in (ax_m, ax_l):
        ax.axvspan(p['start_iter'], p['end_iter'], color='grey', alpha=0.15,
                   label='loss plateau' if (k == 0 and ax is ax_l) else None)
  ax_l.set_ylabel('train loss')
  ax_l.set_xlabel('iteration (sequences seen)')

  ih = result['ih_formation']
  if ih is not None:
    for key, ls in [('iter_10pct', ':'), ('iter_50pct', '--'), ('iter_90pct', ':')]:
      if ih.get(key) is not None:
        for ax in (ax_m, ax_l):
          ax.axvline(ih[key], color='C3', ls=ls, lw=1,
                     label='IH {}'.format(key[5:]) if ax is ax_m else None)
  ax_m.legend(fontsize=8)
  ax_l.legend(fontsize=8)
  fig.tight_layout()
  fig.savefig(fname, dpi=150)
  plt.close(fig)


def fmt_iter(x):
  return 'n/a' if x is None else '{:,.0f}'.format(x)


def print_summary(name, r):
  print('=' * 60)
  print(name)
  ih = r['ih_formation']
  if ih is not None:
    print('  Induction head formation ({}):'.format(r['metric']))
    print('    {:.3f} -> peak {:.3f} at iter {} (final {:.3f})'.format(
      ih['start'], ih['peak'], fmt_iter(ih['peak_iter']), ih['final']))
    print('    10% / 50% / 90% of rise at iters {} / {} / {}'.format(
      fmt_iter(ih['iter_10pct']), fmt_iter(ih['iter_50pct']), fmt_iter(ih['iter_90pct'])))
    if 'steepest_rise_interval' in ih:
      print('    steepest rise between iters {} and {}'.format(
        *map(fmt_iter, ih['steepest_rise_interval'])))
    if 'eval_resolution_at_50pct' in ih:
      print('    (eval spacing near 50% point: {} iters)'.format(
        fmt_iter(ih['eval_resolution_at_50pct'])))
  if r.get('loss_plateaus') is not None:
    print('  Train loss: {:.3f} -> {:.3f}'.format(r['initial_train_loss'], r['final_train_loss']))
    if len(r['loss_plateaus']) == 0:
      print('    no plateaus found (try a larger --plateau_tol)')
    for p in r['loss_plateaus']:
      print('    plateau at loss ~{:.3f} from iter {} to {}{}'.format(
        p['mean_loss'], fmt_iter(p['start_iter']), fmt_iter(p['end_iter']),
        ' (end of training)' if p['reaches_end_of_training'] else ''))
    print('  Converged (final plateau starts) at iter', fmt_iter(r['converged_iter']))
  if 'note' in r:
    print('  NOTE:', r['note'])


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('paths', nargs='+',
                      help='Run folders (containing log.h5) or folders to search for runs')
  parser.add_argument('--metric', default='fsl_val_rl/acc',
                      help='Eval metric (evaluator/metric in log.h5) used to detect induction heads')
  parser.add_argument('--min_rise', type=float, default=0.05,
                      help='Warn if the metric rises by less than this')
  parser.add_argument('--smooth_iters', type=int, default=2000,
                      help='Rolling-mean window for the train loss, in iterations')
  parser.add_argument('--plateau_window', type=int, default=5000,
                      help='Look-ahead window (iterations) for deciding if the loss is flat')
  parser.add_argument('--plateau_tol', type=float, default=0.02,
                      help='Max relative change in smoothed loss over the window to count as flat')
  parser.add_argument('--min_plateau_len', type=int, default=10000,
                      help='Minimum plateau length in iterations')
  parser.add_argument('--out', default='investigate_out',
                      help='Folder to write phases.json (and plots) to')
  parser.add_argument('--plot', action='store_true', help='Save a plot per run')
  opts = parser.parse_args()

  runs = find_runs(opts.paths)
  if len(runs) == 0:
    raise SystemExit('No runs (folders containing log.h5) found in {}'.format(opts.paths))
  os.makedirs(opts.out, exist_ok=True)

  all_results = {}
  for run_dir in runs:
    name = os.path.relpath(run_dir)
    result, curves = analyze(run_dir, opts)
    all_results[name] = result
    print_summary(name, result)
    if opts.plot and result['ih_formation'] is not None:
      fname = os.path.join(opts.out, 'phases_{}.png'.format(name.replace(os.sep, '__')))
      plot_run(name, result, curves, opts, fname)
      print('  plot ->', fname)

  out_file = os.path.join(opts.out, 'phases.json')
  with open(out_file, 'w') as f:
    json.dump(all_results, f, indent=2)
  print('=' * 60)
  print('Wrote', out_file)
