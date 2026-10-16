"""Step 3 check of an extended-version training run.

    python check_extended_run.py <run_folder> [--full]

Checks the run's pool (4 items per sequence, the real next item stored, the
next symbol always in context, repeat share about p_repeat), that the new
metrics were logged, that symbol metrics are NaN on the held-out-class
evaluator, and prints the final validation metrics. With --full (a 1M-sequence
run) it also checks the model learned all three predictions.
"""
import json
import os
import sys

import h5py
import numpy as np

run = sys.argv[1]
full = '--full' in sys.argv
ok = True


def check(name, cond):
  global ok
  print(('PASS ' if cond else 'FAIL ') + name)
  ok = ok and bool(cond)


with open(os.path.join(run, 'config.json')) as f:
  cfg = json.load(f)
check('config has extended = true', cfg.get('extended') is True)
p = cfg.get('p_repeat', 0.5)

with h5py.File(os.path.join(run, 'pool_gen0.h5'), 'r') as f:
  c, e, l = f['class_idxs'][()], f['exemplar_inds'][()], f['labels'][()]
  check(f'pool has 4 items per sequence {c.shape}', c.shape[1] == 4 and l.shape[1] == 4)
  check('pool stores the real next item',
        all(k in f for k in ['true_query_label', 'true_next_class', 'true_next_exemplar',
                             'true_next_label', 'true_next_is_repeat']))
  check('true_query_label is the query (item 2)', np.array_equal(f['true_query_label'][()], l[:, 2]))
  check('next symbol always one of the two context symbols',
        np.all((c[:, 3] == c[:, 0]) | (c[:, 3] == c[:, 1])))
  ctx_pos = np.where(c[:, 3] == c[:, 0], 0, 1)
  rows = np.arange(len(c))
  check('next label = that context item\'s label', np.array_equal(l[:, 3], l[rows, ctx_pos]))
  share = f['true_next_is_repeat'][()].mean()
  check(f'pool repeat share {share:.4f} close to p_repeat {p}', abs(share - p) < 0.01)

with h5py.File(os.path.join(run, 'log.h5'), 'r') as f:
  ev = 'fsl_val_rl'
  new = ['next_label_in_context_acc', 'next_label_loss', 'symbol_loss', 'symbol_repeat_prob',
         'symbol_other_prob', 'symbol_use_context_prob', 'symbol_repeat_share', 'symbol_pred_in_context',
         'total_loss']
  check('new metrics logged for the validation set', all(f'{ev}/{m}' in f for m in new))
  check('symbol metrics are NaN on the held-out-class set (cannot name those classes)',
        np.all(np.isnan(f['fsl_test_class/symbol_loss'][-1])))
  check('query metrics are finite on the held-out-class set',
        np.all(np.isfinite(f['fsl_test_class/in_context_acc'][-1])))
  check('training loss finite', np.all(np.isfinite(f['train_loss'][()])))

  final = {m: float(np.mean(f[f'{ev}/{m}'][-1])) for m in
           ['in_context_acc', 'loss', 'use_context_prob', 'next_label_in_context_acc', 'next_label_loss',
            'symbol_loss', 'symbol_use_context_prob', 'symbol_repeat_prob', 'symbol_other_prob',
            'symbol_pred_in_context']}
  final['symbol_repeat_share'] = final['symbol_repeat_prob'] / (final['symbol_repeat_prob'] + final['symbol_other_prob'])
  print('\nFinal validation metrics (fsl_val_rl), at', int(f['eval_iter'][-1]), 'sequences:')
  for m, v in final.items():
    print(f'  {m:28s} {v:.4f}')

if full:
  print()
  check('query label learned (in_context_acc > 0.95)', final['in_context_acc'] > 0.95)
  check('next label learned (next_label_in_context_acc > 0.95)', final['next_label_in_context_acc'] > 0.95)
  check('next symbol stays in context (symbol_use_context_prob > 0.95)', final['symbol_use_context_prob'] > 0.95)
  check(f'repeat share close to p_repeat (|{final["symbol_repeat_share"]:.3f} - {p}| < 0.05)',
        abs(final['symbol_repeat_share'] - p) < 0.05)
  # Best possible symbol loss with a 50/50 choice between two symbols is ln 2
  check(f'symbol loss near its floor ln 2 = 0.693 (got {final["symbol_loss"]:.3f})',
        final['symbol_loss'] < 0.693 + 0.1)

print('\nALL PASSED' if ok else '\nSOME CHECKS FAILED')
sys.exit(0 if ok else 1)
