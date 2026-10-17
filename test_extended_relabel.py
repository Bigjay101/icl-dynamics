"""Step 4 test: extended relabelling.

    python test_extended_relabel.py <model_run_folder> <pool_in> <pool_out>

<pool_out> must already have been made by relabel.py from <pool_in> with the
model in <model_run_folder>. Checks that
  1. the context and query items are untouched, and the true tokens are copied,
  2. rows kept real have all three true tokens,
  3. a generated symbol's exemplar is the context item's (or 0 if not in context),
  4. the three tokens were generated in sequence: re-running the model on the
     finished pool gives back the same choices (argmax), or the shares match
     the probabilities they were drawn from (sample),
  5. the summary agrees with the pool.
"""
import json
import os
import sys

import h5py
import jax.numpy as jnp
import numpy as np

import relabel

run, pool_in, pool_out = sys.argv[1:4]
ok = True


def check(name, cond):
  global ok
  print(('PASS ' if cond else 'FAIL ') + name)
  ok = ok and bool(cond)


with h5py.File(pool_in, 'r') as f:
  a = {k: f[k][()] for k in f}
with h5py.File(pool_out, 'r') as f:
  b = {k: f[k][()] for k in f}
  method = f.attrs['method']
  check('output pool is marked extended, one generation later',
        bool(f.attrs['extended']) and f.attrs['generation'] == 1 + int(h5py.File(pool_in, 'r').attrs['generation']))
summary = json.load(open(os.path.splitext(pool_out)[0] + '_summary.json'))
n = len(a['labels'])
kept = b['kept_real']
model_rows = ~kept

# 1. Untouched items and copied true tokens
check('context and query classes unchanged', np.array_equal(a['class_idxs'][:, :3], b['class_idxs'][:, :3]))
check('context and query exemplars unchanged', np.array_equal(a['exemplar_inds'][:, :3], b['exemplar_inds'][:, :3]))
check('context labels unchanged', np.array_equal(a['labels'][:, :2], b['labels'][:, :2]))
check('item types unchanged', np.array_equal(a['idx_types'], b['idx_types']))
check('true tokens copied unchanged',
      all(np.array_equal(a[k], b[k]) for k in ['true_query_label'] + relabel.EXTENDED_TRUE_KEYS))

# 2. Kept rows
check(f'kept rows ({kept.mean():.2%}) have all three true tokens',
      np.array_equal(b['labels'][kept, 2], a['true_query_label'][kept]) and
      np.array_equal(b['class_idxs'][kept, 3], a['true_next_class'][kept]) and
      np.array_equal(b['exemplar_inds'][kept, 3], a['true_next_exemplar'][kept]) and
      np.array_equal(b['labels'][kept, 3], a['true_next_label'][kept]))

# 3. Exemplar of the generated symbol
c, e = b['class_idxs'], b['exemplar_inds']
expected_ex = np.where(c[:, 3] == c[:, 0], e[:, 0], np.where(c[:, 3] == c[:, 1], e[:, 1], 0))
check('next exemplar = context item\'s exemplar, or 0 if not in context',
      np.array_equal(e[model_rows, 3], expected_ex[model_rows]))
check('generated symbols are training classes (0-49)', c[:, 3].min() >= 0 and c[:, 3].max() < 50)

# 4. Generated in sequence
model, fwd_fn, data, _ = relabel.load_model(os.path.join(run, 'config.json'), relabel.last_checkpoint(run))
sub = np.arange(min(n, 8192))
out = relabel.extended_outputs(model, fwd_fn, data, jnp.asarray(c[sub]), jnp.asarray(e[sub]),
                               jnp.asarray(b['labels'][sub]))
m = model_rows[sub]
if method == 'argmax':
  # Each choice must be the model's argmax given the tokens generated before it.
  # On a GPU, two separately compiled forward passes can differ in the last
  # decimal place, which flips near-ties; those are allowed, anything else is not.
  def argmax_check(name, logits, chosen):
    logits = np.asarray(logits)[m]
    chosen = chosen[m]
    mismatch = logits.argmax(-1) != chosen
    # How far the stored choice is below the top logit, for each mismatch
    gap = logits.max(-1) - logits[np.arange(len(chosen)), chosen]
    worst = float(gap[mismatch].max()) if mismatch.any() else 0.0
    check(f'{name} ({int(mismatch.sum())} of {len(chosen)} differ, all near-ties: largest gap {worst:.2e})',
          worst < 1e-3)
  argmax_check('query label = argmax given the context', out['query'], b['labels'][sub, 2])
  argmax_check('next symbol = argmax given the generated query label', out['symbol'], c[sub, 3])
  argmax_check('next label = argmax given the generated symbol', out['next_label'], b['labels'][sub, 3])
else:
  rep = (c[:, 3] == c[:, 2])[model_rows].mean()
  prob = b['symbol_repeat_prob'][model_rows].mean()
  check(f'sampled repeat share {rep:.4f} matches the mean probability {prob:.4f} it was drawn from',
        abs(rep - prob) < 0.005)
  other_class = np.where(c[:, 0] == c[:, 2], c[:, 1], c[:, 0])
  oth = (c[:, 3] == other_class)[model_rows].mean()
  prob_o = b['symbol_other_prob'][model_rows].mean()
  check(f'sampled other share {oth:.4f} matches the mean probability {prob_o:.4f}', abs(oth - prob_o) < 0.005)
  # The stored query logits are the model's, given the context (sampling uses them)
  check('stored query logits match a fresh forward pass',
        np.allclose(np.asarray(out['query']), b['logits'][sub], atol=1e-4))

# 5. Summary agrees with the pool
repeat = c[:, 3] == c[:, 2]
check('summary repeat share matches the pool', abs(summary['next_repeat_share'] - repeat.mean()) < 1e-9)
in_ctx = (c[:, 3] == c[:, 0]) | (c[:, 3] == c[:, 1])
check('summary out-of-context share matches the pool', abs(summary['next_out_of_context'] - (~in_ctx).mean()) < 1e-9)
check('summary query error rate matches the pool',
      abs(summary['error_rate'] - (b['labels'][:, 2] != b['true_query_label']).mean()) < 1e-9)

print(f"\nGeneration {summary['generation']} ({method}): query wrong {summary['error_rate']:.4%}, "
      f"repeat {summary['next_repeat_share']:.4%}, other {summary['next_other_share']:.4%}, "
      f"not in context {summary['next_out_of_context']:.4%}, "
      f"next label matches its symbol {summary['next_label_matches_symbol']:.4%}")
print('\nALL PASSED' if ok else '\nSOME CHECKS FAILED')
sys.exit(0 if ok else 1)
