"""Step 1 test: the extended sampler.

Run from the repo folder:  python test_extended_sampler.py
Checks that
  1. the base sampler (extended off) is unchanged,
  2. with --extended, the first 3 items are exactly the base sequence,
  3. the 4th item is always a context item, with that item's exemplar and label,
  4. the repeat share matches p_repeat (0.5, and the extremes 0 and 1),
  5. the wrap-around case (query matches the 2nd item, next is the 1st) occurs.
"""
import importlib.util
import subprocess
import sys
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import main_utils
import samplers

# The same settings as the collapse runs (run_generations.sh)
ARGS = ("--data_file omniglot_resnet18_randomized_order_s0.h5 --mixing_coeffs 1.0 --pt_burstiness 1 "
        "--train_context_len 2 --fs_relabel 5 --fs_relabel_split 8 2 0 --exemplar_split 1 4 0 "
        "--class_split 50 1473 100 --train_bs 32").split()
opts = main_utils.create_parser().parse_args(ARGS)

data = main_utils.get_data_from_opts(opts)
splits = main_utils.get_splits_from_opts(opts, data.shape)


def build(module, batch, relabel='train', exemplars='train', classes='train', **kw):
  """Build a sampler the same way main.py does."""
  n = len(splits['class'][classes])
  class_sampler = partial(module.get_constant_burst_seq_idxs,
                          classes=splits['class'][classes],
                          class_distr=jnp.ones(n) / n,
                          num_seqs=batch, context_len=2, burstiness=1,
                          distractor=1, no_support=0, unique_rest=0)
  exemplar_sampler = partial(module.get_exemplar_inds,
                             allowed_inds=splits['exemplar'][exemplars],
                             match_query_and_distractors=False)
  return jax.jit(partial(module.make_data_sampler(
    class_sampler, exemplar_sampler, fs_relabel=splits['relabeling'][relabel], **kw), data=data))


# The original samplers.py, from the last commit, for comparison
orig_src = subprocess.run(['git', 'show', 'HEAD:samplers.py'], capture_output=True, text=True, check=True).stdout
spec = importlib.util.spec_from_loader('samplers_orig', loader=None)
samplers_orig = importlib.util.module_from_spec(spec)
exec(orig_src, samplers_orig.__dict__)

key = jax.random.PRNGKey(123)
B = 20000
ok = True


def check(name, cond):
  global ok
  print(('PASS ' if cond else 'FAIL ') + name)
  ok = ok and bool(cond)


# 1. Base sampler unchanged
new = build(samplers, B)(key)
old = build(samplers_orig, B)(key)
check('base sampler unchanged (extended off)',
      all(np.array_equal(np.asarray(new[k]), np.asarray(old[k])) for k in old))

for relabel, exemplars, classes in [('train', 'train', 'train'), ('val', 'train', 'train'),
                                    ('train', 'val', 'train'), ('train', 'train', 'test')]:
  tag = f'[relabel={relabel}, exemplars={exemplars}, classes={classes}]'
  base = build(samplers, B, relabel, exemplars, classes)(key)
  ext = build(samplers, B, relabel, exemplars, classes, extended=True, p_repeat=0.5)(key)
  c, e, l, t = (np.asarray(ext[k]) for k in ['class_idxs', 'exemplar_inds', 'labels', 'idx_types'])
  rep = np.asarray(ext['next_is_repeat'])

  # 2. First 3 items identical to base
  check(f'{tag} first 3 items equal the base sequence',
        all(np.array_equal(np.asarray(ext[k])[:, :3], np.asarray(base[k])) for k in base))
  check(f'{tag} shapes are 4 items', c.shape == (B, 4) and np.asarray(ext['examples']).shape[:2] == (B, 4))

  # 3. The 4th item is a context item, with its exemplar and label
  rows = np.arange(B)
  match = np.argmax(c[:, :2] == c[:, 2:3], axis=1)
  pos = np.where(rep, match, 1 - match)
  check(f'{tag} exactly one context item matches the query', np.all((c[:, :2] == c[:, 2:3]).sum(1) == 1))
  check(f'{tag} 4th class = chosen context class', np.array_equal(c[:, 3], c[rows, pos]))
  check(f'{tag} 4th exemplar = that item\'s exemplar', np.array_equal(e[:, 3], e[rows, pos]))
  check(f'{tag} 4th label = that item\'s label', np.array_equal(l[:, 3], l[rows, pos]))
  check(f'{tag} repeat <=> 4th class equals query class', np.array_equal(c[:, 3] == c[:, 2], rep))
  check(f'{tag} 4th example = data[class, exemplar]',
        np.allclose(np.asarray(ext['examples'])[:, 3], np.asarray(data)[c[:, 3], e[:, 3]]))
  check(f'{tag} 4th item type is NEXT', np.all(t[:, 3] == int(samplers.ItemType.NEXT)))

  # 4. Repeat share
  share = rep.mean()
  check(f'{tag} repeat share {share:.4f} within 0.5 +/- 0.015', abs(share - 0.5) < 0.015)

  # 5. Wrap-around case occurs (query matches the 2nd item, next symbol is the 1st)
  wrap = ((match == 1) & ~rep).mean()
  check(f'{tag} wrap-around share {wrap:.3f} is about 0.25', abs(wrap - 0.25) < 0.02)

for p in [0.0, 1.0]:
  ext = build(samplers, B, extended=True, p_repeat=p)(key)
  share = float(np.asarray(ext['next_is_repeat']).mean())
  check(f'p_repeat={p}: repeat share is exactly {p} (got {share})', share == p)

print('\nALL PASSED' if ok else '\nSOME CHECKS FAILED')
sys.exit(0 if ok else 1)
