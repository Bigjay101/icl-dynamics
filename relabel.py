'''
Makes the training pool for generation n from the model of generation n-1.

The previous generation's model is run over every sequence of the previous
pool, and its prediction for the query replaces the query label. Everything
else in the pool (classes, exemplars, order, context labels) is copied
unchanged, as is true_query_label, so every generation can be compared with
the real labels.

Labels can be chosen by
  argmax: the most likely label
  sample: a label drawn from softmax(logits / temperature)
Optionally a fixed fraction of rows (--keep_real_frac, chosen with
--keep_real_seed, so the same rows every generation) keeps the true label.

Usage (generation 1 from generation 0):
  python relabel.py --run_folder collapse/gen0_pool \
      --pool_in collapse/gen0_pool/pool_gen0.h5 \
      --pool_out collapse/argmax/pool_gen1.h5 --method argmax

The output pool also stores the model's logits for every row, the mask of
rows that kept their true label, and a summary json next to it.
'''

import argparse
import json
import os
import sys
from functools import partial

import numpy as np
import h5py as h5
import jax
import jax.numpy as jnp
import equinox as eqx

import main_utils
import opto
from samplers import ItemType

COPIED_KEYS = ['class_idxs', 'exemplar_inds', 'idx_types', 'true_query_label']


def last_checkpoint(run_folder):
  ckpt_dir = os.path.join(run_folder, 'checkpoints')
  ckpts = sorted(f for f in os.listdir(ckpt_dir) if f.endswith('.eqx'))
  if not ckpts:
    raise FileNotFoundError('No checkpoints in {}'.format(ckpt_dir))
  return os.path.join(ckpt_dir, ckpts[-1])


def load_model(config_path, ckpt_path):
  '''Rebuilds the model from its config.json and loads the checkpoint weights.'''
  # get_opts_from_json_file parses sys.argv, so hide this script's arguments
  argv, sys.argv = sys.argv, sys.argv[:1]
  try:
    opts = main_utils.get_opts_from_json_file(config_path)
  finally:
    sys.argv = argv
  data = main_utils.get_data_from_opts(opts)
  model = main_utils.get_model_from_opts(opts, input_shape=(data.shape[-1],))
  opt_state = main_utils.get_optimizer_from_opts(opts).init(eqx.filter(model, eqx.is_array))
  # Same structure main.py saves, so the leaves line up
  ckpt_fmt = {'iter': -1,
              'seeds': {'eval_model_seed': jax.random.PRNGKey(0),
                        'train_data_seed': jax.random.PRNGKey(0),
                        'train_model_seed': jax.random.PRNGKey(0)},
              'opt_state': opt_state,
              'model': model}
  ckpt = eqx.tree_deserialise_leaves(ckpt_path, ckpt_fmt)
  fwd_fn = opto.make_fn_from_opts(opts)
  return ckpt['model'], fwd_fn, data, int(ckpt['iter'])


@eqx.filter_jit
def query_logits(model, fwd_fn, data, class_idxs, exemplar_inds, labels):
  '''Logits at the query position, [bs, num_labels]. Same forward as eval_step.'''
  x = data[class_idxs, exemplar_inds]
  keys = jax.random.split(jax.random.PRNGKey(0), x.shape[0])  # unused: no dropout
  out = jax.vmap(partial(fwd_fn, model=model))(x=x, y=labels, key=keys)['out']
  return out[:, -1, :]


def all_query_logits(model, fwd_fn, data, pool, batch_size):
  n = len(pool['labels'])
  chunks = []
  for start in range(0, n, batch_size):
    sl = slice(start, start + batch_size)
    # The model never sees the query label (only labels[:-1] are embedded),
    # so the pool's current query labels don't influence the prediction
    chunks.append(np.asarray(query_logits(model, fwd_fn, data,
                                          jnp.asarray(pool['class_idxs'][sl]),
                                          jnp.asarray(pool['exemplar_inds'][sl]),
                                          jnp.asarray(pool['labels'][sl]))))
  return np.concatenate(chunks, axis=0)


def summarize(new_query, pool, kept_real):
  '''Error rates of the new query labels against the true labels.'''
  labels, types = pool['labels'], pool['idx_types']
  true = pool['true_query_label']
  ctx_labels, ctx_types = labels[:, :-1], types[:, :-1]
  distractor_label = np.where(ctx_types == ItemType.DISTRACTOR, ctx_labels, -1).max(axis=1)
  in_context = (ctx_labels == new_query[:, None]).any(axis=1)
  wrong = new_query != true
  n_labels = int(max(labels.max(), new_query.max())) + 1
  return {
    'n': int(len(true)),
    'error_rate': float(wrong.mean()),
    'error_distractor_label': float((wrong & (new_query == distractor_label)).mean()),
    'error_out_of_context': float((wrong & ~in_context).mean()),
    'kept_real_frac': float(kept_real.mean()),
    'error_rate_on_model_labelled_rows': float(wrong[~kept_real].mean()) if (~kept_real).any() else 0.0,
    'query_label_freq': (np.bincount(new_query, minlength=n_labels) / len(new_query)).round(5).tolist(),
    'true_label_freq': (np.bincount(true, minlength=n_labels) / len(true)).round(5).tolist(),
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--run_folder', required=True, help='Run folder of generation n-1 (has config.json and checkpoints/)')
  parser.add_argument('--ckpt', default=None, help='Checkpoint to use. Defaults to the last one in run_folder/checkpoints')
  parser.add_argument('--pool_in', required=True, help='Pool generation n-1 was trained on')
  parser.add_argument('--pool_out', required=True, help='Where to write the pool for generation n')
  parser.add_argument('--method', choices=['argmax', 'sample'], required=True)
  parser.add_argument('--temperature', type=float, default=1.0, help='Softmax temperature for --method sample')
  parser.add_argument('--relabel_seed', type=int, default=0, help='Seed for sampling labels')
  parser.add_argument('--keep_real_frac', type=float, default=0.0, help='Fraction of rows that keep the true label')
  parser.add_argument('--keep_real_seed', type=int, default=0, help='Seed choosing those rows (keep fixed across generations)')
  parser.add_argument('--batch_size', type=int, default=8192, help='Sequences per forward pass')
  args = parser.parse_args()

  if args.temperature <= 0:
    raise ValueError('--temperature must be > 0')
  if not 0.0 <= args.keep_real_frac <= 1.0:
    raise ValueError('--keep_real_frac must be in [0, 1]')
  if os.path.abspath(args.pool_out) == os.path.abspath(args.pool_in):
    raise ValueError('--pool_out must differ from --pool_in')

  ckpt_path = args.ckpt or last_checkpoint(args.run_folder)
  model, fwd_fn, data, ckpt_iter = load_model(os.path.join(args.run_folder, 'config.json'), ckpt_path)
  print('Model: {} (iteration {})'.format(ckpt_path, ckpt_iter))

  with h5.File(args.pool_in, 'r') as f:
    pool = {k: np.array(f[k]) for k in COPIED_KEYS + ['labels']}
    parent_gen = int(f.attrs.get('generation', -1))
  n = len(pool['labels'])
  print('Pool in: {} (generation {}, {} sequences)'.format(args.pool_in, parent_gen, n))

  logits = all_query_logits(model, fwd_fn, data, pool, args.batch_size)

  if args.method == 'argmax':
    model_query = logits.argmax(axis=-1)
  else:
    key = jax.random.PRNGKey(args.relabel_seed)
    model_query = np.asarray(jax.random.categorical(key, jnp.asarray(logits) / args.temperature, axis=-1))
  model_query = model_query.astype(pool['labels'].dtype)

  # Fixed subset of rows that keep the true label (same rows every generation)
  kept_real = np.random.default_rng(args.keep_real_seed).random(n) < args.keep_real_frac
  new_query = np.where(kept_real, pool['true_query_label'], model_query)

  new_labels = pool['labels'].copy()
  new_labels[:, -1] = new_query

  generation = parent_gen + 1
  os.makedirs(os.path.dirname(os.path.abspath(args.pool_out)), exist_ok=True)
  with h5.File(args.pool_out, 'w') as f:
    for k in COPIED_KEYS:
      f.create_dataset(k, data=pool[k])
    f.create_dataset('labels', data=new_labels)
    f.create_dataset('logits', data=logits.astype(np.float32))
    f.create_dataset('kept_real', data=kept_real)
    f.attrs['generation'] = generation
    f.attrs['source'] = 'synthetic'
    f.attrs['method'] = args.method
    f.attrs['temperature'] = args.temperature
    f.attrs['relabel_seed'] = args.relabel_seed
    f.attrs['keep_real_frac'] = args.keep_real_frac
    f.attrs['keep_real_seed'] = args.keep_real_seed
    f.attrs['parent_pool'] = os.path.abspath(args.pool_in)
    f.attrs['model_ckpt'] = os.path.abspath(ckpt_path)

  summary = summarize(new_query, pool, kept_real)
  summary.update({'generation': generation, 'method': args.method, 'temperature': args.temperature,
                  'keep_real_frac': args.keep_real_frac, 'pool_in': args.pool_in,
                  'pool_out': args.pool_out, 'model_ckpt': ckpt_path,
                  'changed_vs_parent': float((new_query != pool['labels'][:, -1]).mean())})
  summary_path = os.path.splitext(args.pool_out)[0] + '_summary.json'
  with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

  print('Pool out: {} (generation {})'.format(args.pool_out, generation))
  print('  query labels wrong       : {:.4%}'.format(summary['error_rate']))
  print('    - distractor\'s label   : {:.4%}'.format(summary['error_distractor_label']))
  print('    - label not in context : {:.4%}'.format(summary['error_out_of_context']))
  print('  changed vs parent pool   : {:.4%}'.format(summary['changed_vs_parent']))
  print('  kept true label          : {:.2%} of rows'.format(summary['kept_real_frac']))
  print('  query label freq         :', summary['query_label_freq'])
  print('  (true label freq         :', summary['true_label_freq'], ')')
  print('Summary:', summary_path)


if __name__ == '__main__':
  main()