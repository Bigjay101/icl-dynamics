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

Extended version (pools with 4 items: context, context, query, next; the
model has a symbol output): three tokens are generated one after another,
each fed back in before the next is predicted, as the model would generate
them: (1) the query's label, (2) the next symbol, (3) the next symbol's label.
The context and query items are never changed. --keep_real_frac rows keep all
three true tokens. A generated next symbol that is not in context gets the
training exemplar of its class (index 0, the only one).

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
# Extended version only
EXTENDED_TRUE_KEYS = ['true_next_class', 'true_next_exemplar', 'true_next_label', 'true_next_is_repeat']
QUERY, NEXT = 2, 3


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


@eqx.filter_jit
def extended_outputs(model, fwd_fn, data, class_idxs, exemplar_inds, labels):
  '''Extended version: the three predictions for a batch, from one forward pass.
  The model is causal, so each prediction only depends on the tokens before it.'''
  x = data[class_idxs, exemplar_inds]
  keys = jax.random.split(jax.random.PRNGKey(0), x.shape[0])  # unused: no dropout
  out = jax.vmap(partial(fwd_fn, model=model))(x=x, y=labels, key=keys)
  return {'query': out['out'][:, QUERY, :],          # read at the query image
          'symbol': out['out_symbol'][:, QUERY, :],   # read at the query's label token
          'next_label': out['out'][:, NEXT, :]}       # read at the next symbol's image


def choose(logits, method, temperature, key):
  if method == 'argmax':
    return jnp.argmax(logits, axis=-1)
  return jax.random.categorical(key, logits / temperature, axis=-1)


@eqx.filter_jit
def generate_batch(model, fwd_fn, data, class_idxs, exemplar_inds, labels, key, method, temperature):
  '''Extended version: generate the query label, the next symbol and the next
  symbol's label in turn, each fed back in before the next is predicted.'''
  k_query, k_symbol, k_label = jax.random.split(key, 3)
  # 1. Query label (depends only on the context and the query image)
  out1 = extended_outputs(model, fwd_fn, data, class_idxs, exemplar_inds, labels)
  query = choose(out1['query'], method, temperature, k_query).astype(labels.dtype)
  labels = labels.at[:, QUERY].set(query)
  # 2. Next symbol, given the generated query label
  out2 = extended_outputs(model, fwd_fn, data, class_idxs, exemplar_inds, labels)
  symbol = choose(out2['symbol'], method, temperature, k_symbol).astype(class_idxs.dtype)
  # Its exemplar: the context item's if the symbol is in context, else exemplar 0
  ex = jnp.where(symbol == class_idxs[:, 0], exemplar_inds[:, 0],
                 jnp.where(symbol == class_idxs[:, 1], exemplar_inds[:, 1], 0)).astype(exemplar_inds.dtype)
  class_idxs = class_idxs.at[:, NEXT].set(symbol)
  exemplar_inds = exemplar_inds.at[:, NEXT].set(ex)
  # 3. The next symbol's label, given the generated symbol
  out3 = extended_outputs(model, fwd_fn, data, class_idxs, exemplar_inds, labels)
  next_label = choose(out3['next_label'], method, temperature, k_label).astype(labels.dtype)
  # The distribution the next symbol was chosen from (the model's own, for argmax)
  sym_prob = jax.nn.softmax(out2['symbol'] / (temperature if method == 'sample' else 1.0))
  rows = jnp.arange(class_idxs.shape[0])
  other = jnp.where(class_idxs[:, 0] == class_idxs[:, QUERY], class_idxs[:, 1], class_idxs[:, 0])
  return {'query': query, 'symbol': symbol, 'exemplar': ex, 'next_label': next_label,
          'query_logits': out1['query'], 'next_label_logits': out3['next_label'],
          'repeat_prob': sym_prob[rows, class_idxs[:, QUERY]],
          'other_prob': sym_prob[rows, other]}


def summarize_extended(new, pool, kept_real):
  '''Statistics of a generated extended pool against the real next items.'''
  c, l = pool['class_idxs'], pool['labels']
  q_true = pool['true_query_label']
  q_wrong = new['query'] != q_true
  ctx_labels = l[:, :QUERY]
  distractor_label = np.where(pool['idx_types'][:, :QUERY] == ItemType.DISTRACTOR, ctx_labels, -1).max(axis=1)
  symbol = new['symbol']
  repeat = symbol == c[:, QUERY]
  other_class = np.where(c[:, 0] == c[:, QUERY], c[:, 1], c[:, 0])
  other = symbol == other_class
  in_ctx = repeat | other
  # Label that belongs to the chosen symbol in context (-1 if not in context)
  pos = np.where(symbol == c[:, 0], 0, 1)
  symbol_label = np.where(in_ctx, ctx_labels[np.arange(len(c)), pos], -1)
  counts = np.bincount(symbol, minlength=int(max(symbol.max(), c.max())) + 1)
  freq = counts / counts.sum()
  entropy = float(-(freq[freq > 0] * np.log(freq[freq > 0])).sum())
  true_counts = np.bincount(pool['true_next_class'], minlength=len(counts))
  true_freq = true_counts / true_counts.sum()
  true_entropy = float(-(true_freq[true_freq > 0] * np.log(true_freq[true_freq > 0])).sum())
  return {
    'n': int(len(c)),
    'kept_real_frac': float(kept_real.mean()),
    # 1. Query label (same meaning as the base version)
    'error_rate': float(q_wrong.mean()),
    'error_distractor_label': float((q_wrong & (new['query'] == distractor_label)).mean()),
    'error_out_of_context': float((q_wrong & ~(ctx_labels == new['query'][:, None]).any(axis=1)).mean()),
    # 2. Next symbol
    'next_repeat_share': float(repeat.mean()),
    'next_other_share': float(other.mean()),
    'next_out_of_context': float((~in_ctx).mean()),
    # Repeat share among in-context symbols: the quantity expected to drift
    'next_repeat_share_in_context': float(repeat.sum() / max(in_ctx.sum(), 1)),
    'true_next_repeat_share': float(pool['true_next_is_repeat'].mean()),
    'next_symbol_entropy': entropy,
    'true_next_symbol_entropy': true_entropy,
    'next_symbol_top5': [[int(i), round(float(freq[i]), 5)] for i in np.argsort(-freq)[:5]],
    'mean_repeat_prob': float(new['repeat_prob'].mean()),
    'mean_other_prob': float(new['other_prob'].mean()),
    # 3. Next symbol's label: does it match the chosen symbol's label in context?
    'next_label_matches_symbol': float((new['next_label'] == symbol_label)[in_ctx].mean()) if in_ctx.any() else 0.0,
    # Chains: a wrong query label, then what happens to the next symbol
    'next_out_of_context_after_wrong_query': float((~in_ctx)[q_wrong].mean()) if q_wrong.any() else 0.0,
  }


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


def relabel_extended(args, model, fwd_fn, data, ckpt_path):
  '''Extended version of the relabelling in main(): three generated tokens.'''
  with h5.File(args.pool_in, 'r') as f:
    missing = [k for k in EXTENDED_TRUE_KEYS if k not in f]
    if missing:
      raise ValueError('{} is not an extended pool (missing {})'.format(args.pool_in, missing))
    pool = {k: np.array(f[k]) for k in COPIED_KEYS + EXTENDED_TRUE_KEYS + ['labels']}
    parent_gen = int(f.attrs.get('generation', -1))
    p_repeat = f.attrs.get('p_repeat', None)
  n = len(pool['labels'])
  if pool['labels'].shape[1] != 4:
    raise ValueError('Extended pools have 4 items per sequence, got {}'.format(pool['labels'].shape[1]))
  print('Pool in: {} (generation {}, {} sequences, extended)'.format(args.pool_in, parent_gen, n))

  key = jax.random.PRNGKey(args.relabel_seed)
  parts = {}
  for b, start in enumerate(range(0, n, args.batch_size)):
    sl = slice(start, start + args.batch_size)
    out = generate_batch(model, fwd_fn, data,
                         jnp.asarray(pool['class_idxs'][sl]), jnp.asarray(pool['exemplar_inds'][sl]),
                         jnp.asarray(pool['labels'][sl]), jax.random.fold_in(key, b),
                         args.method, args.temperature)
    for k, v in out.items():
      parts.setdefault(k, []).append(np.asarray(v))
  new = {k: np.concatenate(v, axis=0) for k, v in parts.items()}

  # Fixed subset of rows that keep all three true tokens (same rows every generation)
  kept_real = np.random.default_rng(args.keep_real_seed).random(n) < args.keep_real_frac
  new['query'] = np.where(kept_real, pool['true_query_label'], new['query'])
  new['symbol'] = np.where(kept_real, pool['true_next_class'], new['symbol'])
  new['exemplar'] = np.where(kept_real, pool['true_next_exemplar'], new['exemplar'])
  new['next_label'] = np.where(kept_real, pool['true_next_label'], new['next_label'])

  class_idxs, exemplar_inds, labels = pool['class_idxs'].copy(), pool['exemplar_inds'].copy(), pool['labels'].copy()
  labels[:, QUERY] = new['query']
  class_idxs[:, NEXT] = new['symbol']
  exemplar_inds[:, NEXT] = new['exemplar']
  labels[:, NEXT] = new['next_label']

  generation = parent_gen + 1
  os.makedirs(os.path.dirname(os.path.abspath(args.pool_out)), exist_ok=True)
  with h5.File(args.pool_out, 'w') as f:
    f.create_dataset('class_idxs', data=class_idxs)
    f.create_dataset('exemplar_inds', data=exemplar_inds)
    f.create_dataset('labels', data=labels)
    for k in ['idx_types', 'true_query_label'] + EXTENDED_TRUE_KEYS:
      f.create_dataset(k, data=pool[k])
    f.create_dataset('logits', data=new['query_logits'].astype(np.float32))
    f.create_dataset('next_label_logits', data=new['next_label_logits'].astype(np.float32))
    # Instead of all 50 symbol logits (200 MB per pool), the two that matter
    f.create_dataset('symbol_repeat_prob', data=new['repeat_prob'].astype(np.float32))
    f.create_dataset('symbol_other_prob', data=new['other_prob'].astype(np.float32))
    f.create_dataset('kept_real', data=kept_real)
    f.attrs['generation'] = generation
    f.attrs['source'] = 'synthetic'
    f.attrs['extended'] = True
    if p_repeat is not None:
      f.attrs['p_repeat'] = p_repeat
    f.attrs['method'] = args.method
    f.attrs['temperature'] = args.temperature
    f.attrs['relabel_seed'] = args.relabel_seed
    f.attrs['keep_real_frac'] = args.keep_real_frac
    f.attrs['keep_real_seed'] = args.keep_real_seed
    f.attrs['parent_pool'] = os.path.abspath(args.pool_in)
    f.attrs['model_ckpt'] = os.path.abspath(ckpt_path)

  out_pool = {'class_idxs': pool['class_idxs'], 'labels': pool['labels'], 'idx_types': pool['idx_types'],
              'true_query_label': pool['true_query_label'], 'true_next_class': pool['true_next_class'],
              'true_next_is_repeat': pool['true_next_is_repeat']}
  summary = summarize_extended(new, out_pool, kept_real)
  summary.update({
    'generation': generation, 'extended': True, 'method': args.method, 'temperature': args.temperature,
    'keep_real_frac': args.keep_real_frac, 'pool_in': args.pool_in, 'pool_out': args.pool_out,
    'model_ckpt': ckpt_path,
    'changed_vs_parent_query': float((labels[:, QUERY] != pool['labels'][:, QUERY]).mean()),
    'changed_vs_parent_next_symbol': float((class_idxs[:, NEXT] != pool['class_idxs'][:, NEXT]).mean()),
    'changed_vs_parent_next_label': float((labels[:, NEXT] != pool['labels'][:, NEXT]).mean()),
  })
  summary_path = os.path.splitext(args.pool_out)[0] + '_summary.json'
  with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

  print('Pool out: {} (generation {}, extended)'.format(args.pool_out, generation))
  print('  query labels wrong            : {:.4%}'.format(summary['error_rate']))
  print('  next symbol repeats the query : {:.4%}  (real data: {:.4%})'.format(
    summary['next_repeat_share'], summary['true_next_repeat_share']))
  print('  next symbol is the other one  : {:.4%}'.format(summary['next_other_share']))
  print('  next symbol not in context    : {:.4%}'.format(summary['next_out_of_context']))
  print('  repeat share among in-context : {:.4f}'.format(summary['next_repeat_share_in_context']))
  print('  next label matches its symbol : {:.4%}'.format(summary['next_label_matches_symbol']))
  print('  kept true tokens              : {:.2%} of rows'.format(summary['kept_real_frac']))
  print('Summary:', summary_path)


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

  # Extended version: the model has a symbol output (trained with --extended)
  if getattr(model, 'symbol_unembed', None) is not None:
    relabel_extended(args, model, fwd_fn, data, ckpt_path)
    return

  with h5.File(args.pool_in, 'r') as f:
    if 'true_next_class' in f:
      raise ValueError('{} is an extended pool, but the model was not trained with --extended'.format(args.pool_in))
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
