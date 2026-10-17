'''
This file contains code for training models. It relies on main_utils.py
for argparse options. It constructs training and evaluation data iterators,
creates a model (supporting loading from checkpoint logic), and trains the
model. Evaluations are run throughout training and checkpoints are saved.

It also factors in any optogenetic clamps that may be used throughout training.
See artificial_optogenetics_guide.md and opto.py for more details.
'''

from typing import Any
from typing import Sequence
from typing import Generator
from typing import List
from typing import Tuple
from jax import Array
from nptyping import NDArray
from nptyping import Floating

import argparse
from functools import partial
from time import time
import os
import json
from datetime import datetime
import pdb

import numpy as np
import h5py as h5
import pickle as pkl

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

import samplers
import main_utils
import opto


ALL_TRAIN_METRICS = ['loss', 'grad_norm', 'grad_batch_stddev']


def smart_index(arr_maybe_none, i, default):
  if arr_maybe_none is None:
    return default
  elif isinstance(arr_maybe_none, str):
    return arr_maybe_none
  else:
    return arr_maybe_none[i]


def accuracy(pred_y: Array, y: Array) -> Array:
  predicted_class = jnp.argmax(pred_y, axis=-1)
  return predicted_class == y


def ce(pred_y: Array, y: Array) -> Array:
  pred_y = jax.nn.log_softmax(pred_y, axis=-1)
  num_classes = pred_y.shape[-1]
  onehot_y = jax.nn.one_hot(y, num_classes)
  return -jnp.sum(pred_y * onehot_y, axis=-1)


# Uncomment this jit if looking to microbatch
# @eqx.filter_jit
# Remember that this only yields the grad w.r.t. first positional arg!
@eqx.filter_value_and_grad
def compute_loss(
  model: eqx.Module,
  fwd_fn,
  weight_decay: float,
  x: Array, 
  y: Array, 
  keys: Array,
  c: Array = None,
) -> Array:
  out = jax.vmap(partial(fwd_fn, model=model))(x=x, y=y, key=keys)
  pred_y = out['out']
  if c is None:
    # Base version (unchanged): loss on the query label, the last item
    task_loss = ce(pred_y[:, -1, :], y[:, -1]).mean()
  else:
    # Extended version: items are [context, context, query, next]
    task_loss = extended_losses(out, y, c)['total'].mean()
  # Hacky, but prevents nan'ing gradients of 0 (which biases are initialized too)
  weight_norm = jnp.sum(jnp.stack(jax.tree_map(lambda x: jnp.linalg.norm(jnp.where(x != 0, x, 0)), jax.tree_leaves(eqx.filter(model, eqx.is_array)))))
  return task_loss + weight_decay * weight_norm


# Extended version. Sequences are 4 items: context, context, query, next.
# The model predicts the query's label (label output at item 2), the next
# symbol (symbol output read at the query's label token) and the next
# symbol's label (label output at item 3).
QUERY, NEXT = 2, 3


def extended_losses(out, y, c):
  '''Per-sequence losses of the three predictions. c: class indices [bs, 4].'''
  query = ce(out['out'][:, QUERY, :], y[:, QUERY])
  symbol = ce(out['out_symbol'][:, QUERY, :], c[:, NEXT])
  next_label = ce(out['out'][:, NEXT, :], y[:, NEXT])
  return {'query': query, 'symbol': symbol, 'next_label': next_label,
          'total': query + symbol + next_label}


# Comment this jit if looking to microbatch
@eqx.filter_jit
def train_step(
  model: eqx.Module,
  fwd_fn,
  optimizer: optax.GradientTransformation,
  opt_state: Array,
  microbs: int,
  weight_decay: float,
  x: Array,
  y: Array,
  key: Array,
  c: Array = None,
) -> Tuple[Array, eqx.Module, Array]:
  keys = jax.random.split(key, x.shape[0])

  losses = []
  grads = []
  lengths = []
  for i in range(0, x.shape[0], microbs):
    use_x = x[i:i+microbs]
    use_y = y[i:i+microbs]
    use_keys = keys[i:i+microbs]
    use_c = None if c is None else c[i:i+microbs]
    # For some reason using keywords for the argument on the next line breaks things
    # I think it has to do with how equinox implements the value_and_grad wrapper
    # Namely, it "priveleges" 'x' I think
    mini_loss, mini_grad = compute_loss(model, fwd_fn, weight_decay, use_x, use_y, use_keys, use_c)
    losses.append(mini_loss)
    grads.append(mini_grad)

  loss = jnp.mean(jnp.array(losses))
  avg_fn = lambda *args: jnp.mean(jnp.array(args),axis=0)
  norm_fn = lambda model_like: jnp.sqrt(jnp.sum(jnp.array([jnp.sum(arr) for arr in jax.tree_util.tree_flatten(model_like)[0]])))

  grad = jax.tree_map(avg_fn, *grads)
  grad_norm = norm_fn(jax.tree_map(lambda x: x**2, grad))
  grads_dev = [jax.tree_map(lambda x,y: (x-y)**2, g, grad) for g in grads]
  grad_batch_var = jax.tree_map(avg_fn, *grads_dev)
  grad_batch_stddev = norm_fn(grad_batch_var)

  update, opt_state = optimizer.update(grad, opt_state)
  model = eqx.apply_updates(model, update)
  # Make sure that ALL_TRAIN_METRICS is correspondingly updated
  # if more metrics are added to this function
  return {'loss': loss, 'grad_norm': grad_norm, 'grad_batch_stddev': grad_batch_stddev}, model, opt_state


@eqx.filter_jit
def eval_step(
  model: eqx.Module, 
  fwd_fn,
  x: Array, 
  y: Array, 
  key: Array
) -> Tuple[Array, Array]:

  keys = jax.random.split(key, x.shape[0])
  pred_y = jax.vmap(partial(fwd_fn, model=model))(x=x, y=y, key=keys)['out']

  in_context_mask = jnp.sum(jax.nn.one_hot(y[:, :-1], pred_y.shape[-1]), axis=1) > 0

  # We don't care about non-final sequence outputs from hereon out
  y = y[:, -1]
  pred_y = pred_y[:, -1, :]

  loss = ce(pred_y, y)
  accs = accuracy(pred_y, y)
  prob_y = jax.nn.softmax(pred_y)

  output_prob = jnp.einsum('bc,bc->b', 
                            prob_y, 
                            jax.nn.one_hot(y, pred_y.shape[-1]))

  # mask out the logits for items that don't appear in context
  # by setting them to a very negative number
  in_context_pred_y = in_context_mask*pred_y - (1-in_context_mask)*1e20
  fsl_accs = accuracy(in_context_pred_y, y)
  fsl_output_prob = jnp.einsum('bc,bc->b', 
                                jax.nn.softmax(in_context_pred_y), 
                                jax.nn.one_hot(y, in_context_pred_y.shape[-1]))

  out_context_pred_y = (1-in_context_mask)*pred_y - in_context_mask*1e20
  out_context_accs = accuracy(out_context_pred_y, y)
  out_context_prob = jnp.einsum('bc,bc->b', 
                                jax.nn.softmax(out_context_pred_y), 
                                jax.nn.one_hot(y, out_context_pred_y.shape[-1]))

  use_context_prob = jnp.sum(in_context_mask*prob_y, axis=-1)

  return {'loss': loss, 
          'acc': accs, 
          'in_context_acc': fsl_accs, 
          'out_context_acc': out_context_accs,
          'prob': output_prob,
          'in_context_prob': fsl_output_prob,
          'out_context_prob': out_context_prob,
          'use_context_prob': use_context_prob}


def label_metrics(pred_y, y, in_context_mask, prefix=''):
  """The base version's label metrics, for one item position."""
  prob_y = jax.nn.softmax(pred_y)
  in_context_pred_y = in_context_mask*pred_y - (1-in_context_mask)*1e20
  return {prefix + 'loss': ce(pred_y, y),
          prefix + 'acc': accuracy(pred_y, y),
          prefix + 'in_context_acc': accuracy(in_context_pred_y, y),
          prefix + 'prob': jnp.einsum('bc,bc->b', prob_y, jax.nn.one_hot(y, pred_y.shape[-1])),
          prefix + 'in_context_prob': jnp.einsum('bc,bc->b', jax.nn.softmax(in_context_pred_y),
                                                 jax.nn.one_hot(y, pred_y.shape[-1])),
          prefix + 'use_context_prob': jnp.sum(in_context_mask*prob_y, axis=-1)}


@eqx.filter_jit
def eval_step_extended(model, fwd_fn, x, y, key, c, num_symbol_classes):
  """Extended version's metrics. c: class indices [bs, 4].

  Query metrics keep the base names (loss, in_context_acc, ...), so base and
  extended runs are compared on the same quantity. next_label_* are the same
  metrics for the next symbol's label. symbol_* describe the next-symbol
  prediction; they are NaN where the true next symbol is outside the 50
  training classes (the held-out-class evaluator), as the symbol output
  cannot name those classes.
  """
  keys = jax.random.split(key, x.shape[0])
  out = jax.vmap(partial(fwd_fn, model=model))(x=x, y=y, key=keys)
  pred = out['out']
  n_labels = pred.shape[-1]
  # The two context labels: the only labels the model can copy
  in_context_mask = jnp.sum(jax.nn.one_hot(y[:, :QUERY], n_labels), axis=1) > 0

  metrics = label_metrics(pred[:, QUERY, :], y[:, QUERY], in_context_mask)
  metrics.update(label_metrics(pred[:, NEXT, :], y[:, NEXT], in_context_mask, prefix='next_label_'))

  sym = out['out_symbol'][:, QUERY, :]
  sym_prob = jax.nn.softmax(sym)
  target = c[:, NEXT]
  rows = jnp.arange(x.shape[0])
  valid = jnp.all(c < num_symbol_classes, axis=1)
  safe = jnp.where(c < num_symbol_classes, c, 0)
  query_class, other_class = safe[:, QUERY], jnp.where(safe[:, 0] == safe[:, QUERY], safe[:, 1], safe[:, 0])
  pred_class = jnp.argmax(sym, axis=-1)
  nan_if_invalid = lambda v: jnp.where(valid, v.astype(jnp.float32), jnp.nan)
  metrics.update({
    'symbol_loss': nan_if_invalid(ce(sym, jnp.where(valid, target, 0))),
    'symbol_acc': nan_if_invalid(pred_class == target),
    # Probability the model gives to repeating the query's own symbol: the
    # quantity that should drift across generations (real data: p_repeat)
    'symbol_repeat_prob': nan_if_invalid(sym_prob[rows, query_class]),
    'symbol_other_prob': nan_if_invalid(sym_prob[rows, other_class]),
    # Probability on the two context symbols together (the rest is on the
    # 48 symbols not in context)
    'symbol_use_context_prob': nan_if_invalid(sym_prob[rows, query_class] + sym_prob[rows, other_class]),
    # Share of the in-context probability that goes to repeating: the model's
    # estimate of p_repeat once out-of-context symbols are set aside
    'symbol_repeat_share': nan_if_invalid(sym_prob[rows, query_class] /
                                          (sym_prob[rows, query_class] + sym_prob[rows, other_class])),
    'symbol_pred_repeat': nan_if_invalid(pred_class == query_class),
    'symbol_pred_in_context': nan_if_invalid((pred_class == query_class) | (pred_class == other_class)),
  })
  losses = extended_losses(out, y, jnp.where(c < num_symbol_classes, c, 0))
  metrics['total_loss'] = nan_if_invalid(losses['total'])
  return metrics


def make_batched_fn(fn, batch_size):
  def batched_fn(model, x, y, key, c=None):
    total = x.shape[0]
    metrics = dict()
    seed = key
    for it in range(0, total, batch_size):
      seed, use = jax.random.split(seed)
      cap = min(it+batch_size, total)
      if c is None:
        out = fn(model=model, x=x[it:cap], y=y[it:cap], key=use)
      else:
        out = fn(model=model, x=x[it:cap], y=y[it:cap], key=use, c=c[it:cap])
      for m in out:
        metrics.setdefault(m, []).append(out[m])
    for m in metrics:
      metrics[m] = jnp.concatenate(metrics[m])
    return metrics
  return batched_fn


def evaluate(
  model: eqx.Module,
  fwd_fn,
  key: Array,
  eval_data,
  eval_batch_size,
  num_symbol_classes=None,
):
  seeds = jax.random.split(key, len(eval_data))
  retval = dict()
  if num_symbol_classes is None:
    eval_fn = make_batched_fn(partial(eval_step, fwd_fn=fwd_fn), eval_batch_size)
    for seed, k in zip(seeds, eval_data):
      retval[k] = eval_fn(model, eval_data[k]['examples'], eval_data[k]['labels'], key=seed)
  else:
    # Extended version: needs each item's class, for the next-symbol metrics
    eval_fn = make_batched_fn(partial(eval_step_extended, fwd_fn=fwd_fn,
                                      num_symbol_classes=num_symbol_classes), eval_batch_size)
    for seed, k in zip(seeds, eval_data):
      retval[k] = eval_fn(model, eval_data[k]['examples'], eval_data[k]['labels'], key=seed,
                          c=eval_data[k]['class_idxs'])
  return retval


POOL_KEYS = ['class_idxs', 'exemplar_inds', 'idx_types', 'labels']
# Extended version only: the real next item, kept unchanged across generations
EXTENDED_TRUE_KEYS = ['true_next_class', 'true_next_exemplar', 'true_next_label', 'true_next_is_repeat']


def build_or_load_pool(opts, train_data_sampler, run_folder):
  '''
  Returns the finite training pool as a dict of numpy arrays, or None.

  - If opts.pool_file is set, loads that pool (generation 1 onwards).
  - Else, if opts.pool_size is set, samples generation 0 from the original
    train sampler (so it has real labels) and saves it to
    <run_folder>/pool_gen0.h5.
  - Otherwise returns None, and training uses the original sampler.

  Pool arrays (one row per sequence, N = pool size, L+1 = context + query):
    class_idxs       [N, L+1]  original class index at each position
    exemplar_inds    [N, L+1]  exemplar index at each position
    idx_types        [N, L+1]  samplers.ItemType of each position
    labels           [N, L+1]  labels the model trains on; between
                               generations only labels[:, -1] (the query
                               label) changes
    true_query_label [N]       the correct query label, never changed
  Extended version (--extended): 4 items per sequence (context, context,
  query, next); the query is item 2, and the pool also keeps the real next
  item (EXTENDED_TRUE_KEYS), which relabelling never changes.
  We store indices rather than embeddings: examples are rebuilt with
  data[class_idxs, exemplar_inds], which keeps the file small.
  '''
  if opts.pool_file is not None:
    with h5.File(opts.pool_file, 'r') as f:
      pool = {k: np.array(f[k]) for k in POOL_KEYS + ['true_query_label'] + EXTENDED_TRUE_KEYS if k in f}
      print('Loaded pool from {} (generation {}, {} sequences)'.format(
        opts.pool_file, f.attrs.get('generation', '?'), len(pool['labels'])))
    return pool

  if opts.pool_size is None:
    return None

  print('Sampling a pool of {} sequences (pool_seed={})'.format(opts.pool_size, opts.pool_seed))
  # Separate key from train_seed, so the pool doesn't depend on training order
  pool_key = jax.random.PRNGKey(opts.pool_seed)
  chunks = {k: [] for k in POOL_KEYS}
  n_batches = -(-opts.pool_size // opts.train_bs)  # ceiling division
  for _ in range(n_batches):
    pool_key, use_key = jax.random.split(pool_key)
    batch = train_data_sampler(use_key)
    for k in POOL_KEYS:
      # np.asarray moves each batch off the GPU as we go
      chunks[k].append(np.asarray(batch[k]))
  pool = {k: np.concatenate(chunks[k], axis=0)[:opts.pool_size] for k in POOL_KEYS}
  # .copy() so this never changes when the labels are overwritten later
  if opts.extended:
    pool['true_query_label'] = pool['labels'][:, QUERY].copy()
    pool['true_next_class'] = pool['class_idxs'][:, NEXT].copy()
    pool['true_next_exemplar'] = pool['exemplar_inds'][:, NEXT].copy()
    pool['true_next_label'] = pool['labels'][:, NEXT].copy()
    pool['true_next_is_repeat'] = (pool['class_idxs'][:, NEXT] == pool['class_idxs'][:, QUERY])
  else:
    pool['true_query_label'] = pool['labels'][:, -1].copy()

  path = os.path.join(run_folder, 'pool_gen0.h5')
  with h5.File(path, 'w') as f:
    for k, v in pool.items():
      f.create_dataset(k, data=v)
    f.attrs['generation'] = 0
    f.attrs['source'] = 'real'
    f.attrs['pool_seed'] = opts.pool_seed
    f.attrs['extended'] = bool(opts.extended)
    if opts.extended:
      f.attrs['p_repeat'] = opts.p_repeat
  print('Saved pool to', path)
  return pool


@partial(jax.jit, static_argnames='batch_size')
def get_pool_batch(data, pool_dev, perm, offset, batch_size):
  '''
  Builds one training batch from the pool, in the same format as
  train_data_sampler: {'examples': [bs, L+1, dim], 'labels': [bs, L+1]}.

  Takes the `batch_size` rows of the pool at positions offset..offset+bs of
  the shuffled order `perm`, and looks their embeddings up in `data`.
  offset is a traced argument, so this compiles once rather than per step.
  '''
  rows = jax.lax.dynamic_slice(perm, (offset,), (batch_size,))
  return {'examples': data[pool_dev['class_idxs'][rows], pool_dev['exemplar_inds'][rows]],
          'labels': pool_dev['labels'][rows],
          'class_idxs': pool_dev['class_idxs'][rows]}


def run_with_opts(opts):

  ### Process input opts ###

  if opts.config_from_file is not None:
    opts = main_utils.get_opts_from_json_file(opts.config_from_file)
    opts.config_from_file = None

  if opts.train_microbs is None:
    opts.train_microbs = opts.train_bs
  else:
    if opts.train_bs % opts.train_microbs > 0:
      print("WARNING: using microbatching. Make sure jitting is configured properly -- see comments in main.py")
      print('Setting train_bs to a multiple of train_microbs')
      new_bs = opts.train_microbs*((opts.train_bs//opts.train_microbs) + 1)
      print("Updating train_bs from {} to {}".format(opts.train_bs, new_bs))
      opts.train_bs = new_bs

  main_utils.check_opts(opts)

  if opts.extended:
    # The extended sampler assumes 2 context items, one of them the query's
    if opts.train_context_len != 2 or list(opts.pt_burstiness) != [1]:
      raise ValueError('--extended needs --train_context_len 2 and --pt_burstiness 1')
    if opts.load_eval_data is not None:
      raise ValueError('--extended does not support --load_eval_data (no class indices in the file)')

  if opts.disable_jit:
    print("Disabling jit")
    jax.config.update('jax_disable_jit', True)

  if opts.eval_sched is not None:
    opts.eval_every = None
    opts.eval_sched_file = None
  elif opts.eval_every is not None:
    opts.eval_sched = np.arange(0, opts.train_iters, opts.eval_every)
    opts.eval_every = None
  elif opts.eval_sched_file is not None:
    with open(opts.eval_sched_file, 'rb') as f:
      opts.eval_sched = pkl.load(f)
    opts.eval_sched_file = None
  # Sort and canonicalize for json
  opts.eval_sched = [int(x) for x in sorted(opts.eval_sched)]

  if opts.ckpt_sched is not None:
    opts.ckpt_every = None
    opts.ckpt_sched_file = None
  elif opts.ckpt_every is not None:
    opts.ckpt_sched = np.arange(0, opts.train_iters, opts.ckpt_every)
    opts.ckpt_every = None
  elif opts.ckpt_sched_file is not None:
    with open(opts.ckpt_sched_file, 'rb') as f:
      opts.ckpt_sched = pkl.load(f) 
    opts.ckpt_sched_file = None
  else:
    # If all ckpt opts are None, avoid checkpointing
    opts.ckpt_sched = np.array([opts.train_iters*2]).astype(int)
  # Sort and canonicalize for json
  opts.ckpt_sched = [int(x) for x in sorted(opts.ckpt_sched)]

  # Model output classes default logic
  if opts.model_output_classes is None:
    if opts.fs_relabel > 0:
      opts.model_output_classes = opts.fs_relabel
    else:
      opts.model_output_classes = opts.class_split[0]
    print("Defaulting model output classes to", opts.model_output_classes)

  # It's important to set all the defaults before saving the opts
  # For example, when loading a checkpoint, we want things like model output classes
  # to be filled in

  if opts.raw_name:
    run_name = opts.run
  else:
    run_name = '_'.join([datetime.now().strftime("%Y%m%d%H%M%S"), opts.run])

  run_folder = '/'.join([opts.base_folder, run_name])
  os.makedirs(run_folder, exist_ok=True)
  with open('/'.join([run_folder, 'config.json']), 'w') as f:
    json.dump(vars(opts), f, indent='\t')

  if opts.use_wandb:
    import wandb
    wandb.login()
    wandb.init(
      project="icl-transience", config=opts, name=opts.run, dir=opts.base_folder)
  
  opts.train_seed = jax.random.PRNGKey(opts.train_seed)
  opts.eval_seed = jax.random.PRNGKey(opts.eval_seed)

  if opts.ckpt_sched is not None:
    ckpt_folder = '/'.join([run_folder, 'checkpoints'])
    os.makedirs(ckpt_folder, exist_ok=True)
  else:
    opts.ckpt_sched = [2*opts.train_iters] # will never checkpoint

  ### Make train data sampler ###

  data = main_utils.get_data_from_opts(opts)

  splits = main_utils.get_splits_from_opts(opts, data.shape)

  zipf_ranks = jnp.arange(1, opts.class_split[0]+1, dtype=jnp.float32)
  zipf_probs = 1 / zipf_ranks**opts.zipf_alpha
  train_class_distr = zipf_probs / jnp.sum(zipf_probs)

  
  assert len(opts.pt_burstiness) == len(opts.mixing_coeffs)

  train_samplers = []
  assert len(opts.mixing_coeffs) > 0, "Must have at least 1 train sampler"
  for i in range(len(opts.mixing_coeffs)):
    train_samplers.append(partial(samplers.get_constant_burst_seq_idxs,
                                  classes=splits['class']['train'],
                                  class_distr=train_class_distr,
                                  num_seqs=opts.train_bs,
                                  context_len=opts.train_context_len, 
                                  burstiness=opts.pt_burstiness[i],
                                  distractor=smart_index(opts.pt_distract, i, 1),
                                  no_support=smart_index(opts.pt_no_support, i, 0),
                                  unique_rest=smart_index(opts.pt_unique_rest, i, 0)))
  train_class_sampler = partial(samplers.get_mixed_seq_idxs, 
                            mix_probabilities=jnp.array(opts.mixing_coeffs), 
                            mix_substrate_fns=train_samplers)

  train_exemplar_sampler = partial(
    samplers.get_exemplar_inds,
    allowed_inds=splits['exemplar']['train'],
    match_query_and_distractors=opts.match_query_and_distractors)
  # the below will now take in a key, and output a batch, all jitted up!
  train_data_sampler = jax.jit(partial(samplers.make_data_sampler(
    train_class_sampler,
    train_exemplar_sampler,
    fs_relabel=splits['relabeling']['train'],
    noise_scale=opts.noise_scale_train,
    assign_query_label_random=opts.assign_query_label_random,
    extended=opts.extended,
    p_repeat=opts.p_repeat),
    data=data))

  # Finite training pool (None unless --pool_size or --pool_file is given)
  pool = build_or_load_pool(opts, train_data_sampler, run_folder)
  if pool is not None:
    # A base pool has 3 items per sequence, an extended pool 4: never mix them
    n_items = pool['labels'].shape[1]
    if opts.extended and n_items != 4:
      raise ValueError('--extended needs an extended pool (4 items per sequence), but {} has {}. '
                       'Is GEN0_RUN/GEN0_POOL pointing at a base run?'.format(opts.pool_file, n_items))
    if not opts.extended and n_items == 4:
      raise ValueError('{} is an extended pool (4 items per sequence): pass --extended'.format(opts.pool_file))

  train_data_seed, train_model_seed = jax.random.split(opts.train_seed, 2)
  eval_data_seed, eval_model_seed = jax.random.split(opts.eval_seed, 2)

  ### Make eval data ###

  # We first make a sampler for each evaluator, 
  # then take one sample from it to use as our eval data
  # This allows us to re-use the machinery in samplers.py

  eval_data = {}
  if opts.load_eval_data is not None:
    for eval_file in opts.load_eval_data:
      # an evaluator is a mini dataset of sequences and labels
      print("loading some evaluators from file")
      eval_h5 = h5.File(eval_file, 'r')
      print("Found these eval sets in file:", eval_h5.keys())
      for k in eval_h5:
        if k in eval_data:
          raise ValueError("Found duplicate eval key. Behavior in this case is not defined")
        eval_data[k] = {'examples': jnp.array(eval_h5['/'.join([k, 'examples'])]),
                          'labels': jnp.array(eval_h5['/'.join([k, 'labels'])])}
      print("Loaded in file:", eval_h5.keys())
      eval_h5.close()

  eval_data_seeds = jax.random.split(eval_data_seed, len(opts.pe_names))
  for i, name in enumerate(opts.pe_names):
    if name in eval_data:
      print("ERROR: Found overlapping name of new evaluator with old evaluator from file! Please rename new evaluator or stop using file.")
      pdb.set_trace()

    assert name not in ['train', 'train_iter', 'eval_iter']

    classes = splits['class'][opts.pe_classes[i]]
    eval_class_sampler = partial(samplers.get_constant_burst_seq_idxs,
                                  classes=classes,
                                  class_distr=jnp.ones(len(classes))/len(classes),
                                  num_seqs=opts.eval_iters,
                                  context_len=smart_index(opts.pe_context_len, i, opts.train_context_len),
                                  burstiness=opts.pe_burstiness[i],
                                  distractor=smart_index(opts.pe_distract, i, 1),
                                  no_support=smart_index(opts.pe_no_support, i, 0),
                                  unique_rest=smart_index(opts.pe_unique_rest, i, 0))
    allowed_inds = splits['exemplar'][opts.pe_exemplars[i]]
    eval_exemplar_sampler = partial(
      samplers.get_exemplar_inds,
      allowed_inds=allowed_inds,
      match_query_and_distractors=opts.match_query_and_distractors)
    eval_data_sampler = samplers.make_data_sampler(
      eval_class_sampler,
      eval_exemplar_sampler,
      fs_relabel=splits['relabeling'][smart_index(opts.pe_fs_relabel_scheme, i, 'None')],
      noise_scale=smart_index(opts.pe_noise_scale, i, opts.noise_scale_train),
      assign_query_label_random=smart_index(opts.pe_assign_query_label_random, i, 0),
      extended=opts.extended,
      p_repeat=opts.p_repeat)
    eval_data[name] = eval_data_sampler(eval_data_seeds[i], data)

  if opts.save_eval_data is not None:
    fname = '/'.join([run_folder, opts.save_eval_data])
    print("Writing eval data to file:", fname)
    with h5.File(fname, 'w') as f:
      for k in eval_data:
        for s in eval_data[k]:
          f.create_dataset('/'.join([k,s]), data=eval_data[k][s])
    print("Done writing")


  ### Setup model and optimizer ###
  # Includes logic for loading from checkpoints

  fwd_fn_to_use = opto.make_fn_from_opts(opts)

  if opts.load_from_ckpt is not None:
    assert opts.load_from_ckpt_cfg is not None, "Must specify config for loading model checkpoint"
    print("Resetting config to checkpointed config")
    load_opts = main_utils.get_opts_from_json_file(opts.load_from_ckpt_cfg)
    model = main_utils.get_model_from_opts(load_opts)
  else:
    model = main_utils.get_model_from_opts(opts, input_shape=(data.shape[-1],))

  optimizer = main_utils.get_optimizer_from_opts(opts)
  opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

  start_iter = 0
  if opts.load_from_ckpt is not None:
    ckpt_fmt = {'iter': -1, 
                'seeds': {'eval_model_seed': eval_model_seed,
                          'train_data_seed': train_data_seed,
                          'train_model_seed': train_model_seed}, 
                'opt_state': opt_state,
                'model': model}
    ckpt = eqx.tree_deserialise_leaves(opts.load_from_ckpt, ckpt_fmt)

    start_iter = ckpt['iter']
    if not opts.no_load_seeds:
      eval_model_seed = ckpt['seeds']['eval_model_seed']
      train_data_seed = ckpt['seeds']['train_data_seed']
      train_model_seed = ckpt['seeds']['train_model_seed']
    if not opts.no_load_opt_state:
      opt_state = ckpt['opt_state']
    model = ckpt['model']

  ### Setup log.h5 ###

  eval_ind = np.searchsorted(opts.eval_sched, start_iter)
  ckpt_ind = np.searchsorted(opts.ckpt_sched, start_iter)

  train_metrics = {m: [] for m in ALL_TRAIN_METRICS}
  # We'll always want train iters
  train_metrics['iter'] = []

  results = h5.File('/'.join([run_folder, 'log.h5']), 'a')
  for m in train_metrics:
    if 'train_{}'.format(m) not in results:
      results.create_dataset('train_{}'.format(m), shape=(0,), maxshape=(None,), dtype=float)
  if 'eval_iter' not in results:
    results.create_dataset('eval_iter', shape=(0,), maxshape=(None,), dtype=int)
  results.close()

  ### Setup training from the pool (if any) ###
  # Each pass over the pool uses a fresh shuffle, derived from train_seed and
  # the pass number. Because it only depends on i, resuming from a checkpoint
  # with --load_from_ckpt continues with exactly the right batches.
  if pool is not None:
    pool_n = len(pool['labels'])
    if pool_n % opts.train_bs != 0:
      raise ValueError('Pool size ({}) must be a multiple of train_bs ({})'.format(pool_n, opts.train_bs))
    if opts.noise_scale_train > 0:
      raise ValueError('noise_scale_train > 0 is not supported when training from a pool')
    # Move the index arrays to the device once (~50MB for 1M rows)
    pool_dev = {k: jnp.asarray(pool[k]) for k in ['class_idxs', 'exemplar_inds', 'labels']}
    pool_shuffle_key = jax.random.fold_in(opts.train_seed, 1)
    pool_pass, pool_perm = None, None
    print('Training from pool: {} rows, {:.2f} passes'.format(
      pool_n, (opts.train_iters - start_iter) / pool_n))

  # Extended version: number of classes the symbol output can name
  num_symbol_classes = opts.class_split[0] if opts.extended else None

  ### MAIN TRAIN LOOP ###
  # Runs eval and checkpoints according to schedule
  # Note i = iterations = # sequences seen

  for i in range(start_iter, opts.train_iters, opts.train_bs):
    if eval_ind < len(opts.eval_sched) and i >= opts.eval_sched[eval_ind]:
      if not opts.suppress_output:
        print('-'*10 + str(i))
      eval_model_seed, current_eval_seed = jax.random.split(eval_model_seed, 2)
      out = evaluate(model=model, 
                      fwd_fn=fwd_fn_to_use, 
                      key=current_eval_seed, 
                      eval_data=eval_data, 
                      eval_batch_size=opts.eval_bs,
                      num_symbol_classes=num_symbol_classes)
      results = h5.File('/'.join([run_folder, 'log.h5']), 'a')
      for k in out:
        for m in out[k]:
          if not opts.suppress_output:
            print(k, m, jnp.mean(out[k][m]))
          if opts.use_wandb:
            wandb.log({'-'.join([k, m]): jnp.mean(out[k][m]), 'iteration': i}, step=i//opts.train_bs)
          addr = '/'.join([k,m])
          if addr in results:
            results[addr].resize(results[addr].shape[0] + 1, axis=0)
            results[addr][-1] = out[k][m]
          else:
            results.create_dataset(addr, data=jnp.broadcast_to(out[k][m], (1, len(out[k][m]))), maxshape=(None, len(out[k][m])))
      results['eval_iter'].resize(results['eval_iter'].shape[0]+1, axis=0)
      results['eval_iter'][-1] = i
      if len(train_metrics['iter']) > 0:
        if not opts.suppress_output:
          print('-')
        if opts.use_wandb:
          wandb.log({**{m: train_metrics[m][-1] for m in ALL_TRAIN_METRICS}, 'iteration': i}, step=i//opts.train_bs)
        for m in train_metrics:
          results['train_{}'.format(m)].resize(results['train_{}'.format(m)].shape[0]+len(train_metrics[m]), axis=0)
          results['train_{}'.format(m)][-len(train_metrics[m]):] = train_metrics[m]
          if not opts.suppress_output:
            print(m, train_metrics[m][-1])
          train_metrics[m] = []
      results.close()
      eval_ind += 1

    if ckpt_ind < len(opts.ckpt_sched) and i >= opts.ckpt_sched[ckpt_ind]:
      if not opts.suppress_output:
        print("Checkpointing...", i)
      ckpt = {'iter': i,
              'seeds': {'eval_model_seed': eval_model_seed,
                        'train_data_seed': train_data_seed,
                        'train_model_seed': train_model_seed}, 
              'opt_state': opt_state,
              'model': model}
      num = '{}'.format(i).zfill(11)
      eqx.tree_serialise_leaves('/'.join([ckpt_folder, num+".eqx"]), ckpt)
      if len(train_metrics['iter']) > 0:
        if not opts.suppress_output:
          print(train_metrics['iter'][-1], train_metrics['loss'][-1])
      ckpt_ind += 1

    # Train step -- the train_data_seed is split and passed along (a la functional
    # programming)
    train_data_seed, current_data_seed = jax.random.split(train_data_seed)
    # batch is a dict with keys 'examples' and 'labels'
    # 'examples' is [batch_size, train_context_len + 1, embedding_dim]
    # 'labels' is [batch_size, train_context_len + 1]
    # (the +1 is for the target/query)
    if pool is None:
      batch = train_data_sampler(current_data_seed)
    else:
      # Which pass over the pool we're on, and where in it
      this_pass, offset = divmod(i, pool_n)
      if this_pass != pool_pass:
        # Shuffle once per pass (not per step: permuting 1M rows is slow)
        pool_perm = jax.random.permutation(jax.random.fold_in(pool_shuffle_key, this_pass), pool_n)
        pool_pass = this_pass
      batch = get_pool_batch(data, pool_dev, pool_perm, offset, batch_size=opts.train_bs)

    train_model_seed, current_model_seed = jax.random.split(train_model_seed)
    metrics, model, opt_state = train_step(model=model, 
                                            fwd_fn=fwd_fn_to_use,
                                            optimizer=optimizer, 
                                            opt_state=opt_state, 
                                            microbs=opts.train_microbs, 
                                            weight_decay=opts.weight_decay,
                                            x=batch['examples'], 
                                            y=batch['labels'], 
                                            key=current_model_seed,
                                            c=batch['class_idxs'] if opts.extended else None)
    train_metrics['iter'].append(i+opts.train_bs)
    for m in metrics:
      train_metrics[m].append(metrics[m])
    if opts.use_wandb:
      if i % 1000*opts.train_bs == 0:
        # Ideally we'd log learning rate, but hard to access with jax
        wandb.log({**metrics, 'iteration': i}, step=1+i//opts.train_bs)

  print("End of training")
  eval_model_seed, current_eval_seed = jax.random.split(eval_model_seed, 2)
  out = evaluate(model=model, 
                  fwd_fn=fwd_fn_to_use,
                  key=current_eval_seed, 
                  eval_data=eval_data, 
                  eval_batch_size=opts.eval_bs,
                  num_symbol_classes=num_symbol_classes)
  results = h5.File('/'.join([run_folder, 'log.h5']), 'a')
  if len(train_metrics['iter']) > 0:
    print('-')
    for m in train_metrics:
      results['train_{}'.format(m)].resize(results['train_{}'.format(m)].shape[0]+len(train_metrics[m]), axis=0)
      results['train_{}'.format(m)][-len(train_metrics[m]):] = train_metrics[m]
      print(m, train_metrics[m][-1])
  for k in out:
    for m in out[k]:
      print(k, m, jnp.mean(out[k][m]))
      addr = '/'.join([k,m])
      if addr in results:
        results[addr].resize(results[addr].shape[0] + 1, axis=0)
        results[addr][-1] = out[k][m]
      else:
        results.create_dataset(addr, data=jnp.broadcast_to(out[k][m], (1, len(out[k][m]))), maxshape=(None,len(out[k][m])))
  results['eval_iter'].resize(results['eval_iter'].shape[0]+1, axis=0)
  results['eval_iter'][-1] = i + opts.train_bs
  results.close()

  if opts.ckpt_sched[-1] <= opts.train_iters:
    print("Checkpointing...", i + opts.train_bs)
    ckpt = {'iter': i + opts.train_bs,
            'seeds': {'eval_model_seed': eval_model_seed,
                      'train_data_seed': train_data_seed,
                      'train_model_seed': train_model_seed}, 
            'opt_state': opt_state,
            'model': model}
    num = '{}'.format(i + opts.train_bs).zfill(11)
    eqx.tree_serialise_leaves('/'.join([ckpt_folder, num+".eqx"]), ckpt)


if __name__ == "__main__":
  parser = main_utils.create_parser()
  opto.add_args_to_parser(parser)
  opts = parser.parse_args()
  start = time()
  run_with_opts(opts)
  end = time()
  print("total training time (min): ", (end-start) / 60.)