"""Everything that touches JAX / the repo's model code.

* loading a run follows relabel.load_model (the same code that builds every pool),
* the probe is the run's own `fsl_val_rl` evaluator, rebuilt exactly as main.py
  builds it (same eval_seed -> the same 1000 sequences that log.h5 reports on),
* attention patterns come from SequenceClassifier.call_with_all_aux,
* head ablation zeroes a head's values through the cache / cache_mask mechanism,
  which is exactly what --opto_ablate_heads does in opto.py.

Run every script from the repo root (config.json stores the data file path relative to it).
"""
from __future__ import annotations

import os
import re
import sys
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import main_utils
import opto
import samplers
from relabel import last_checkpoint

QUERY, NEXT = 2, 3


def smart_index(arr_maybe_none, i, default):          # same as main.smart_index
  if arr_maybe_none is None:
    return default
  if isinstance(arr_maybe_none, str):
    return arr_maybe_none
  return arr_maybe_none[i]


def load_opts(run_folder):
  argv, sys.argv = sys.argv, sys.argv[:1]               # get_opts_from_json_file parses sys.argv
  try:
    return main_utils.get_opts_from_json_file(os.path.join(run_folder, "config.json"))
  finally:
    sys.argv = argv


def list_checkpoints(run_folder):
  d = os.path.join(run_folder, "checkpoints")
  out = []
  for f in os.listdir(d):
    m = re.fullmatch(r"(\d+)\.eqx", f)
    if m:
      out.append((int(m.group(1)), os.path.join(d, f)))
  return sorted(out)


_TEMPLATES: dict = {}


def _template(run_folder):
  """Checkpoint structure for one run folder, built once (same structure relabel.load_model
  and main.py use), so scoring many checkpoints of a run only reads their weights."""
  key = os.path.abspath(run_folder)
  if key not in _TEMPLATES:
    opts = load_opts(run_folder)
    dim = main_utils.get_data_from_opts(opts).shape[-1]
    model = main_utils.get_model_from_opts(load_opts(run_folder), input_shape=(dim,))
    opt_state = main_utils.get_optimizer_from_opts(opts).init(eqx.filter(model, eqx.is_array))
    zero = jax.random.PRNGKey(0)
    _TEMPLATES[key] = {"iter": -1,
                       "seeds": {"eval_model_seed": zero, "train_data_seed": zero, "train_model_seed": zero},
                       "opt_state": opt_state, "model": model}
    if opto.make_fn_from_opts(load_opts(run_folder)) is not opto.default_model_fwd_fn:
      print(f"WARNING: {run_folder} was trained with optogenetic manipulations; "
            "the analysis runs the plain model without them.")
  return _TEMPLATES[key]


def load_run(run_folder, ckpt=None):
  """Returns (model, opts, iteration). Default checkpoint: the run's last one.
  Loads exactly as relabel.load_model does (config.json -> model -> deserialise)."""
  ckpt = ckpt or last_checkpoint(run_folder)
  ck = eqx.tree_deserialise_leaves(ckpt, _template(run_folder))
  return ck["model"], load_opts(run_folder), int(ck["iter"])


def build_probe(opts, evaluator="fsl_val_rl"):
  """Rebuild one evaluator exactly as main.run_with_opts does. numpy dict with
  examples [B,n,D], labels [B,n], class_idxs [B,n], idx_types [B,n] (n = 3 base, 4 extended)."""
  data = main_utils.get_data_from_opts(opts)
  splits = main_utils.get_splits_from_opts(opts, data.shape)
  eval_seed = jax.random.PRNGKey(opts.eval_seed)
  eval_data_seed, _ = jax.random.split(eval_seed, 2)
  seeds = jax.random.split(eval_data_seed, len(opts.pe_names))
  i = list(opts.pe_names).index(evaluator)
  classes = splits["class"][opts.pe_classes[i]]
  class_sampler = partial(samplers.get_constant_burst_seq_idxs,
                          classes=classes, class_distr=jnp.ones(len(classes)) / len(classes),
                          num_seqs=opts.eval_iters,
                          context_len=smart_index(opts.pe_context_len, i, opts.train_context_len),
                          burstiness=opts.pe_burstiness[i],
                          distractor=smart_index(opts.pe_distract, i, 1),
                          no_support=smart_index(opts.pe_no_support, i, 0),
                          unique_rest=smart_index(opts.pe_unique_rest, i, 0))
  exemplar_sampler = partial(samplers.get_exemplar_inds,
                             allowed_inds=splits["exemplar"][opts.pe_exemplars[i]],
                             match_query_and_distractors=opts.match_query_and_distractors)
  sampler = samplers.make_data_sampler(
    class_sampler, exemplar_sampler,
    fs_relabel=splits["relabeling"][smart_index(opts.pe_fs_relabel_scheme, i, "None")],
    noise_scale=smart_index(opts.pe_noise_scale, i, opts.noise_scale_train),
    assign_query_label_random=smart_index(opts.pe_assign_query_label_random, i, 0),
    extended=getattr(opts, "extended", False), p_repeat=getattr(opts, "p_repeat", 0.5))
  out = sampler(seeds[i], data)
  return {k: np.asarray(v) for k, v in out.items()}


def model_dims(model):
  blocks = model.transformer.blocks
  H = blocks[0].attn.num_heads
  return len(blocks), H, model.transformer.embed_dim // H


def ablation_cache(model, T, heads):
  """cache/cache_mask that zero the values of the given (layer, head) pairs (as opto.py)."""
  L, H, dh = model_dims(model)
  cache = {"transformer_output": {"block_outputs": [{"attn_output": {}} for _ in range(L)]}}
  mask = {"transformer_output": {"block_outputs": [{"attn_output": {}} for _ in range(L)]}}
  for l, h in heads:
    c = cache["transformer_output"]["block_outputs"][l]["attn_output"]
    m = mask["transformer_output"]["block_outputs"][l]["attn_output"]
    c.setdefault("v", jnp.zeros((1, H, T, dh)))
    m["v"] = m.get("v", jnp.zeros((1, H, T, dh), dtype=bool)).at[0, h].set(True)
  return cache, mask


@eqx.filter_jit
def _forward(model, x, y, cache, cache_mask):
  key = jax.random.PRNGKey(0)                          # unused: no dropout in these runs
  f = lambda xx, yy: model.call_with_all_aux(examples=xx, labels=yy, key=key,
                                             cache=cache, cache_mask=cache_mask)
  r = jax.vmap(f)(x, y)
  attn = jnp.stack([b["attn_output"]["attn_scores"][:, 0]
                    for b in r["transformer_output"]["block_outputs"]])   # [L, B, H, T, T]
  out = {"attn": attn, "out": r["out"]}
  if "out_symbol" in r:
    out["out_symbol"] = r["out_symbol"]
  return out


def forward(model, probe, ablate=(), batch=1000):
  """Returns numpy dict: attn [L,B,H,T,T], out [B,n,C] (+ out_symbol [B,n-1,S] if extended)."""
  x, y = jnp.asarray(probe["examples"]), jnp.asarray(probe["labels"])
  T = 2 * y.shape[1] - 1
  cache, mask = ablation_cache(model, T, list(ablate))
  parts = []
  for s in range(0, x.shape[0], batch):
    parts.append({k: np.asarray(v) for k, v in _forward(model, x[s:s + batch], y[s:s + batch], cache, mask).items()})
  return {k: np.concatenate([p[k] for p in parts], axis=1 if k == "attn" else 0) for k in parts[0]}


def _log_softmax(z):
  z = z - z.max(-1, keepdims=True)
  return z - np.log(np.exp(z).sum(-1, keepdims=True))


def label_metrics(logits, y, ctx_labels):
  lp = _log_softmax(logits.astype(np.float64))
  rows = np.arange(len(y))
  in_ctx = np.zeros_like(logits, dtype=bool)
  in_ctx[rows[:, None], ctx_labels] = True
  masked = np.where(in_ctx, logits, -1e20)
  return {"acc": float((logits.argmax(-1) == y).mean()),
          "in_context_acc": float((masked.argmax(-1) == y).mean()),
          "loss": float(-lp[rows, y].mean()),
          "use_context_prob": float((np.exp(lp) * in_ctx).sum(-1).mean())}


def per_sequence_behaviour(fwd, probe) -> dict:
  """Per-probe-sequence behaviour [B] at the query (for paired tests across generations)."""
  y = probe["labels"]
  logits = fwd["out"][:, QUERY].astype(np.float64)
  t = y[:, QUERY]
  lp = _log_softmax(logits)
  rows = np.arange(len(t))
  in_ctx = np.zeros_like(logits, dtype=bool)
  in_ctx[rows[:, None], y[:, :QUERY]] = True
  masked = np.where(in_ctx, logits, -1e20)
  return {"acc": (logits.argmax(-1) == t).astype(float),
          "in_context_acc": (masked.argmax(-1) == t).astype(float),
          "loss": -lp[rows, t],
          "use_context_prob": (np.exp(lp) * in_ctx).sum(-1)}


def metrics(fwd, probe) -> dict:
  """Query-label metrics (base names, as in log.h5) + extended metrics if present."""
  y, c = probe["labels"], probe["class_idxs"]
  m = label_metrics(fwd["out"][:, QUERY], y[:, QUERY], y[:, :QUERY])
  if "out_symbol" in fwd and y.shape[1] > NEXT:
    m.update({"next_label_" + k: v for k, v in
              label_metrics(fwd["out"][:, NEXT], y[:, NEXT], y[:, :QUERY]).items()})
    sym = fwd["out_symbol"][:, QUERY]
    p = np.exp(_log_softmax(sym.astype(np.float64)))
    rows = np.arange(len(y))
    other = np.where(c[:, 0] == c[:, QUERY], c[:, 1], c[:, 0])
    valid = (c < sym.shape[-1]).all(1)
    rep, oth = p[rows, np.where(valid, c[:, QUERY], 0)], p[rows, np.where(valid, other, 0)]
    m["symbol_repeat_share"] = float((rep / (rep + oth))[valid].mean())
    m["symbol_use_context_prob"] = float((rep + oth)[valid].mean())
  return m
