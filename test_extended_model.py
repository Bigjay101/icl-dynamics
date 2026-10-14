"""Step 2 test: the model's optional next-symbol output.

Run from the repo folder:
    python test_extended_model.py collapse/gen0_pool
(the argument is any finished base run folder, used to check old checkpoints still load)

Checks that
  1. without --extended, the model is identical to the original code (same weights, same outputs),
  2. a base checkpoint saved before this change still loads and gives the same outputs,
  3. with --extended, every shared weight is initialised exactly as in the base model,
  4. the extended model gives a label output at each image position (4 x 5) and a
     symbol output at each label position (3 x 50) on a 4-pair sequence,
  5. adding the symbol output does not change the label output.
"""
import importlib.util
import os
import subprocess
import sys
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import main_utils
import models
import samplers

run_folder = sys.argv[1] if len(sys.argv) > 1 else 'collapse/gen0_pool'

ARGS = ("--data_file omniglot_resnet18_randomized_order_s0.h5 --mixing_coeffs 1.0 --pt_burstiness 1 "
        "--train_context_len 2 --fs_relabel 5 --fs_relabel_split 8 2 0 --exemplar_split 1 4 0 "
        "--class_split 50 1473 100 --train_bs 32 --lr 0.00001 --d_model 64 --init_seed 5").split()
ok = True


def check(name, cond):
  global ok
  print(('PASS ' if cond else 'FAIL ') + name)
  ok = ok and bool(cond)


def opts_for(extra=()):
  o = main_utils.create_parser().parse_args(ARGS + list(extra))
  o.model_output_classes = o.fs_relabel  # what main.py defaults it to when fs_relabel > 0
  return o


def leaves(m):
  return jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_array))


# The original models.py, from the last commit before Step 2
orig_src = subprocess.run(['git', 'show', 'HEAD:models.py'], capture_output=True, text=True, check=True).stdout
spec = importlib.util.spec_from_loader('models_orig', loader=None)
models_orig = importlib.util.module_from_spec(spec)
exec(orig_src, models_orig.__dict__)

opts = opts_for()
data = main_utils.get_data_from_opts(opts)
splits = main_utils.get_splits_from_opts(opts, data.shape)
input_shape = (data.shape[-1],)


def sampler(extended):
  n = len(splits['class']['train'])
  cs = partial(samplers.get_constant_burst_seq_idxs, classes=splits['class']['train'],
               class_distr=jnp.ones(n) / n, num_seqs=256, context_len=2, burstiness=1,
               distractor=1, no_support=0, unique_rest=0)
  es = partial(samplers.get_exemplar_inds, allowed_inds=splits['exemplar']['train'],
               match_query_and_distractors=False)
  return samplers.make_data_sampler(cs, es, fs_relabel=splits['relabeling']['train'],
                                    extended=extended, p_repeat=0.5)(jax.random.PRNGKey(0), data)


def forward(model, batch):
  keys = jax.random.split(jax.random.PRNGKey(1), batch['examples'].shape[0])
  f = lambda x, y, k: model.call_with_all_aux(examples=x, labels=y, key=k)
  return jax.vmap(f)(batch['examples'], batch['labels'], keys)


base_batch = sampler(False)
ext_batch = sampler(True)

# 1. Base model identical to the original code
new_base = main_utils.get_model_from_opts(opts_for(), input_shape=input_shape)
o = opts_for()
o.init_seed = jax.random.PRNGKey(o.init_seed)
orig_base = models_orig.SequenceClassifier(
  example_shape=input_shape, example_type=o.data_type, num_classes=o.model_output_classes,
  embed_dim=o.d_model, key=o.init_seed, depth=o.depth, num_heads=o.num_heads,
  mlp_ratio=o.mlp_ratio, causal=(not o.not_causal), pos_embedding_type=o.pos_embedding_type,
  sin_time=o.sin_time, norm_layer=(eqx.nn.Identity if o.no_norm else eqx.nn.LayerNorm))
orig_base = main_utils.scale_model_init(orig_base, o.init_rescale)
check('base model has no symbol output', new_base.symbol_unembed is None)
check('base model weights identical to the original code',
      len(leaves(new_base)) == len(leaves(orig_base)) and
      all(np.array_equal(a, b) for a, b in zip(leaves(new_base), leaves(orig_base))))
check('base model outputs identical to the original code',
      np.array_equal(np.asarray(forward(new_base, base_batch)['out']),
                     np.asarray(forward(orig_base, base_batch)['out'])))

# 2. An existing base checkpoint still loads
import relabel  # noqa: E402  (uses load_model from the pipeline)
cfg = os.path.join(run_folder, 'config.json')
if os.path.exists(cfg):
  ckpt = relabel.last_checkpoint(run_folder)
  model, fwd_fn, _, it = relabel.load_model(cfg, ckpt)
  check(f'base checkpoint {os.path.basename(ckpt)} loads with the new code', model.symbol_unembed is None)
  out = forward(model, base_batch)['out'][:, -1, :]
  acc = float((jnp.argmax(out, -1) == base_batch['labels'][:, -1]).mean())
  check(f'loaded base model still solves the task (train accuracy {acc:.3f})', acc > 0.95)
else:
  print(f'SKIP checkpoint check: {cfg} not found')

# 3. Extended model: shared weights initialised as in the base model
ext = main_utils.get_model_from_opts(opts_for(['--extended']), input_shape=input_shape)
check('extended model has a symbol output over 50 classes',
      ext.symbol_unembed is not None and ext.symbol_unembed.out_features == 50)
shared = eqx.tree_at(lambda m: m.symbol_unembed, ext, None)
check('extended model: every shared weight equals the base model\'s',
      all(np.array_equal(a, b) for a, b in zip(leaves(shared), leaves(new_base))))

# 4. Output shapes on a 4-pair sequence
r = forward(ext, ext_batch)
check(f'label output shape {tuple(r["out"].shape)} = (256, 4, 5)', tuple(r['out'].shape) == (256, 4, 5))
check(f'symbol output shape {tuple(r["out_symbol"].shape)} = (256, 3, 50)',
      tuple(r['out_symbol'].shape) == (256, 3, 50))

# 5. The symbol output does not change the label output
r_base_arch = forward(new_base, ext_batch)
check('label output unchanged by adding the symbol output',
      np.allclose(np.asarray(r['out']), np.asarray(r_base_arch['out'])))

# jit works
jit_fwd = eqx.filter_jit(forward)
check('extended forward pass compiles under jit', jit_fwd(ext, ext_batch)['out_symbol'].shape == (256, 3, 50))

print('\nALL PASSED' if ok else '\nSOME CHECKS FAILED')
sys.exit(0 if ok else 1)
