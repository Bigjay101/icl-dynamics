# Tests for the induction-head analysis in interp/.
# Run from the repo folder:  python test_interp_heads.py
# With a trained run it also checks the accuracy against the training log:
# python test_interp_heads.py collapse/gen0_pool

import sys

import numpy as np

from interp.head_scores import (all_scores, first_crossing, first_move, formation_iter, lead_time,
                                make_reader, per_sequence, pick_heads, rise_points)

ok = True


def check(name, cond):
  global ok
  cond = bool(cond)
  print(('PASS ' if cond else 'FAIL ') + name)
  ok = ok and cond


rng = np.random.default_rng(0)
B, L, H = 200, 2, 8
c = np.stack([rng.choice(50, 2, replace=False) for _ in range(B)])
c = np.concatenate([c, c[np.arange(B), rng.integers(0, 2, B)][:, None]], 1)

T = 5
causal = np.tril(np.ones((T, T)))
attn = np.broadcast_to(causal / causal.sum(-1, keepdims=True), (L, B, H, T, T)).copy()
attn[0, :, 2] = np.eye(T, k=-1)
attn[0, :, 2, 0, 0] = 1

q = make_reader(c, 2, "query")
attn[1, :, 3, 4] = q.correct / q.correct.sum(1, keepdims=True)

sc = all_scores(attn, [q], 3)
check('planted previous-token head scores 1', np.isclose(sc['prev_token'][0, 2], 1))
check('planted induction head has matching score 1', np.isclose(sc['matching'][1, 3], 1))
check('uniform heads have matching score ~0', abs(sc['matching'][1, 0]) < 0.2)

pk = pick_heads(sc)
check('pick_heads finds IH 1:3 and PT 0:2', pk['induction'] == [1, 3] and pk['prev_token'] == [0, 2])
check('previous-token set holds exactly the PT head', pk['prev_token_set'] == [[0, 2]])

ps = per_sequence(attn, q, 3, (1, 3), [(0, 2)])
check('per-sequence values average to the head scores',
      np.isclose(ps['matching'].mean(), sc['matching'][1, 3]) and np.isclose(ps['prev_token'].mean(), sc['prev_token'][0, 2]))
check('chance level is 1/5 with one matching label', np.isclose(sc['chance'], 0.2))


r = make_reader(np.array([[7, 9, 7]]), 2, 'query')
check('query mask marks y0 only', r.correct[0].tolist() == [False, True, False, False, False])
check('incorrect mask marks y1 only', r.incorrect[0].tolist() == [False, False, False, True, False])
check('match mask marks x0 only', r.match_exemplar[0].tolist() == [True, False, False, False, False])

r = make_reader(np.array([[7, 9, 7, 7]]), 3, 'next')
check('extended next-symbol reader at token 6', r.pos == 6)
check('extended next mask marks y0 and yq', r.correct[0].tolist() == [False, True, False, False, False, True, False])


m, a = [0.9, 0.88, 0.7, 0.5, 0.3], [0.99, 0.99, 0.98, 0.85, 0.6]
check('first_crossing', first_crossing(m) == 2 and first_crossing(a) == 3)
check('lead_time = +1 generation', lead_time(m, a)['lead'] == 1)
check('formation_iter interpolates', np.isclose(formation_iter([0, 10, 20], [0.0, 0.4, 0.8], 0.5), 12.5))

rp = rise_points([0, 10, 20, 30], [0.5, 0.5, 0.9, 1.0])
check('rise_points: 50% of the rise (0.75) reached at iteration 16.25', np.isclose(rp['iter_50pct'], 16.25))

rng2 = np.random.default_rng(1)
x0 = rng2.normal(1.0, 0.1, 1000)
xs = [x0, x0 + rng2.normal(0, 0.01, 1000), x0 - 0.05 + rng2.normal(0, 0.01, 1000)]
g, zs = first_move(x0, xs, 'down')
check('first_move flags the generation that drops beyond noise (gen 2), not gen 1', g == 2)
g, _ = first_move(x0, [x0, x0 + 0.05], 'up')
check('first_move with worse=up flags an increase (e.g. loss)', g == 1)


try:
  import jax
  import jax.numpy as jnp
  from functools import partial
  import main_utils
  import opto
  from interp import jax_backend as jb
except ImportError as e:
  print('SKIP JAX checks:', e)
else:
  ARGS = ("--data_file omniglot_resnet18_randomized_order_s0.h5 --mixing_coeffs 1.0 --pt_burstiness 1 "
          "--train_context_len 2 --fs_relabel 5 --fs_relabel_split 8 2 0 --exemplar_split 1 4 0 "
          "--pe_names fsl_train fsl_val_rl --pe_classes train train --pe_exemplars train train "
          "--pe_fs_relabel_scheme train val --pe_burstiness 1 1 --eval_iters 64 "
          "--class_split 50 1473 100 --train_bs 32 --lr 0.00001 --d_model 64 --init_seed 5").split()

  for extended in (False, True):
    extra = ['--extended'] if extended else []

    opts = main_utils.create_parser().parse_args(ARGS + extra)
    opts.model_output_classes = opts.fs_relabel
    probe = jb.build_probe(opts, 'fsl_val_rl')

    mopts = main_utils.create_parser().parse_args(ARGS + extra)
    mopts.model_output_classes = mopts.fs_relabel
    model = main_utils.get_model_from_opts(mopts)

    tag = 'extended' if extended else 'base'
    n = probe['labels'].shape[1]
    check(f'[{tag}] probe has {3 + extended} items per sequence', n == 3 + extended)

    fwd = jb.forward(model, probe)
    keys = jax.random.split(jax.random.PRNGKey(0), len(probe['labels']))
    ref = jax.vmap(partial(opto.default_model_fwd_fn, model=model))(
      x=jnp.asarray(probe['examples']), y=jnp.asarray(probe['labels']), key=keys)['out']
    check(f'[{tag}] analysis forward matches the normal forward', np.allclose(fwd['out'], np.asarray(ref), atol=1e-5))
    check(f'[{tag}] attention shape [L,B,H,T,T]', fwd['attn'].shape == (2, len(probe['labels']), 8, 2 * n - 1, 2 * n - 1))
    check(f'[{tag}] attention rows sum to 1', np.allclose(fwd['attn'].sum(-1), 1, atol=1e-5))

    same = jb.forward(model, probe, ablate=[])
    diff = jb.forward(model, probe, ablate=[(1, 3)])
    check(f'[{tag}] empty ablation leaves outputs unchanged', np.allclose(same['out'], fwd['out']))
    check(f'[{tag}] ablating head 1:3 changes the output', not np.allclose(diff['out'], fwd['out']))
    check(f'[{tag}] ablation does not change layer-0 attention', np.allclose(diff['attn'][0], fwd['attn'][0]))
    check(f'[{tag}] metrics computed', 0 <= jb.metrics(fwd, probe)['acc'] <= 1)

  if len(sys.argv) > 1:
    from interp.find_heads_gen0 import logged_metric
    run = sys.argv[1]
    model, ropts, it = jb.load_run(run)
    probe = jb.build_probe(ropts, 'fsl_val_rl')
    acc = jb.metrics(jb.forward(model, probe), probe)['acc']
    logged = logged_metric(run, it)
    print(f'  probe acc {acc:.4f}; log.h5 fsl_val_rl/acc {logged}')
    check('log.h5 has an fsl_val_rl/acc entry', logged is not None)
    if logged is not None and logged['iter'] == it:
      check('rebuilt probe reproduces the accuracy main.py logged at that checkpoint',
            abs(acc - logged['value']) < 0.002)
    elif logged is not None:
      print(f"NOTE no eval was logged at exactly iteration {it} (closest {logged['iter']}); "
            "the two numbers above should still be close")

print('\nALL PASSED' if ok else '\nSOME CHECKS FAILED')
sys.exit(0 if ok else 1)