"""Checks that two training runs are identical: every array in log.h5, every
checkpoint byte for byte, and (if present) the contents of pool_gen0.h5.

Used to show a code change leaves the base version untouched:
    python compare_runs.py <run_folder_a> <run_folder_b>
"""
import filecmp
import os
import sys

import h5py
import numpy as np


def datasets(f):
  names = []
  f.visit(lambda k: names.append(k) if isinstance(f[k], h5py.Dataset) else None)
  return names


a, b = sys.argv[1], sys.argv[2]
ok = True
with h5py.File(os.path.join(a, 'log.h5'), 'r') as fa, h5py.File(os.path.join(b, 'log.h5'), 'r') as fb:
  ka, kb = datasets(fa), datasets(fb)
  same = set(ka) == set(kb) and all(np.array_equal(fa[k][()], fb[k][()]) for k in ka)
  print(f'log.h5: {len(ka)} arrays, identical: {same}')
  ok &= same

ca = sorted(os.listdir(os.path.join(a, 'checkpoints')))
cb = sorted(os.listdir(os.path.join(b, 'checkpoints')))
same = ca == cb and all(filecmp.cmp(os.path.join(a, 'checkpoints', c), os.path.join(b, 'checkpoints', c),
                                    shallow=False) for c in ca)
print(f'checkpoints: {len(ca)} files, byte-identical: {same}')
ok &= same

pa, pb = os.path.join(a, 'pool_gen0.h5'), os.path.join(b, 'pool_gen0.h5')
if os.path.exists(pa) and os.path.exists(pb):
  with h5py.File(pa, 'r') as fa, h5py.File(pb, 'r') as fb:
    same = sorted(fa.keys()) == sorted(fb.keys()) and all(np.array_equal(fa[k][()], fb[k][()]) for k in fa)
  print(f'pool_gen0.h5 contents identical: {same}')
  ok &= same

print('IDENTICAL' if ok else 'DIFFERENT')
sys.exit(0 if ok else 1)
