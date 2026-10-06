#!/bin/bash
# Runs every collapse condition, one after another (safe to re-run: finished
# generations are skipped). Start it inside tmux, e.g.
#   tmux new -s collapse
#   bash run_all_conditions.sh 2>&1 | tee collapse/run_all.log
#
# Needs generation 0 first: collapse/gen0_pool (trained on collapse/pool_gen0.h5).
#
# Argmax only gets 3 generations: with argmax, generation 0 makes no mistakes on
# its training pool, so every pool (and, with the same seeds, every model) is
# identical to generation 0. Three generations are enough to show that.

set -euo pipefail

#                     condition         method  T    keep  generations
bash run_generations.sh sample_T2         sample  2.0  0.0   10
bash run_generations.sh sample_T2_keep10  sample  2.0  0.1   10
bash run_generations.sh sample_T1         sample  1.0  0.0   10
bash run_generations.sh argmax            argmax  1.0  0.0   3

echo "All conditions finished at $(date)"