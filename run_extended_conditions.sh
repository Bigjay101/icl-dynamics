#!/bin/bash
# Runs every extended-version collapse condition, one after another. Safe to
# re-run: finished generations (and generation-0 models) are skipped.
#
#   LR=0.00003 bash run_extended_conditions.sh 2>&1 | tee -a collapse_ext/run_all.log
#
# LR must be set: the learning rate chosen on the validation set for the
# extended version (at 1e-5 the extended model is not converged after 1M
# sequences; see check_extended_run.py).
#
# Conditions (all trained from scratch on 1M-sequence pools, init seed 5):
#   gen0_p50  generation 0 on real data, next symbol repeats with p = 0.5
#   argmax    3 generations, argmax relabelling (from gen0_p50)
#   T1        10 generations, sampling at T=1 (from gen0_p50): the main result
# With RUN_DETERMINISTIC=1, also
#   gen0_p0   generation 0 with p = 0 (the next symbol is always the other one)
#   T1_p0     10 generations, sampling at T=1 (from gen0_p0): shows whether the
#             drift needs randomness in the real data
# A p=0 condition needs its own generation 0: a model trained on 50/50 data is
# not a model of the p=0 task.

set -euo pipefail

PY=${PY:-python}
LR=${LR:?Set LR to the learning rate chosen on the validation set, e.g. LR=0.00003}
BASE=${BASE:-collapse_ext}
TRAIN_ITERS=${TRAIN_ITERS:-1000000}
CKPT_EVERY=${CKPT_EVERY:-10000}
EVAL_EVERY=${EVAL_EVERY:-5000}
INIT_SEED=${INIT_SEED:-5}
POOL_SIZE=${POOL_SIZE:-1000000}
RUN_DETERMINISTIC=${RUN_DETERMINISTIC:-0}
GENERATIONS=${GENERATIONS:-10}

# Same task settings as run_generations.sh (the paper's Figure 3a run)
TRAIN_ARGS="--data_file omniglot_resnet18_randomized_order_s0.h5 --mixing_coeffs 1.0 --pt_burstiness 1 \
 --train_context_len 2 --fs_relabel 5 --fs_relabel_split 8 2 0 --exemplar_split 1 4 0 \
 --pe_names fsl_train fsl_val_rl fsl_train_valex fsl_test_class --pe_classes train train train test \
 --pe_exemplars train train val train --pe_fs_relabel_scheme train val train train --pe_burstiness 1 1 1 1 \
 --train_bs 32 --eval_iters 1000 --lr 0.00001 --d_model 64 --class_split 50 1473 100 --raw_name"

mkdir -p "$BASE"
FINAL_CKPT=$(printf '%011d.eqx' "$TRAIN_ITERS")
echo "=== Extended conditions in $BASE: lr=$LR train_iters=$TRAIN_ITERS deterministic=$RUN_DETERMINISTIC ($(date)) ==="

train_gen0() {
  local name=$1 p=$2
  local run="$BASE/$name"
  if [ -f "$run/checkpoints/$FINAL_CKPT" ]; then
    echo "[$(date '+%H:%M:%S')] $name: already trained, skipping"
  else
    rm -rf "$run"
    echo "[$(date '+%H:%M:%S')] $name: training generation 0 (p_repeat=$p)"
    $PY main.py $TRAIN_ARGS --extended --p_repeat "$p" --lr "$LR" --train_iters "$TRAIN_ITERS" \
      --eval_every "$EVAL_EVERY" --ckpt_every "$CKPT_EVERY" --init_seed "$INIT_SEED" \
      --pool_size "$POOL_SIZE" --base_folder "$BASE" --run "$name" > "$BASE/${name}_train.log" 2>&1 \
      || { echo "ERROR: training $name failed; end of $BASE/${name}_train.log:"; tail -n 5 "$BASE/${name}_train.log"; exit 1; }
  fi
  # Report convergence. Not fatal: thresholds are strict, and the numbers are
  # what goes in the write-up, but read the FAIL lines before trusting later runs.
  $PY check_extended_run.py "$run" --full | tail -n 20 || echo "WARNING: $name did not pass every convergence check (see above)"
}

loop() {
  local gen0=$1 p=$2 cond=$3 method=$4 temp=$5 ngen=$6
  GEN0_RUN="$BASE/$gen0" GEN0_POOL="$BASE/$gen0/pool_gen0.h5" BASE="$BASE" PY="$PY" \
    TRAIN_ITERS="$TRAIN_ITERS" CKPT_EVERY="$CKPT_EVERY" EVAL_EVERY="$EVAL_EVERY" INIT_SEED="$INIT_SEED" \
    EXTRA_ARGS="--extended --p_repeat $p --lr $LR" \
    bash run_generations.sh "$cond" "$method" "$temp" 0.0 "$ngen"
}

train_gen0 gen0_p50 0.5
loop gen0_p50 0.5 argmax argmax 1.0 3
loop gen0_p50 0.5 T1 sample 1.0 "$GENERATIONS"

if [ "$RUN_DETERMINISTIC" = "1" ]; then
  train_gen0 gen0_p0 0.0
  loop gen0_p0 0.0 T1_p0 sample 1.0 "$GENERATIONS"
fi

echo "All extended conditions finished at $(date)"
