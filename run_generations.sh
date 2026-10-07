#!/bin/bash
# Runs the model-collapse loop for one condition:
#   for n = 1..G:  relabel with generation n-1's model -> pool_gen{n}.h5
#                  train generation n from scratch on pool_gen{n}.h5
#
# Usage:
#   bash run_generations.sh <condition> <method> <temperature> <keep_real_frac> <n_generations>
# e.g.
#   bash run_generations.sh sample_T2 sample 2.0 0.0 10
#
# Every condition starts from the same generation 0 (GEN0_RUN trained on GEN0_POOL).
# Results go to $BASE/<condition>/: pool_gen{n}.h5 (+ _summary.json), gen{n}/ run
# folders, and gen{n}_relabel.log / gen{n}_train.log.
#
# Safe to re-run: finished generations are skipped. An unfinished generation's
# run folder is deleted and retrained, since main.py appends to log.h5 and a
# restart in the same folder would mix two runs in one log.
#
# Settings can be overridden with environment variables, e.g.
#   TRAIN_ITERS=20000 bash run_generations.sh ...

set -euo pipefail

if [ $# -ne 5 ]; then
  echo "Usage: bash run_generations.sh <condition> <method: argmax|sample> <temperature> <keep_real_frac> <n_generations>"
  exit 1
fi
COND=$1
METHOD=$2
TEMP=$3
KEEP=$4
NGEN=$5

PY=${PY:-python}
BASE=${BASE:-collapse}
GEN0_RUN=${GEN0_RUN:-collapse/gen0_pool}
GEN0_POOL=${GEN0_POOL:-collapse/pool_gen0.h5}
TRAIN_ITERS=${TRAIN_ITERS:-1000000}
CKPT_EVERY=${CKPT_EVERY:-10000}
EVAL_EVERY=${EVAL_EVERY:-5000}
INIT_SEED=${INIT_SEED:-5}
# Sampling seed for generation n is RELABEL_SEED_BASE*1000 + n, so each
# generation draws fresh random labels (the same seed every generation would
# flip the same rows each time). KEEP_REAL_SEED stays fixed, so the same rows
# keep their true label in every generation.
RELABEL_SEED_BASE=${RELABEL_SEED_BASE:-0}
KEEP_REAL_SEED=${KEEP_REAL_SEED:-0}

# Same settings as the paper's Figure 3a run (ih_paper_runs.sh)
TRAIN_ARGS="--data_file omniglot_resnet18_randomized_order_s0.h5 --mixing_coeffs 1.0 --pt_burstiness 1 \
 --train_context_len 2 --fs_relabel 5 --fs_relabel_split 8 2 0 --exemplar_split 1 4 0 \
 --pe_names fsl_train fsl_val_rl fsl_train_valex fsl_test_class --pe_classes train train train test \
 --pe_exemplars train train val train --pe_fs_relabel_scheme train val train train --pe_burstiness 1 1 1 1 \
 --train_bs 32 --eval_iters 1000 --lr 0.00001 --d_model 64 --class_split 50 1473 100 --raw_name"

OUT="$BASE/$COND"
mkdir -p "$OUT"
FINAL_CKPT=$(printf '%011d.eqx' "$TRAIN_ITERS")

for f in "$GEN0_POOL" "$GEN0_RUN/config.json"; do
  [ -e "$f" ] || { echo "ERROR: $f not found (generation 0 is needed first)"; exit 1; }
done

echo "=== Condition $COND: method=$METHOD T=$TEMP keep_real_frac=$KEEP generations=$NGEN ==="
echo "    generation 0: $GEN0_RUN on $GEN0_POOL; train_iters=$TRAIN_ITERS ckpt_every=$CKPT_EVERY"

for n in $(seq 1 "$NGEN"); do
  if [ "$n" -eq 1 ]; then
    PREV_RUN=$GEN0_RUN; PREV_POOL=$GEN0_POOL
  else
    PREV_RUN="$OUT/gen$((n-1))"; PREV_POOL="$OUT/pool_gen$((n-1)).h5"
  fi
  POOL="$OUT/pool_gen$n.h5"
  SUMMARY="$OUT/pool_gen${n}_summary.json"
  RUN="$OUT/gen$n"

  if [ -f "$RUN/checkpoints/$FINAL_CKPT" ]; then
    echo "[$(date '+%H:%M:%S')] gen $n: already finished, skipping"
    continue
  fi

  # 1. Relabel (the summary is written last, so its presence means the pool is complete)
  if [ -f "$SUMMARY" ]; then
    echo "[$(date '+%H:%M:%S')] gen $n: pool exists, skipping relabel"
  else
    echo "[$(date '+%H:%M:%S')] gen $n: relabelling with $PREV_RUN"
    LOG="$OUT/gen${n}_relabel.log"
    $PY relabel.py --run_folder "$PREV_RUN" --pool_in "$PREV_POOL" --pool_out "$POOL" \
      --method "$METHOD" --temperature "$TEMP" --keep_real_frac "$KEEP" \
      --relabel_seed $((RELABEL_SEED_BASE * 1000 + n)) --keep_real_seed "$KEEP_REAL_SEED" \
      > "$LOG" 2>&1 || { echo "ERROR: relabel for gen $n failed; end of $LOG:"; tail -n 5 "$LOG"; exit 1; }
    grep -E "wrong|distractor|not in|kept|query label freq" "$LOG" || true
  fi

  # 2. Train generation n from scratch on its pool
  rm -rf "$RUN"
  LOG="$OUT/gen${n}_train.log"
  echo "[$(date '+%H:%M:%S')] gen $n: training"
  $PY main.py $TRAIN_ARGS --train_iters "$TRAIN_ITERS" --eval_every "$EVAL_EVERY" \
    --ckpt_every "$CKPT_EVERY" --init_seed "$INIT_SEED" \
    --base_folder "$OUT" --run "gen$n" --pool_file "$POOL" \
    > "$LOG" 2>&1 || { echo "ERROR: training gen $n failed; end of $LOG:"; tail -n 5 "$LOG"; exit 1; }
  [ -f "$RUN/checkpoints/$FINAL_CKPT" ] || { echo "ERROR: gen $n did not save its final checkpoint; see $LOG"; exit 1; }
  echo "[$(date '+%H:%M:%S')] gen $n: done ($(grep 'total training time' "$OUT/gen${n}_train.log"))"
done

echo "=== Condition $COND finished ==="
