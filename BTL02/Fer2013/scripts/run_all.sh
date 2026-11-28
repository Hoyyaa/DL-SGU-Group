#!/usr/bin/env bash
# Chạy train -> evaluate -> visualize cho các config (mặc định: mọi configs/e*.yaml).
#   bash scripts/run_all.sh
#   SEED=1 bash scripts/run_all.sh configs/e1_small.yaml configs/e5_large_bn.yaml
#   WANDB=1 bash scripts/run_all.sh                # ghi log lên Weights & Biases
set -e
cd "$(dirname "$0")/.."
SEED=${SEED:-42}
EXTRA=()
[ "${WANDB:-0}" = "1" ] && EXTRA+=(--wandb)
CONFIGS=("$@")
[ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=(configs/e*.yaml)
for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  run="runs/${name}_s${SEED}"
  if [ -f "$run/last.pt" ]; then
    python -m fer.train --config "$cfg" --seed "$SEED" --resume "${EXTRA[@]}"
  else
    python -m fer.train --config "$cfg" --seed "$SEED" "${EXTRA[@]}"
  fi
  python -m fer.evaluate --run "$run"
  python -m fer.visualize --run "$run"
done
