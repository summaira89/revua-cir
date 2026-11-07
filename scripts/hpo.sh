#!/usr/bin/env bash
# Optuna hyperparameter search (TPE sampler + Hyperband pruning). Writes the
# best config found to configs/best_from_hpo.yaml, ready to feed into train.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT="${1:-./datasets}"
N_TRIALS="${2:-30}"

python -m revua_cir.hpo \
    --config configs/default.yaml \
    --data-root "${DATA_ROOT}" \
    --n-trials "${N_TRIALS}" \
    --max-epochs-per-trial 3 \
    --train-subset-size 5000 \
    --val-subset-size 500 \
    --study-name revua_cir_search \
    --storage sqlite:///runs/hpo.db \
    --output-config configs/best_from_hpo.yaml
