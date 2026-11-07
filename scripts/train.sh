#!/usr/bin/env bash
# Train the full ReVUA-CIR model (all 4 modules) on FashionIQ + CIRR.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT="${1:-./datasets}"
OUT_DIR="${2:-./runs/revua_cir_base}"

python -m revua_cir.train \
    --config configs/default.yaml \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUT_DIR}"
