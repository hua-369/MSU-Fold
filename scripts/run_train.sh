#!/usr/bin/env bash
# 训练（单臂 / 双臂共用同一套共享热力图头权重）
#
#   bash scripts/run_train.sh single        # 单臂：1000 条演示，40 epoch，num_scales=8
#   bash scripts/run_train.sh bimanual      # 双臂（含单臂混合步）
#   N=100 EPOCHS=100 K=8 bash scripts/run_train.sh single   # 小规模 / 改采样层数
set -euo pipefail

KIND=${1:-single}
N=${N:-1000}
EPOCHS=${EPOCHS:-40}
K=${K:-8}                 # MultiScaleVisualAdapter 采样层数（默认 8）
BATCH=${BATCH:-4}
GPUS=${GPUS:-"0,1,2,3"}
PY=${PY:-python}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

case "${KIND}" in
  single)   DATASET=single_sequential ;;
  bimanual) DATASET=bimanual_dual ;;
  *) echo "用法: bash scripts/run_train.sh [single|bimanual]" >&2; exit 1 ;;
esac

DEVICE_IDS="[${GPUS}]"

"${PY}" -m msufold \
  "dataset@train_dataset=${DATASET}" \
  "dataset@test_dataset=${DATASET}" \
  "train_dataset.n_samples=${N}" \
  "test_dataset.n_samples=${N}" \
  "batch_size=${BATCH}" \
  "epochs=${EPOCHS}" \
  "device_ids=${DEVICE_IDS}" \
  "model.multi_scale=true" \
  "model.num_scales=${K}"
