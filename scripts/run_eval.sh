#!/usr/bin/env bash
# 闭环仿真评估（SoftGym）：每步由模型预测 pick/place，执行后重新拍照进入下一步
#
#   bash scripts/run_eval.sh single            # 评 5 个单臂任务
#   bash scripts/run_eval.sh bimanual          # 评 4 个单/双臂混合任务
#   NUM_EVALS=10 bash scripts/run_eval.sh single
#
# 结果写在 outputs/eval_<kind>/eval_<dataset>.yaml，average_success 为闭环折叠成功率。
set -euo pipefail

KIND=${1:-single}
N=${N:-1000}                       # 用哪个规模的 pkl（All_<N>.pkl）
NUM_EVALS=${NUM_EVALS:-50}        # 每个任务跑几个回合
PY=${PY:-python}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# PyFlex 运行期环境（见 deps/README.md）
export PYFLEXROOT=${PYFLEXROOT:-"${ROOT}/deps/PyFlex"}
export PYTHONPATH="${PYFLEXROOT}/bindings/build:${PYTHONPATH}"
export LD_LIBRARY_PATH="${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:${LD_LIBRARY_PATH}"
export CLOTH3D_PATH=${CLOTH3D_PATH:-"${ROOT}/data_generation/cloth3d"}
export FLEX_SHIM_PROPS=0

case "${KIND}" in
  single)   DATASET=single_sequential; CKPT_DIR=unimanual ;;
  bimanual) DATASET=bimanual_dual;     CKPT_DIR=bimanual ;;
  *) echo "用法: bash scripts/run_eval.sh [single|bimanual] [checkpoint.pth]" >&2; exit 1 ;;
esac
CKPT=${2:-outputs/${CKPT_DIR}/checkpoints/best.pth}

RUN_DIR="${ROOT}/outputs/eval_${KIND}"
mkdir -p "${RUN_DIR}/checkpoints"
ln -sf "$(realpath "${CKPT}")" "${RUN_DIR}/checkpoints/best.pth"

"${PY}" -m msufold \
  "dataset@train_dataset=${DATASET}" \
  "dataset@test_dataset=${DATASET}" \
  "train_dataset.n_samples=${N}" \
  "test_dataset.n_samples=${N}" \
  eval_only=true \
  load_best=true \
  "num_evals=${NUM_EVALS}" \
  hydra.run.dir="${RUN_DIR}"
