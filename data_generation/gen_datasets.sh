#!/usr/bin/env bash
# 生成训练数据：专家演示（SoftGym 仿真）-> 打包成 pkl
#
#   bash data_generation/gen_datasets.sh              # 默认每个任务 1000 条
#   N_DEMOS=100 bash data_generation/gen_datasets.sh  # 只做 100 条（快很多）
#   SKIP_DEMOS=1 bash data_generation/gen_datasets.sh # 跳过仿真，只重新打包
#
# 产出：
#   datasets/single_data_sequential/All_<N>.pkl   单臂（unimanual）时序数据集
#   datasets/dual_data_sequential/All_<N>.pkl     双臂（单/双臂混合步）时序数据集
#   datasets/softgym_cache/<Cloth>.pkl            闭环评估/推理用的初始布料状态
#
# 支持断点续跑：某个任务已有 M 条就只补跑 N_DEMOS-M 条。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEN="${ROOT}/data_generation"
PY=${PY:-python}
N_DEMOS=${N_DEMOS:-1000}
N_CONFIGS=${N_CONFIGS:-1000}
SKIP_DEMOS=${SKIP_DEMOS:-0}

# 仿真环境变量（PyFlex 已编译，见 deps/README.md）
export PYFLEXROOT=${PYFLEXROOT:-"${ROOT}/deps/PyFlex"}
export PYTHONPATH="${PYFLEXROOT}/bindings/build:${GEN}:${PYTHONPATH}"
export LD_LIBRARY_PATH="${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:${LD_LIBRARY_PATH}"
export CLOTH3D_PATH=${CLOTH3D_PATH:-"${GEN}/cloth3d"}
export FLEX_SHIM_PROPS=0

SINGLE_TASKS=(
  "CornerFold:Square"
  "TriangleFold:Square"
  "StraightFold:Rectangular"
  "TshirtFold:Tshirt"
  "TrousersFold:Trousers"
)
DUAL_TASKS=(
  "MixedSquareHalfFold"
  "MixedSquareCornerFold"
  "MixedTshirtFold"
  "MixedTrousersFold"
)

log() { echo -e "\n[$(date +%H:%M:%S)] $*"; }

"${PY}" -c "import pyflex" || { echo "错误: pyflex 不可用，请先编译（见 deps/README.md）"; exit 1; }

cd "${GEN}"

# ---------- 1. 初始布料 configurations ----------
log "步骤 1/4: 生成初始 configurations"
for cloth in Square Rectangular Tshirt Trousers; do
  if [ -f "configs/${cloth}.pkl" ]; then
    echo "  configs/${cloth}.pkl 已存在，跳过"
  else
    "${PY}" generate_configs.py --cloth_type "${cloth}" --num_configs "${N_CONFIGS}"
  fi
done

count_demos() { [ -d "$1" ] && find "$1" -maxdepth 1 -mindepth 1 -type d | wc -l || echo 0; }

# ---------- 2. 专家演示 ----------
if [ "${SKIP_DEMOS}" = "1" ]; then
  log "步骤 2/4: SKIP_DEMOS=1，跳过仿真"
else
  log "步骤 2/4: 生成单臂演示（每种任务 ${N_DEMOS} 条）"
  for item in "${SINGLE_TASKS[@]}"; do
    task="${item%%:*}"; cloth="${item##*:}"
    have=$(count_demos "raw_data/${task}"); need=$((N_DEMOS - have))
    [ "${need}" -le 0 ] && { echo "  ${task}: 已有 ${have} 条，跳过"; continue; }
    echo "  ${task} (${cloth}): 再生成 ${need} 条"
    "${PY}" generate_demo_fold.py --task "${task}" --cloth_type "${cloth}" \
      --randomize_pose --num_demonstrations "${need}"
  done

  log "步骤 2b: 生成双臂演示（每种任务 ${N_DEMOS} 条）"
  for task in "${DUAL_TASKS[@]}"; do
    have=$(count_demos "raw_data_mixed/${task}"); need=$((N_DEMOS - have))
    [ "${need}" -le 0 ] && { echo "  ${task}: 已有 ${have} 条，跳过"; continue; }
    echo "  ${task}: 再生成 ${need} 条"
    "${PY}" bimanual/generate_demo_fold.py --task "${task}" \
      --randomize_pose --num_demonstrations "${need}"
  done
fi

# ---------- 3. softgym cache（评估/推理的初始状态） ----------
log "步骤 3/4: 拷贝初始状态到 datasets/softgym_cache"
mkdir -p "${ROOT}/datasets/softgym_cache"
cp -n configs/*.pkl "${ROOT}/datasets/softgym_cache/"

# ---------- 4. 打包 ----------
log "步骤 4/4: 打包时序数据集"
cd "${ROOT}"
"${PY}" scripts/create_unimanual_sequential_dataset.py \
  --use_rgb --tasks All --n_demos "${N_DEMOS}" \
  --root "${GEN}/raw_data" \
  --save_path_root "${ROOT}/datasets/single_data_sequential"

cd "${GEN}"
"${PY}" bimanual/create_dataset.py \
  --use_rgb --tasks All --n_demos "${N_DEMOS}" \
  --root raw_data_mixed \
  --save_path_root "${ROOT}/datasets/dual_data_sequential"

log "完成，产出："
ls -lh "${ROOT}/datasets/single_data_sequential/" "${ROOT}/datasets/dual_data_sequential/"
