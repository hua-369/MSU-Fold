#!/usr/bin/env bash
# 一句自然语言指令 -> 选择一个折叠技能 -> 闭环执行全部折叠步骤（仿真 / 真机通用）
#
# 用法:
#   bash run_skill_softgym.sh --instruction "把这件 T 恤折起来" --num-evals 1
#   bash run_skill_softgym.sh --skill bimanual-tshirt-fold --num-evals 1
#   bash run_skill_softgym.sh --backend real --robot-factory myrobot:build_backend \
#        --instruction "用两只手臂把这件 T 恤折好"
#
# 注意：必须直接运行 inference_skill_softgym.py（python 文件方式），
# 不能用 `python -m msufold ...`（那是 hydra 训练入口）。

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PY:-python}

# 仿真（PyFlex）运行期环境变量，见 deps/README.md
export PYFLEXROOT=${PYFLEXROOT:-"${REPO_ROOT}/deps/PyFlex"}
export PYTHONPATH="${PYFLEXROOT}/bindings/build:${PYTHONPATH}"
export LD_LIBRARY_PATH="${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:${LD_LIBRARY_PATH}"
export CLOTH3D_PATH=${CLOTH3D_PATH:-"${REPO_ROOT}/data_generation/cloth3d"}
export FLEX_SHIM_PROPS=0

cd "${REPO_ROOT}" || exit 1
exec "${PY}" "${REPO_ROOT}/inference_skill_softgym.py" "$@"
