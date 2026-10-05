# 设置 PyFlex 运行期环境变量（source 本文件后即可跑仿真）
#   . deps/prepare.sh
export PYFLEXROOT=${PWD}/deps/PyFlex
export PYTHONPATH=${PYFLEXROOT}/bindings/build:${PYTHONPATH}
export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:${LD_LIBRARY_PATH}
export CLOTH3D_PATH=${CLOTH3D_PATH:-${PWD}/data_generation/cloth3d}
export FLEX_SHIM_PROPS=0
