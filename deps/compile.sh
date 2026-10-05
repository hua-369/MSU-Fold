# 编译 PyFlex 的 python bindings（先 conda activate MSU-Fold，并在仓库根目录执行）
#   bash deps/compile.sh
set -e
cd deps/PyFlex/bindings
rm -rf build
mkdir build
cd build
PY_VERSION=${PYBIND11_PYTHON_VERSION:-$(python -c 'import sys; print("{}.{}".format(sys.version_info.major, sys.version_info.minor))')}
echo "Building PyFlex bindings for python ${PY_VERSION}"
cmake -DPYBIND11_PYTHON_VERSION=${PY_VERSION} \
      -Dpybind11_DIR=$(python -m pybind11 --cmakedir) ..
make -j
cd ../../..
echo "Done: $(ls deps/PyFlex/bindings/build/*.so)"
