# deps: SoftGym / PyFlex

Data generation, closed-loop evaluation and skill rollouts all run on SoftGym
(PyFlex). PyFlex is a C++/CUDA extension that must be compiled locally, so it is
**not shipped with this repository**.

## 1. Get the PyFlex sources

PyFlex comes from [SoftGym](https://github.com/Xingyu-Lin/softgym); this project uses
the CLOTH3D-enabled version shipped with [BiFold](https://github.com/Barbany/bifold)
(MIT licensed). Copy its `deps/PyFlex` here:

```bash
git clone https://github.com/Barbany/bifold /tmp/bifold
cp -r /tmp/bifold/deps/PyFlex ./deps/PyFlex
```

## 2. Build

```bash
conda activate MSU-Fold
cd /path/to/MSU-Fold
bash deps/compile.sh          # -> deps/PyFlex/bindings/build/pyflex*.so
```

## 3. Export the runtime environment

Add these to your shell profile (or to the conda env vars):

```bash
export PYFLEXROOT=/path/to/MSU-Fold/deps/PyFlex
export PYTHONPATH=$PYFLEXROOT/bindings/build:$PYTHONPATH
export LD_LIBRARY_PATH=$PYFLEXROOT/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH
export CLOTH3D_PATH=/path/to/MSU-Fold/data_generation/cloth3d   # Tshirt / Trousers
export FLEX_SHIM_PROPS=0
```

Check:

```bash
python -c "import pyflex; print('pyflex OK')"
```

> Real-robot inference does not need PyFlex; skip this section in that case.
