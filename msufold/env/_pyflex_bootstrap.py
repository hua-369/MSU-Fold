"""兜底加载 pyflex。

pyflex 的编译产物在 deps/PyFlex/bindings/build，正常依赖 PYTHONPATH / LD_LIBRARY_PATH
（见 deps/prepare.sh，已固化到 conda 环境）。如果当前 shell 没带上这些变量（比如直接用
torchrun 启动、或环境是在变量固化前激活的），这里做兜底：把编译目录加进 sys.path，
并预先加载自带的 SDL2。
"""
import ctypes
import os
import sys


def _pyflex_root():
    root = os.environ.get("PYFLEXROOT")
    if root:
        return root
    here = os.path.dirname(os.path.abspath(__file__))  # msufold/env
    repo_root = os.path.dirname(os.path.dirname(here))  # 仓库根
    return os.path.join(repo_root, "deps", "PyFlex")


def ensure_pyflex():
    try:
        import pyflex

        return pyflex
    except ImportError:
        pass

    root = _pyflex_root()
    build = os.path.join(root, "bindings", "build")
    if not os.path.isdir(build):
        raise ImportError(
            f"找不到 pyflex 编译产物：{build}。请先编译：cd <repo> && . ./deps/compile.sh"
        )

    # pyflex.so 依赖自带的 libSDL2-2.0.so.0，缺 LD_LIBRARY_PATH 时先手动加载
    sdl = os.path.join(root, "external", "SDL2-2.0.4", "lib", "x64", "libSDL2-2.0.so.0")
    if os.path.isfile(sdl):
        try:
            ctypes.CDLL(sdl, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

    # pyflex.so 还链接了 CUDA 9.2 的 cudart（deps/compile.sh 的 CUDART9_PATH），
    # 否则动态链接阶段会直接段错误
    import glob

    cudart_candidates = []
    if os.environ.get("CUDART9_PATH"):
        cudart_candidates.append(os.path.join(os.environ["CUDART9_PATH"], "libcudart.so.9*"))
    cudart_candidates.append("/root/miniconda3/envs/cudart9/lib/libcudart.so.9*")
    cudart_candidates.extend(glob.glob("/root/miniconda3/envs/*/lib/libcudart.so.9*"))
    for pattern in cudart_candidates:
        matches = sorted(glob.glob(pattern))
        if matches:
            try:
                ctypes.CDLL(matches[0], mode=ctypes.RTLD_GLOBAL)
                break
            except OSError:
                continue

    if build not in sys.path:
        sys.path.insert(0, build)

    import pyflex

    return pyflex
