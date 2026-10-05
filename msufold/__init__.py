"""msufold 包初始化：补齐 PyFlex（SoftGym 仿真）的运行期环境变量。

pyflex 依赖 LD_LIBRARY_PATH 指向自带的 SDL2（以及可选的 CUDA 9.2 运行时），
而该变量只在进程启动时被动态链接器读取，导入后再补就晚了。因此在导入本包时
检测一次，缺什么补什么，然后用同样的参数重启自身进程。

需要在 shell 里先 export 的变量（见 README「环境准备」）：
    PYFLEXROOT      deps/PyFlex 根目录
    CLOTH3D_PATH    Cloth3D 网格目录（Tshirt / Trousers 场景需要）
    CUDART9_PATH    可选，CUDA 9.2 运行时 lib 目录
"""
import os
import sys

_ENV_READY = "_MSUFOLD_ENV_READY"


def _fix_runtime_env():
    if os.environ.get(_ENV_READY) == "1":
        return

    pyflex_root = os.environ.get("PYFLEXROOT")
    if not pyflex_root:
        # 没有编译 PyFlex（例如只做真机推理）就不需要仿真环境
        return

    missing = []
    cudart9 = os.environ.get("CUDART9_PATH")
    if cudart9 and os.path.isdir(cudart9):
        missing.append(cudart9)
    sdl = os.path.join(pyflex_root, "external", "SDL2-2.0.4", "lib", "x64")
    if os.path.isdir(sdl):
        missing.append(sdl)

    ld = os.environ.get("LD_LIBRARY_PATH", "")
    missing = [d for d in missing if d not in ld]
    if not missing:
        return

    os.environ["LD_LIBRARY_PATH"] = ":".join(missing + ([ld] if ld else []))
    os.environ.setdefault("CLOTH3D_PATH", os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data_generation", "cloth3d",
    ))
    os.environ.setdefault("FLEX_SHIM_PROPS", "0")
    os.environ[_ENV_READY] = "1"

    # 用相同的入口重启自身：LD_LIBRARY_PATH 只在进程启动时被动态链接器读取。
    # `python -m msufold` 与直接运行脚本两种入口都要保持原样
    import __main__

    if getattr(__main__, "__spec__", None) is not None:
        argv = [sys.executable, "-m", "msufold"] + sys.argv[1:]
    else:
        argv = [sys.executable] + sys.argv
    os.execv(sys.executable, argv)


_fix_runtime_env()
