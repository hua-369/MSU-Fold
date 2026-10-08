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
    # - 脚本方式（argv[0] 是文件路径）原样重启；
    # - `python -m msufold` / torchrun 方式用 -m 重启；
    # - 无法判断入口时给出提示并继续，绝不拼出错误的命令行。
    import __main__

    spec = getattr(__main__, "__spec__", None)
    if sys.argv and os.path.isfile(sys.argv[0]):
        argv = [sys.executable] + sys.argv
    elif spec is not None and spec.name.endswith(".__main__"):
        argv = [sys.executable, "-m", spec.name[: -len(".__main__")]] + sys.argv[1:]
    else:
        print(
            "[msufold] 无法安全重启以补齐 LD_LIBRARY_PATH，请在 shell 中先执行："
            "export LD_LIBRARY_PATH=$PYFLEXROOT/external/SDL2-2.0.4/lib/x64:$CUDART9_PATH:$LD_LIBRARY_PATH"
        )
        return
    os.execv(sys.executable, argv)


_fix_runtime_env()
