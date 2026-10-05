"""屏蔽 PyFlex/C++ 侧直接打到 stdout 的调试日志。

pyflex.cpp 在读取深度缓冲时有一行
    printf("[dbg] depth_bits=%d fbo=%d glerr=%d\\n", ...)
它走的是 C 层的 fd 1，`contextlib.redirect_stdout` 拦不住。
这里用 dup2 在**文件描述符层面**把 fd 1 临时指向 /dev/null，
只影响 C/C++ 的 printf；tqdm 的进度条走 stderr，不受影响。
"""

import os
import sys
from contextlib import contextmanager


@contextmanager
def silence_c_stdout(streams=(1,)):
    """临时把指定的 fd（默认 stdout）指向 /dev/null。

    进入前会 flush Python 的 sys.stdout，退出后恢复，
    因此上下文之外的正常 print 不受影响。
    """
    saved = {}
    devnull_fd = None
    try:
        try:
            sys.stdout.flush()
        except Exception:
            pass

        devnull_fd = os.open(os.devnull, os.O_RDWR)
        for fd in streams:
            saved[fd] = os.dup(fd)
            os.dup2(devnull_fd, fd)
        yield
    finally:
        for fd, saved_fd in saved.items():
            os.dup2(saved_fd, fd)
            os.close(saved_fd)
        if devnull_fd is not None:
            os.close(devnull_fd)
