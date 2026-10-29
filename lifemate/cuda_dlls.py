"""Make the pip-installed CUDA libraries (nvidia-cublas-cu12, nvidia-cudnn-cu12) visible to CTranslate2 on Windows.
Import this before faster_whisper."""
from __future__ import annotations

import os
import site
from pathlib import Path

for base in site.getsitepackages():
    for sub in ("nvidia/cublas/bin", "nvidia/cudnn/bin"):
        d = Path(base) / sub
        if d.is_dir():
            os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")

if os.name != "nt":  # Linux: preload the pip wheels' libraries so CTranslate2 finds them without LD_LIBRARY_PATH
    import ctypes
    import glob

    for base in site.getsitepackages():
        for pat in ("nvidia/cublas/lib/libcublas.so.12", "nvidia/cublas/lib/libcublasLt.so.12", "nvidia/cudnn/lib/libcudnn.so.9"):
            for lib in glob.glob(str(Path(base) / pat)):
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
