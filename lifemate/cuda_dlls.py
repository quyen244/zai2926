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
