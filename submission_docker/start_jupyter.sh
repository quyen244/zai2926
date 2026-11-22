#!/bin/bash
cd /code
export LD_LIBRARY_PATH="$(python3 -c 'import os,nvidia.cublas,nvidia.cudnn;print(":".join(os.path.join(m.__path__[0],"lib") for m in (nvidia.cublas,nvidia.cudnn)))'):${LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
jupyter lab --port 9777 --ip 0.0.0.0 --NotebookApp.password='zac2026' --NotebookApp.token='zac2026' --allow-root --no-browser
