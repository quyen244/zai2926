FROM nvidia/cuda:13.2.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive TZ=Asia/Ho_Chi_Minh PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip libsndfile1 ffmpeg tzdata \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /code/requirements.txt
RUN pip3 install -r /code/requirements.txt
COPY code/ /code/
RUN chmod +x /code/predict.sh /code/start_jupyter.sh && python3 -m compileall -q /code/lifemate /code/runtime.py /code/predict.py
WORKDIR /code
CMD ["bash", "/code/predict.sh"]
