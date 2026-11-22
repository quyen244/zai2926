#!/bin/bash
# Assemble the build context: build/code = what lands in /code inside the image.
set -e
P=/d/zaloai2026; B=$P/submission_docker/build; C=$B/code
rm -rf $B; mkdir -p $C/saved_models $C/training_code
cp $P/submission_docker/{Dockerfile,requirements.txt} $B/
cp $P/submission_docker/{runtime.py,predict.py,predict.sh,start_jupyter.sh,predict_notebook.ipynb,requirements.txt,README.md} $C/
cp -r $P/lifemate $C/lifemate; find $C/lifemate -name __pycache__ -prune -exec rm -rf {} +
cp -rl $P/work/models/qwen3-4b $C/saved_models/qwen3-4b; rm -rf $C/saved_models/qwen3-4b/.cache
cp -rl $P/work/kaggle/recovery_train/out/traj_rec/adapter $C/saved_models/rec_adapter
cp -rl $P/work/kaggle/asr_decode/out/phowhisper-ft-ct2 $C/saved_models/phowhisper-ft-ct2
cp -rl $P/work/kaggle/embed_train_q/out/app_embedder_e5 $C/saved_models/app_embedder_e5
cp -r $P/scripts $P/kaggle $P/prompts $C/training_code/
cp $P/submission_docker/TRAINING.md $C/training_code/README.md
cp -r $P/splits $C/training_code/
du -sh $C/saved_models/*
