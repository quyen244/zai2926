"""Kaggle job (account quyenuit24): DPO of the tool-only RAT adapter (+ SFT term on the chosen step).

Flow: unpack quyenuit24/lifemate-dpo-data (kaggle/dpo_train/pack.py: lifemate/ code + work/dpo/rat.{train,val}.jsonl, pairs from
tool-only RAT greedy rollouts on train folds 1-4, gold and ASR text, no app filter) -> RAT adapter from lifemate-rat-toolonly-train
(out/traj_rat_toolonly/adapter) -> install the training stack exactly as kaggle/recovery_train (unsloth) -> GPU 0:
lifemate.dpo_train beta 0.1, alpha 1, lr 1e-5, accum 8, 2 epochs, r16, max length 10240
-> /kaggle/working/out/traj_dpo_rat_toolonly/{adapter/, log.jsonl, stats.json, train.log}.
Full run: beta 0.1, chosen-step NLL alpha 1, lr 5e-6, 1 epoch, rank 16, accum 8.
"""
import json
import os
import shutil
import site
import subprocess
import sys
import tarfile
import time
from pathlib import Path

MODEL = "Qwen/Qwen3-4B"
WORK = Path("/kaggle/working")
REPO = Path("/tmp/repo")  # outside /kaggle/working: never part of the kernel output


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def install_train_stack() -> None:
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    commands = (
        pip + ["torch", "torchvision", "torchaudio", "xformers", "--index-url", "https://download.pytorch.org/whl/cu128"],
        pip + ["unsloth"],
        pip + ["--no-deps", "--upgrade", "torchao>=0.16.0"],
        pip + ["transformers==4.57.6"],
        pip + ["--no-deps", "trl==0.22.2"],
        pip + ["protobuf==3.20.3"],
    )
    for cmd in commands:
        subprocess.run(cmd, check=False)

    # Kaggle's current Python 3.13 image keeps CUDA wheel libraries under nvidia/*/lib, which
    # are not on the dynamic loader path by default. torchao's mslk extension needs libnvrtc.so.12.
    cuda_libs = set()
    for root in site.getsitepackages():
        cuda_libs.update(str(path.parent) for path in Path(root).glob("nvidia/*/lib/libnvrtc.so*"))
        cuda_libs.update(str(path) for path in Path(root).glob("nvidia/*/lib"))
        torch_lib = Path(root) / "torch/lib"
        if torch_lib.is_dir():
            cuda_libs.add(str(torch_lib))
    old_ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join([*sorted(cuda_libs), old_ld] if old_ld else sorted(cuda_libs))
    log("CUDA wheel library directories:", sorted(cuda_libs))

    check = subprocess.run([sys.executable, "-c", (
        "import unsloth; import torch, bitsandbytes as bnb, xformers, transformers, peft;"
        "print('torch',torch.__version__,'cuda',torch.version.cuda,'bnb',bnb.__version__,'transformers',transformers.__version__,"
        "'unsloth',unsloth.__version__,'peft',peft.__version__);"
        "w=torch.randn(64,64,dtype=torch.float16,device='cuda');q,s=bnb.functional.quantize_4bit(w,quant_type='nf4');"
        "print('nf4 ok',bnb.functional.dequantize_4bit(q,s).shape)")], check=False, env=os.environ.copy())
    if check.returncode:
        raise SystemExit("training environment check failed")


def main() -> None:
    REPO.mkdir(parents=True, exist_ok=True)
    src = next((p for p in Path("/kaggle/input").rglob("repo.tar.gz") if "rec-rollouts-e5-input" in str(p)), None)
    if not src:
        raise SystemExit("missing lifemate-rec-rollouts-e5-input/repo.tar.gz")
    with tarfile.open(src) as tar:
        tar.extractall(REPO)
    rat_src = next((p for p in Path("/kaggle/input").rglob("repo.tar.gz") if "rat-toolonly-data" in str(p)), None)
    if not rat_src:
        raise SystemExit("missing lifemate-rat-toolonly-data/repo.tar.gz")
    with tarfile.open(rat_src) as tar:
        tar.extractall(REPO)

    install_train_stack()
    rollout_dir = next((p for p in Path("/kaggle/input").rglob("cache") if "rat-toolonly-rollouts" in str(p) and p.is_dir()), None)
    if not rollout_dir:
        raise SystemExit("missing completed lifemate-rat-toolonly-rollouts output")
    rollouts = {}
    for variant in ("gold", "asr_ft"):
        candidates = sorted(rollout_dir.glob(f"Qwen3-4B-trajRAT-toolonly.{variant}.sc6.idguard.nofilter.*.jsonl"))
        if not candidates:
            raise SystemExit(f"missing tool-only {variant} rollout cache in {rollout_dir}")
        rollouts[variant] = candidates[-1]
    subprocess.run([sys.executable, "scripts/dpo_data.py", "--rollouts",
                    f"{rollouts['gold']}:gold", f"{rollouts['asr_ft']}:asr_ft",
                    "--normalise", "false", "--tag", "rat"], cwd=REPO, check=True)
    n = {part: sum(1 for l in (REPO / f"work/dpo/rat.{part}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()) for part in ("train", "val")}
    if not n["train"] or not n["val"]:
        raise SystemExit(f"fresh tool-only DPO data is empty: {n}")
    log("fresh tool-only pairs", n)

    adapter = next((p.parent for p in Path("/kaggle/input").rglob("adapter_config.json")
                    if "rat-toolonly-data" in str(p) and (p.parent / "adapter_model.safetensors").is_file()), None)
    if not adapter:
        raise SystemExit("tool-only RAT adapter not found in lifemate-rat-toolonly-data")
    log("tool-only RAT adapter", adapter, sorted(x.name for x in adapter.iterdir()))
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL)
    out = WORK / "out" / "traj_dpo_rat_toolonly"
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": str(REPO)}
    cmd = [sys.executable, "-X", "utf8", "-m", "lifemate.dpo_train", "--init-adapter", str(adapter), "--out", str(out),
           "--data-dir", "work/dpo", "--tag", "rat", "--model", MODEL, "--max-len", "10240", "--beta", "0.1",
           "--alpha", "1.0", "--lr", "5e-6", "--accum", "8", "--epochs", "1", "--rank", "16", "--max-hours", "10.3"]
    p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=open(out / "train.log", "w"), stderr=subprocess.STDOUT)
    log("started:", " ".join(cmd[3:]))
    t0 = time.time()
    while p.poll() is None:
        time.sleep(60)
        tail = (out / "train.log").read_text(errors="ignore").strip().splitlines()[-1:]
        log(f"{time.time() - t0:.0f}s", tail[0][-200:] if tail else "")
    log("exit code", p.returncode)
    print("--- train.log tail ---", *(out / "train.log").read_text(errors="ignore").splitlines()[-30:], sep="\n")
    shutil.rmtree(REPO, ignore_errors=True)
    sys.exit(p.returncode)


if __name__ == "__main__":
    main()
