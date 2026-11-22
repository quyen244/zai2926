"""Kaggle job (account tquynlnguyn): recovery SFT = continue B2 on its 2115 rows + 300 ID-guard recovery rows.

Flow: unpack tquynlnguyn/lifemate-recovery-data (kaggle/recovery_train/pack.py: lifemate/ code with the per-message loss
mask, work/sft_rec/traj.{train,val}.jsonl) -> check the rows (2415 train, 300 with one untrained blocked call, 60 val) ->
B2 adapter from kernel tquynlnguyn/lifemate-b2-train (out/traj_b2/adapter) -> install the training stack as
kaggle/b2_rs_train/ -> GPU 0: lifemate.sft_train --init-adapter <B2> lr 1e-5, 1 epoch, accum 8, r16, max length 10240,
val loss 4 times -> /kaggle/working/out/traj_rec/{adapter/, log.jsonl, stats.json, train.log}.
Then kaggle/recovery_eval/ serves out/traj_rec/adapter exactly as the ID-guard eval served B2 (63.01).
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
    src = next((p for p in Path("/kaggle/input").rglob("repo.tar.gz")), None)
    if src:
        with tarfile.open(src) as tar:
            tar.extractall(REPO)
    else:  # Kaggle unpacks an uploaded tar: the dataset holds lifemate/ and work/ directly
        d = next(p.parent.parent for p in Path("/kaggle/input").rglob("lifemate/sft_train.py"))
        shutil.copytree(d, REPO, dirs_exist_ok=True)
    rows = [json.loads(l) for l in (REPO / "work/sft_rec/traj.train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    val = (REPO / "work/sft_rec/traj.val.jsonl").read_text(encoding="utf-8").splitlines()
    rec = [r for r in rows if any(m.get("train") is False for m in r["messages"])]
    log(f"train rows {len(rows)}, recovery rows {len(rec)}, val rows {len([l for l in val if l.strip()])}")
    if len(rows) != 2415 or len(rec) != 300:
        sys.exit("wrong dataset version (expected 2115 B2 + 300 recovery rows)")
    adapter = next(p.parent for p in Path("/kaggle/input").rglob("out/traj_b2/adapter/adapter_model.safetensors"))
    log("B2 adapter", adapter)
    install_train_stack()
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL)
    out = WORK / "out" / "traj_rec"
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": str(REPO)}
    cmd = [sys.executable, "-X", "utf8", "-m", "lifemate.sft_train", "--data", "traj", "--data-dir", "work/sft_rec",
           "--init-adapter", str(adapter), "--out", str(out), "--model", MODEL, "--max-len", "10240", "--lr", "1e-5",
           "--accum", "8", "--epochs", "1", "--rank", "16", "--evals", "4", "--max-hours", "10.8"]
    p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=open(out / "train.log", "w"), stderr=subprocess.STDOUT)
    log("started traj_rec on GPU 0:", " ".join(cmd[3:]))
    t0 = time.time()
    while p.poll() is None:
        time.sleep(60)
        tail = (out / "train.log").read_text(errors="ignore").strip().splitlines()[-1:]
        log(f"{time.time() - t0:.0f}s", tail[0][-160:] if tail else "")
    log("exit code", p.returncode)
    shutil.rmtree(out / "ckpt", ignore_errors=True)
    print("--- train.log tail ---\n" + "\n".join((out / "train.log").read_text(errors="ignore").splitlines()[-25:]))
    shutil.rmtree(REPO, ignore_errors=True)
    sys.exit(p.returncode)


if __name__ == "__main__":
    main()
