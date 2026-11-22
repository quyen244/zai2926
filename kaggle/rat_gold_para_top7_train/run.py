"""Kaggle RAT tool-only SFT on gold + gold paraphrases with E5 top-7 tool lists.

Continue the B2 adapter, using a fresh one-epoch run. The prompt tools are
filtered before packaging, exactly as inference does.
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
    data_dir = "work/sft_rat_toolonly_gold_para_top7"
    rows = [json.loads(l) for l in (REPO / data_dir / "traj.train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    val = [json.loads(l) for l in (REPO / data_dir / "traj.val.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rec = [r for r in rows if any(m.get("train") is False for m in r["messages"])]
    calls = [m for r in rows for m in r["messages"] if m.get("train") and m.get("tool_calls")]
    sel = [m for m in calls if "<select>" in (m.get("content") or "")]
    if any("<args>" in (m.get("content") or "") for r in rows for m in r["messages"] if m.get("train")):
        sys.exit("tool-only dataset contains parameter-reasoning spans")
    log(f"train rows {len(rows)}, recovery rows {len(rec)}, val rows {len(val)}, "
        f"trained calls with a rationale {len(sel)}/{len(calls)}")
    if len(rows) != 2690 or len(val) != 59 or rec or len(sel) != len(calls):
        sys.exit("wrong dataset version or missing tool-selection rationale")
    if any(r["variant"] not in {"gold", "para1", "para2"} for r in rows):
        sys.exit("unexpected training variant")
    for row in rows + val:
        offered = {t["function"]["name"] for t in row["tools"]}
        if any(c["function"]["name"] not in offered for m in row["messages"] if m.get("train") for c in m.get("tool_calls", [])):
            sys.exit(f"gold tool missing from filtered prompt: {row['id']}")
    adapter = next((p.parent for p in Path("/kaggle/input").rglob("out/traj_b2/adapter/adapter_model.safetensors")), None)
    if adapter is None:
        raise SystemExit("B2 adapter missing from b2-train kernel input")
    log("B2 adapter", adapter)
    install_train_stack()
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL)
    out = WORK / "out" / "traj_rat_gold_para_top7"
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": str(REPO)}
    cmd = [sys.executable, "-X", "utf8", "-m", "lifemate.sft_train", "--data", "traj", "--data-dir", data_dir,
           "--init-adapter", str(adapter), "--out", str(out), "--model", MODEL, "--max-len", "10240", "--lr", "5e-5",
           "--accum", "8", "--epochs", "1", "--rank", "16", "--evals", "4", "--max-hours", "10.5"]
    p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=open(out / "train.log", "w"), stderr=subprocess.STDOUT)
    log("started traj_rat on GPU 0:", " ".join(cmd[3:]))
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
