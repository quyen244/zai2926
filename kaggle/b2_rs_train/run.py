"""Train two B2 RS ablations on T4 x2, then evaluate both on the same 309-sample test fold."""
from __future__ import annotations

import json
import os
import site
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODEL = "Qwen/Qwen3-4B"
WORK = Path("/kaggle/working")
REPO = Path("/tmp/rs_ablation_repo")
OUT = WORK / "out"
STEP_CAP = 6
WORKERS_PER_GPU = 6
PORTS = [8000, 8001]
SYSTEMS = {"rs_only": (0, "rs_only"), "rs_plus_v1": (1, "rs_plus_v1")}


def log(*args) -> None:
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def unpack() -> None:
    src = next((p for p in Path("/kaggle/input").rglob("repo.tar.gz")), None)
    if src:
        with tarfile.open(src) as tar:
            tar.extractall(REPO)
    else:  # Kaggle extracts uploaded .tar.gz archives into the input directory tree.
        trainer = next((p for p in Path("/kaggle/input").rglob("lifemate/sft_train.py")), None)
        if not trainer:
            raise SystemExit("trainer files missing from Kaggle input")
        shutil.copytree(trainer.parent.parent, REPO, dirs_exist_ok=True)
    log("repo at", REPO)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_inputs() -> None:
    expected = {"rs_only": 643, "rs_plus_v1": 2995}
    for name, count in expected.items():
        train_file = REPO / f"work/rs_sft/{name}.train.jsonl"
        val_file = REPO / f"work/rs_sft/{name}.val.jsonl"
        train = read_jsonl(train_file)
        val = read_jsonl(val_file)
        if len(train) != count or len(val) != 60:
            raise SystemExit(f"{name}: expected train={count}, val=60; got train={len(train)}, val={len(val)}")
        trainer_dir = REPO / "work/rs_sft" / name
        trainer_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(train_file, trainer_dir / "traj.train.jsonl")
        shutil.copy2(val_file, trainer_dir / "traj.val.jsonl")
    if len(read_jsonl(REPO / "work/sft/plan.test.jsonl")) != 309:
        raise SystemExit("expected 309 test-fold rows")
    if not (REPO / "adapters/traj_b2/adapter_model.safetensors").is_file():
        raise SystemExit("B2 initial adapter missing")
    devices = subprocess.run(["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
                             check=True, capture_output=True, text=True).stdout.strip().splitlines()
    if len(devices) < 2 or any("T4" not in row for row in devices[:2]):
        raise SystemExit(f"expected two visible T4 GPUs, got {devices}")
    log("GPU check:", devices[:2])
    log("training sets: rs_only=643; rs_plus_v1=643 RS + 2352 v1=2995; one epoch each")


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


def train_both() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL)
    procs: dict[str, subprocess.Popen] = {}
    files = {}
    for name, (gpu, data_name) in SYSTEMS.items():
        run_out = OUT / name
        run_out.mkdir(parents=True, exist_ok=True)
        files[name] = open(run_out / "train.log", "w", encoding="utf-8")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(REPO)}
        cmd = [sys.executable, "-X", "utf8", "-m", "lifemate.sft_train", "--data", "traj",
               "--data-dir", f"work/rs_sft/{data_name}", "--init-adapter", str(REPO / "adapters/traj_b2"),
               "--out", str(run_out), "--model", MODEL, "--max-len", "10240", "--lr", "1e-5",
               "--accum", "8", "--epochs", "1", "--rank", "16", "--evals", "4", "--max-hours", "10"]
        procs[name] = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=files[name], stderr=subprocess.STDOUT)
        log(f"started {name} on physical GPU {gpu}, one epoch, effective batch 8")

    t0 = time.time()
    while any(proc.poll() is None for proc in procs.values()):
        failed = [name for name, proc in procs.items() if proc.poll() not in (None, 0)]
        if failed:
            for name, proc in procs.items():
                if proc.poll() is None:
                    proc.terminate()
            for name, fp in files.items():
                fp.close()
            for name in failed:
                tail = (OUT / name / "train.log").read_text(errors="ignore").splitlines()[-60:]
                print(f"--- {name} training failure ---\n" + "\n".join(tail))
            raise SystemExit(f"training failed for {failed}")
        time.sleep(60)
        for name in procs:
            tail = (OUT / name / "train.log").read_text(errors="ignore").strip().splitlines()[-1:]
            log(name, f"{time.time()-t0:.0f}s", tail[0][-140:] if tail else "")
    for fp in files.values():
        fp.close()
    for name, proc in procs.items():
        if proc.returncode:
            raise SystemExit(f"{name} exited {proc.returncode}")
        stats = json.loads((OUT / name / "stats.json").read_text(encoding="utf-8"))
        log(name, "trained", json.dumps(stats, ensure_ascii=False))


def install_eval_stack() -> None:
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    for cmd in (pip + ["vllm==0.10.2"], pip + ["transformers==4.57.1"]):
        subprocess.run(cmd, check=True)


def start_eval_servers() -> list[subprocess.Popen]:
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL)
    procs = []
    for name, (gpu, _) in SYSTEMS.items():
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "VLLM_ATTENTION_BACKEND": "TRITON_ATTN_VLLM_V1",
               "VLLM_CACHE_ROOT": f"/tmp/vllm_cache_gpu{gpu}"}
        adapter = OUT / name / "adapter"
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL,
               "--port", str(PORTS[gpu]), "--max-model-len", "12288", "--gpu-memory-utilization", "0.88",
               "--dtype", "half", "--seed", "0", "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
               "--enable-lora", "--max-lora-rank", "16", "--max-loras", "1", "--lora-modules", f"{name}={adapter}"]
        procs.append(subprocess.Popen(cmd, env=env, stdout=open(WORK / f"vllm_gpu{gpu}.log", "w"), stderr=subprocess.STDOUT))
    t0 = time.time()
    for gpu, port in enumerate(PORTS):
        while True:
            proc = procs[gpu]
            if proc.poll() is not None:
                raise SystemExit(f"vLLM GPU {gpu} died:\n" + (WORK / f"vllm_gpu{gpu}.log").read_text()[-4000:])
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5)
                break
            except OSError:
                time.sleep(5)
        log(f"eval server GPU {gpu} ready in {time.time()-t0:.0f}s")
    return procs


def check_template(tok, template: str, name: str, port: int, row: dict) -> None:
    from lifemate.llm import tools_body
    from lifemate.sft import template_messages

    trained = [i for i, msg in enumerate(row["messages"]) if msg.get("train")]
    if len(trained) < 3:
        return
    j = trained[2]
    wire = [{k: v for k, v in m.items() if k != "train"} for m in row["messages"][:j]]
    local = template_messages(row)[:j + 1]
    counts = {}
    for label, chat_template in (("native", None), ("trained", template)):
        rendered = tok.apply_chat_template(local, tools=row["tools"], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False, **({"chat_template": chat_template} if chat_template else {}))
        want = len(tok(rendered, add_special_tokens=False).input_ids)
        body = tools_body(name, False, row["system"], wire, row["tools"], 1, 0, chat_template)
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        got = json.load(urllib.request.urlopen(req, timeout=300))["usage"]["prompt_tokens"]
        counts[label] = (got, want)
    log(f"template check {name} :{port}: {counts}")
    if any(got != want for got, want in counts.values()) or counts["native"][0] == counts["trained"][0]:
        raise SystemExit(f"template mismatch for {name}: {counts}")


def evaluate_both() -> None:
    import urllib.request

    from transformers import AutoTokenizer
    from lifemate.agent import run_sample
    from lifemate.data import fuzzy_params, load_samples
    from lifemate.gold import replay
    from lifemate.llm import LLM
    from lifemate.scoring import score_split
    from lifemate.sft import history_think_template

    test_rows = read_jsonl(REPO / "work/sft/plan.test.jsonl")
    by_id = {row["id"]: row for row in load_samples("train")}
    samples = [by_id[row["id"]] for row in sorted(test_rows, key=lambda row: row["id"])]
    if len(samples) != 309:
        raise SystemExit(f"expected 309 test-fold samples, got {len(samples)}")
    gold = {}
    for sample in samples:
        lab = sample["label"]
        env, results = replay("train", sample["id"], lab["tool_calls"])
        if results != lab["tool_results"] or not all(env.get_state().get(k) == v for k, v in lab["final_state"].items()):
            raise SystemExit(f"gold replay mismatch: {sample['id']}")
        gold[sample["id"]] = {"tool_calls": lab["tool_calls"], "state_hash": env.get_state_hash()}

    tok = AutoTokenizer.from_pretrained(MODEL)
    template = history_think_template(tok.chat_template)
    probe = next(row for row in test_rows if sum(bool(m.get("train")) for m in row["messages"]) >= 3)
    for name, (gpu, _) in SYSTEMS.items():
        check_template(tok, template, name, PORTS[gpu], probe)

    llms = {name: LLM(model=name, base_url=f"http://127.0.0.1:{PORTS[gpu]}/v1", timeout=900,
                      chat_template=template) for name, (gpu, _) in SYSTEMS.items()}
    results = {name: {} for name in SYSTEMS}
    locks = {name: threading.Lock() for name in SYSTEMS}
    failures = []
    t_eval = time.time()

    def one(name: str, sample: dict) -> None:
        replies = []
        t0 = time.time()
        try:
            pred = run_sample(sample, "train", sample["audio_text"], llms[name], None, STEP_CAP,
                              trace=lambda event, data: event == "llm_reply" and replies.append(data["reply"]))
            pred.update(sec=round(time.time() - t0, 1), replies=replies)
            with locks[name]:
                results[name][sample["id"]] = pred
                with open(OUT / "eval" / f"{name}.jsonl", "a", encoding="utf-8") as fp:
                    fp.write(json.dumps(pred, ensure_ascii=False) + "\n")
        except Exception as exc:
            with locks[name]:
                failures.append((name, sample["id"], type(exc).__name__, str(exc)[:250]))
            log("FAILED", name, sample["id"], type(exc).__name__, str(exc)[:180])

    def eval_system(name: str, gpu: int) -> None:
        llm = llms[name]
        with ThreadPoolExecutor(WORKERS_PER_GPU) as pool:
            list(pool.map(lambda sample: one(name, sample), samples))
        log(f"{name} eval done on GPU {gpu}: {len(results[name])}/{len(samples)}")

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda item: eval_system(item[0], item[1][0]), SYSTEMS.items()))
    if failures or any(len(rows) != 309 for rows in results.values()):
        (OUT / "eval" / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(f"evaluation incomplete: failures={len(failures)}")

    fz = fuzzy_params("train")
    report = {"model": MODEL, "test_fold": 0, "samples": len(samples), "context": "full_tools",
              "step_cap": STEP_CAP, "eval_s": round(time.time() - t_eval), "runs": {}}
    for name, preds in results.items():
        score, per = score_split({sid: {"tool_calls": row["tool_calls"], "state_hash": row["state_hash"]}
                                  for sid, row in preds.items()}, gold, fz)
        report["runs"][name] = {"n": len(preds), "score": round(score, 4),
                                "hash_ok": sum(row.hash_ok for row in per.values()),
                                "mean_sec": round(sum(row["sec"] for row in preds.values()) / len(preds), 1)}
        log(f"{name}: score={score:.4f}, hash_ok={report['runs'][name]['hash_ok']}/309")
    report["score_delta_rs_plus_v1_minus_rs_only"] = round(report["runs"]["rs_plus_v1"]["score"] - report["runs"]["rs_only"]["score"], 4)
    (OUT / "eval" / "scores.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log("ablation scores", json.dumps(report, ensure_ascii=False))


def main() -> None:
    started = time.time()
    unpack()
    check_inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    install_train_stack()
    train_both()
    install_eval_stack()
    (OUT / "eval").mkdir(parents=True, exist_ok=True)
    servers = start_eval_servers()
    try:
        evaluate_both()
    finally:
        for proc in servers:
            proc.terminate()
    log(f"complete in {time.time()-started:.0f}s")
    for name in SYSTEMS:
        log("---", name, "train tail ---")
        print("\n".join((OUT / name / "train.log").read_text(errors="ignore").splitlines()[-20:]))
        log("---", name, "stats ---")
        print((OUT / name / "stats.json").read_text(encoding="utf-8"))
    log("--- evaluation scores ---")
    print((OUT / "eval/scores.json").read_text(encoding="utf-8"))
    shutil.rmtree(REPO, ignore_errors=True)


if __name__ == "__main__":
    main()
