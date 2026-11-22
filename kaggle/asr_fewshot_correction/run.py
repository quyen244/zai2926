"""Correct fold-0 P2 transcripts with Qwen and fine-tuned E5 top-2/top-3 demos."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MODEL = "Qwen/Qwen3-4B"
SERVED_MODEL = "b2"
RETRIEVER = "intfloat/multilingual-e5-base"
WORK = Path("/kaggle/working")
OUT = WORK / "out"
PORTS = (8000, 8001)
NUM = {"không": "0", "một": "1", "hai": "2", "ba": "3", "bốn": "4", "năm": "5", "sáu": "6",
       "bảy": "7", "tám": "8", "chín": "9", "mười": "10"}


def log(*parts) -> None:
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def one_file(name: str) -> Path:
    paths = list(Path("/kaggle/input").rglob(name))
    if len(paths) != 1:
        raise RuntimeError(f"expected one {name} under /kaggle/input; found {paths}")
    return paths[0]


def retriever_dir() -> Path:
    paths = [p.parent for p in Path("/kaggle/input").rglob("model.safetensors")
             if "asr-gold-retriever-train" in str(p).lower() and p.parent.name == "retriever"]
    if len(paths) != 1:
        raise RuntimeError(f"expected one fine-tuned retriever checkpoint, found {paths}")
    return paths[0]


def b2_adapter_dir() -> Path:
    paths = [p.parent for p in Path("/kaggle/input").rglob("adapter_model.safetensors")
             if "lifemate-b2-train" in str(p).lower() and "traj_b2" in str(p).lower()]
    if len(paths) != 1:
        raise RuntimeError(f"expected one B2 adapter from lifemate-b2-train, found {paths}")
    return paths[0]


def norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text.lower())
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(NUM.get(word, word) for word in text.split())


def prompt(asr: str, examples: list[dict]) -> list[dict]:
    demonstration = "\n\n".join(
        f"Ví dụ {i}\nASR: {row['hyp']}\nBản đúng: {row['text']}" for i, row in enumerate(examples, 1))
    return [
        {"role": "system", "content": "Bạn sửa lỗi nhận dạng tiếng Việt trong một bản chép lời. "
         "Học kiểu lỗi từ các ví dụ, không chép tên riêng, con số hay nội dung của ví dụ vào câu mới. "
         "Chỉ sửa khi có căn cứ rõ trong câu ASR; giữ nguyên tên người, địa điểm, số tiền, số điện thoại và mã nếu không chắc. "
         "Không thêm thông tin. Chỉ xuất bản chép lời đã sửa, không giải thích."},
        {"role": "user", "content": f"{demonstration}\n\nCâu cần sửa\nASR: {asr}\nBản đúng:"},
    ]


def start_servers() -> list[subprocess.Popen]:
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL)
    adapter = b2_adapter_dir()
    log("using B2 adapter", adapter)
    procs = []
    for gpu, port in enumerate(PORTS):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
               "VLLM_ATTENTION_BACKEND": "TRITON_ATTN_VLLM_V1", "VLLM_CACHE_ROOT": f"/tmp/vllm_asr_gpu{gpu}"}
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL,
               "--port", str(port), "--max-model-len", "4096", "--gpu-memory-utilization", "0.88",
               "--dtype", "half", "--seed", "0", "--enable-lora", "--max-lora-rank", "16",
               "--max-loras", "1", "--lora-modules", f"{SERVED_MODEL}={adapter}"]
        procs.append(subprocess.Popen(cmd, env=env, stdout=open(WORK / f"vllm_gpu{gpu}.log", "w"),
                                      stderr=subprocess.STDOUT))
    for gpu, port in enumerate(PORTS):
        while True:
            if procs[gpu].poll() is not None:
                raise RuntimeError(f"vLLM GPU {gpu} exited: " + (WORK / f"vllm_gpu{gpu}.log").read_text(errors="ignore")[-3000:])
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5)
                break
            except OSError:
                time.sleep(5)
        log("vLLM ready", gpu, port)
    return procs


def complete(port: int, messages: list[dict]) -> str:
    body = {"model": SERVED_MODEL, "messages": messages, "temperature": 0, "max_tokens": 160,
            "chat_template_kwargs": {"enable_thinking": False}}
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                     data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                answer = json.load(response)["choices"][0]["message"].get("content") or ""
            break
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    answer = answer.strip().strip('"“”')
    answer = re.sub(r"^(?:Bản đúng|Bản sửa|Văn bản đã sửa)\s*:\s*", "", answer, flags=re.I)
    return answer.strip()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    real = read_jsonl(one_file("real.jsonl"))
    original = read_jsonl(one_file("baseline.jsonl"))
    train_asr = read_jsonl(one_file("train_p2.jsonl"))
    real_by_audio = {row["audio"]: row for row in real}
    fold0 = {r["audio"] for r in real if r["fold"] == 0}
    train_audio = {r["audio"] for r in real if r["fold"] != 0}
    if len(fold0) != 1009 or len(train_audio) != 4027:
        raise RuntimeError("real clip split changed")
    if {r["audio"] for r in original} != fold0 or {r["audio"] for r in train_asr} != train_audio:
        raise RuntimeError("P2 transcript coverage or split mismatch")
    train = [{**r, "text": real_by_audio[r["audio"]]["text"]} for r in train_asr]
    test = sorted(original, key=lambda r: r["audio"])
    checkpoint = retriever_dir()
    log("retrieval bank", len(train), "fold0", len(test), "retriever", checkpoint)

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm==0.10.2",
                    "jiwer==4.0.0", "transformers==4.57.1"], check=True)
    import jiwer
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    retriever = AutoModel.from_pretrained(checkpoint).cuda().eval()

    def embed(texts: list[str], prefix: str, batch_size: int = 64) -> np.ndarray:
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                encoded = tokenizer([prefix + text for text in texts[start:start + batch_size]],
                                    padding=True, truncation=True, max_length=128, return_tensors="pt").cuda()
                with torch.autocast("cuda", dtype=torch.float16):
                    hidden = retriever(**encoded).last_hidden_state
                    mask = encoded["attention_mask"].unsqueeze(-1)
                    pooled = (hidden * mask).sum(1) / mask.sum(1)
                vectors.append(F.normalize(pooled.float(), dim=-1).cpu().numpy())
        return np.concatenate(vectors)

    bank_vectors = embed([r["text"] for r in train], "passage: ")
    query_vectors = embed([r["hyp"] for r in test], "query: ")
    ranking = np.argsort(-(query_vectors @ bank_vectors.T), axis=1, kind="stable")[:, :3]
    del retriever, tokenizer, bank_vectors, query_vectors
    torch.cuda.empty_cache()
    log("E5 top-3 retrieval ready")

    procs = start_servers()
    path = OUT / "predictions.jsonl"
    done = {(r["audio"], int(r["k"])): r for r in read_jsonl(path)} if path.is_file() else {}
    lock = threading.Lock()

    def one(i: int, row: dict, k: int) -> dict:
        retrieved = [train[int(j)] for j in ranking[i, :k]]
        answer = complete(PORTS[i % 2], prompt(row["hyp"], retrieved))
        if not answer:
            raise RuntimeError(f"empty correction: {row['audio']} k={k}")
        return {"audio": row["audio"], "k": k, "asr": row["hyp"], "corrected": answer,
                "retrieved": [{"audio": r["audio"], "text": r["text"]} for r in retrieved]}

    futures = {}
    with ThreadPoolExecutor(max_workers=12) as pool:
        for i, row in enumerate(test):
            for k in (2, 3):
                if (row["audio"], k) not in done:
                    futures[pool.submit(one, i, row, k)] = (row["audio"], k)
        for future in as_completed(futures):
            result = future.result()
            key = result["audio"], result["k"]
            with lock:
                done[key] = result
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                if len(done) % 100 == 0:
                    log("correction variants", len(done), "/", len(test) * 2)

    expected = {(r["audio"], k) for r in test for k in (2, 3)}
    if set(done) != expected:
        raise RuntimeError(f"correction incomplete: {len(done)}/{len(expected)}")

    ref_by_audio = {r["audio"]: real_by_audio[r["audio"]]["text"] for r in test}
    words = 0
    baseline_edits = {2: 0, 3: 0}
    corrected_edits = {2: 0, 3: 0}
    exact_after = {2: 0, 3: 0}
    for row in test:
        ref = norm(ref_by_audio[row["audio"]])
        raw_result = jiwer.process_words(ref, norm(row["hyp"]))
        raw_edits = raw_result.substitutions + raw_result.deletions + raw_result.insertions
        words += len(ref.split())
        for k in (2, 3):
            corrected = norm(done[(row["audio"], k)]["corrected"])
            result = jiwer.process_words(ref, corrected)
            baseline_edits[k] += raw_edits
            corrected_edits[k] += result.substitutions + result.deletions + result.insertions
            exact_after[k] += result.wer == 0
    if words != 14531 or baseline_edits[2] != 2076 or baseline_edits[3] != 2076:
        raise RuntimeError("fold-0 baseline does not match audited P2 WER")
    metrics = {"clips": len(test), "reference_words": words, "baseline_edits": 2076,
               "baseline_wer": 2076 / words, "retriever": "fine-tuned multilingual-e5-base",
               "retriever_parameters": 278043648, "retrieval_bank": 4027, "fold0_overlap": 0,
               "variants": {str(k): {"corrected_edits": corrected_edits[k],
                                      "corrected_wer": corrected_edits[k] / words,
                                      "delta_wer": (corrected_edits[k] - 2076) / words,
                                      "relative_wer_change": (corrected_edits[k] - 2076) / 2076,
                                      "exact_clips": exact_after[k]} for k in (2, 3)},
               "correction_model": f"{MODEL} + B2 LoRA"}
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    log("metrics", json.dumps(metrics, ensure_ascii=False))
    for proc in procs:
        proc.terminate()


if __name__ == "__main__":
    main()
