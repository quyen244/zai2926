"""Kaggle job (quyenuit24): decode every real train clip with the fine-tuned PhoWhisper and measure it against the base model.

Inputs: dataset quyenuit24/lifemate-phowhisper-data (real.zip + real.jsonl {id, audio, text, fold}, 5036 clips: final and
history user turns of all train samples) + kernel output quyenuit24/lifemate-phowhisper-train-p2 (out/adapter, LoRA).

Steps:
 1. Merge the LoRA into vinai/PhoWhisper-large (fp32 on CPU) and convert both the merged and the base model to CTranslate2
    float16, the same engine and conversion as the local pipeline (scripts/asr_run.py).
 2. Gate: the first 100 fold-0 clips with both models (one per GPU). Stop if the new model is clearly worse
    (WER > base + 0.03) or returns more than 5 empty hypotheses.
 3. Full: the new model on all 5036 clips, the base model on all 1009 fold-0 clips, each list split over the 2 GPUs.
Decoding = scripts/asr_run.py: language vi, beam 5, no previous-text conditioning, float16; a flagged hypothesis (empty,
filler phrase, compression ratio > 2.4) is re-decoded once with peak normalisation + VAD.

Outputs (/kaggle/working/out): new.jsonl (all clips, asr_run.py format, keyed by audio path), base_fold0.jsonl,
stats.json (quick WER per fold), phowhisper-ft-ct2/ (CTranslate2 model for the Docker image).
Final scoring is done locally (scripts/asr_compare.py).
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import zipfile
from pathlib import Path

WORK = Path("/kaggle/working")
OUT = WORK / "out"
TMP = Path("/tmp/asr")
BASE_ID = "vinai/PhoWhisper-large"
COPY = ["preprocessor_config.json", "tokenizer_config.json", "vocab.json", "merges.txt", "normalizer.json",
        "special_tokens_map.json", "added_tokens.json"]
HALLU = re.compile(r"subscribe|đăng k[ýí]|ghiền mì gõ|cảm ơn các bạn đã (theo dõi|xem)|hẹn gặp lại|like|share|la la school", re.I)
NUM = {"không": "0", "một": "1", "hai": "2", "ba": "3", "bốn": "4", "năm": "5", "sáu": "6", "bảy": "7", "tám": "8", "chín": "9", "mười": "10"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------------------------------------------------------------- worker (one process per GPU)

def flagged(hyp: str, comp: float) -> bool:
    return not hyp.strip() or bool(HALLU.search(hyp)) or comp > 2.4


def peak_norm(x, db: float = -3.0):
    p = float(abs(x).max())
    return x * (10 ** (db / 20) / p) if p > 0 else x


def decode(fn, audio, **kw):
    segs, _ = fn(audio, language="vi", beam_size=5, condition_on_previous_text=False, **kw)
    segs = list(segs)
    return " ".join(s.text.strip() for s in segs).strip(), {
        "avg_logprob": round(min((s.avg_logprob for s in segs), default=0.0), 3),
        "no_speech_prob": round(max((s.no_speech_prob for s in segs), default=0.0), 3),
        "compression_ratio": round(max((s.compression_ratio for s in segs), default=0.0), 3)}


def worker(gpu: int, tasks_file: str) -> None:
    import soundfile as sf
    from faster_whisper import WhisperModel
    for task in json.loads(Path(tasks_file).read_text(encoding="utf-8")):
        out = Path(task["out"])
        done = {r["audio"] for r in jsonl(out)} if out.is_file() else set()
        todo = [c for c in task["clips"] if c not in done]
        if not todo:
            continue
        model = WhisperModel(task["model"], device="cuda", device_index=gpu, compute_type="float16")
        t_all, audio_sec = time.time(), 0.0
        with open(out, "a", encoding="utf-8") as f:
            for k, rel in enumerate(todo, 1):
                x, sr = sf.read(str(Path(task["base"]) / rel), dtype="float32")
                if x.ndim > 1:
                    x = x.mean(1)
                if sr != 16000:
                    raise SystemExit(f"{rel}: sample rate {sr}, expected 16000")
                t = time.time()
                hyp, m = decode(model.transcribe, x)
                redo = False
                if flagged(hyp, m["compression_ratio"]):
                    h2, m2 = decode(model.transcribe, peak_norm(x), vad_filter=True)
                    if h2 and not flagged(h2, m2["compression_ratio"]):
                        hyp, m, redo = h2, m2, True
                f.write(json.dumps({"audio": rel, "hyp": hyp, "redo": redo, "sec_audio": round(len(x) / sr, 2),
                                    "sec_decode": round(time.time() - t, 2), **m}, ensure_ascii=False) + "\n")
                f.flush()
                audio_sec += len(x) / sr
                if k % 100 == 0:
                    print(f"gpu{gpu} {task['name']}: {k}/{len(todo)}  RTF {(time.time() - t_all) / audio_sec:.3f}", flush=True)
        print(f"gpu{gpu} {task['name']}: {len(todo)} clips, {audio_sec:.0f}s audio in {time.time() - t_all:.0f}s", flush=True)
        del model


# ---------------------------------------------------------------- driver

def base_for(data: Path, tmp: Path) -> Path:
    """Directory under which real.jsonl's relative audio paths exist (Kaggle may keep real.zip or unpack it itself)."""
    if (data / "real.zip").is_file():
        tmp.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(data / "real.zip") as zf:
            zf.extractall(tmp)
        return tmp
    rel = jsonl(data / "real.jsonl")[0]["audio"]
    cands = [data] + [d for d in data.iterdir() if d.is_dir()] + [e for d in data.iterdir() if d.is_dir() for e in d.iterdir() if e.is_dir()]
    hit = next((d for d in cands if (d / rel).is_file()), None)
    if hit is None:
        raise SystemExit(f"{rel} not found under {data}")
    return hit


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s.lower())
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(NUM.get(w, w) for w in s.split())


def wer(rows: list[dict], ref: dict) -> dict:
    import jiwer
    edits = words = empty = 0
    for r in rows:
        rw = norm(ref[r["audio"]])
        n = len(rw.split())
        h = norm(r["hyp"])
        edits += (jiwer.wer(rw, h) if h else 1.0) * n
        words += n
        empty += not r["hyp"].strip()
    return {"clips": len(rows), "wer": round(edits / max(words, 1), 4), "empty": empty}


def build_models(adapter: Path) -> tuple[Path, Path]:
    import torch
    from ctranslate2.converters import TransformersConverter
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration
    base_hf = Path(snapshot_download(BASE_ID, local_dir=str(TMP / "base_hf")))
    log("base downloaded", base_hf)
    model = WhisperForConditionalGeneration.from_pretrained(str(base_hf), torch_dtype=torch.float32)
    ref_w = model.model.decoder.layers[0].self_attn.q_proj.weight.detach().clone()
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    delta = (model.model.decoder.layers[0].self_attn.q_proj.weight - ref_w).abs().mean().item()
    log(f"LoRA merged, mean |delta| decoder.0.q_proj = {delta:.2e}")
    if delta == 0:
        raise SystemExit("merge changed nothing: adapter not applied")
    merged = TMP / "merged_hf"
    model.save_pretrained(str(merged), safe_serialization=True)
    for f in COPY:
        if (base_hf / f).is_file():
            shutil.copy(base_hf / f, merged / f)
    del model
    ct2_base, ct2_new = TMP / "ct2_base", OUT / "phowhisper-ft-ct2"
    for src, dst in ((base_hf, ct2_base), (merged, ct2_new)):
        TransformersConverter(str(src), copy_files=[f for f in COPY if (src / f).is_file()]).convert(str(dst), quantization="float16", force=True)
        log("converted", src.name, "->", dst, f"{sum(p.stat().st_size for p in dst.rglob('*')) / 1e9:.2f} GB")
    return ct2_base, ct2_new


def run_workers(per_gpu: list[list[dict]], tag: str) -> None:
    env = dict(os.environ)
    import nvidia.cublas, nvidia.cudnn  # noqa: E401  (ctranslate2 needs libcublas / libcudnn 9 on the path)
    libs = [str(Path(m.__path__[0]) / "lib") for m in (nvidia.cublas, nvidia.cudnn)]
    env["LD_LIBRARY_PATH"] = ":".join(libs + [env.get("LD_LIBRARY_PATH", "")])
    procs = []
    for gpu, tasks in enumerate(per_gpu):
        tf = TMP / f"tasks_{tag}_{gpu}.json"
        tf.write_text(json.dumps(tasks), encoding="utf-8")
        procs.append(subprocess.Popen([sys.executable, __file__, "worker", str(gpu), str(tf)], env=env))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"{tag}: worker exit codes {codes}")


def merge_shards(name: str, order: list[str]) -> list[dict]:
    rows = {}
    for p in sorted(TMP.glob(f"{name}.gpu*.jsonl")):
        rows.update({r["audio"]: r for r in jsonl(p)})
    out = [rows[c] for c in order if c in rows]
    (OUT / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    return out


def main():
    t0 = time.time()
    inp = Path("/kaggle/input")
    data = next(p.parent for p in inp.rglob("real.jsonl"))
    adapter = next(p.parent for p in inp.rglob("adapter_config.json"))
    log("dataset", data, "| adapter", adapter)
    TMP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    real = jsonl(data / "real.jsonl")
    base = base_for(data, TMP / "real")
    miss = [r["audio"] for r in real if not (base / r["audio"]).is_file()]
    log(f"real.jsonl {len(real)} clips, audio base {base}, missing {len(miss)}")
    if miss:
        raise SystemExit(f"{len(miss)} clips missing, e.g. {miss[:3]}")
    subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.1", "peft==0.17.1", "ctranslate2==4.8.4",
                    "faster-whisper==1.2.1", "jiwer==4.0.0", "soundfile"], check=True)
    ct2_base, ct2_new = build_models(adapter)
    log(f"models ready at {time.time() - t0:.0f}s")

    ref = {r["audio"]: r["text"] for r in real}
    fold = {r["audio"]: r["fold"] for r in real}
    allc = sorted(ref)
    f0 = [c for c in allc if fold[c] == 0]
    gate = f0[:100]
    new_t = lambda clips, g: {"name": "new", "model": str(ct2_new), "base": str(base), "out": str(TMP / f"new.gpu{g}.jsonl"), "clips": clips}
    base_t = lambda clips, g: {"name": "base_fold0", "model": str(ct2_base), "base": str(base), "out": str(TMP / f"base_fold0.gpu{g}.jsonl"), "clips": clips}

    # 2. gate: same 100 fold-0 clips, new model on GPU 0, base on GPU 1
    run_workers([[new_t(gate, 0)], [base_t(gate, 1)]], "gate")
    g_new = wer(jsonl(TMP / "new.gpu0.jsonl"), ref)
    g_base = wer(jsonl(TMP / "base_fold0.gpu1.jsonl"), ref)
    log("GATE new", g_new, "| base", g_base)
    stats = {"gate": {"new": g_new, "base": g_base}}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    if g_new["wer"] > g_base["wer"] + 0.03 or g_new["empty"] > 5:
        raise SystemExit("gate failed: new model clearly worse than base on 100 fold-0 clips, stopping before the full run")

    # 3. full run; each GPU continues its own shard files (gate clips already done are skipped)
    rest_new = [c for c in allc if c not in set(gate)]
    rest_base = [c for c in f0 if c not in set(gate)]
    run_workers([[new_t(rest_new[0::2], 0), base_t(rest_base[0::2], 0)],
                 [new_t(rest_new[1::2], 1), base_t(rest_base[1::2], 1)]], "full")
    new_rows, base_rows = merge_shards("new", allc), merge_shards("base_fold0", f0)
    if len(new_rows) != len(allc) or len(base_rows) != len(f0):
        raise SystemExit(f"incomplete: new {len(new_rows)}/{len(allc)}, base {len(base_rows)}/{len(f0)}")
    stats["fold0"] = {"new": wer([r for r in new_rows if fold[r["audio"]] == 0], ref), "base": wer(base_rows, ref)}
    stats["folds1_4"] = {"new": wer([r for r in new_rows if fold[r["audio"]] != 0], ref)}
    stats["decode_sec_per_clip"] = {"new": round(sum(r["sec_decode"] for r in new_rows) / len(new_rows), 3),
                                    "base": round(sum(r["sec_decode"] for r in base_rows) / len(base_rows), 3)}
    stats["redo"] = {"new": sum(r["redo"] for r in new_rows), "base_fold0": sum(r["redo"] for r in base_rows)}
    stats["seconds"] = round(time.time() - t0)
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log("STATS", json.dumps(stats))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker(int(sys.argv[2]), sys.argv[3])
    else:
        main()
