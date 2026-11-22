"""Evaluate two P2 decoding settings on the 1009 held-out real clips.

The reference fold is used only for scoring. Prompt names come from folds 1-4.
No training or local GPU is used.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

WORK = Path("/kaggle/working")
OUT = WORK / "out"
TMP = Path("/tmp/asr_p2_sweep")
BASE_ID = "vinai/PhoWhisper-large"
COPY = ["preprocessor_config.json", "tokenizer_config.json", "vocab.json", "merges.txt", "normalizer.json", "special_tokens_map.json", "added_tokens.json"]
NUM = {"không": "0", "một": "1", "hai": "2", "ba": "3", "bốn": "4", "năm": "5", "sáu": "6", "bảy": "7", "tám": "8", "chín": "9", "mười": "10"}
SETTINGS = {"baseline": {"beam_size": 5}, "beam8": {"beam_size": 8}, "prompt": {"beam_size": 5}}


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def norm(s):
    s = unicodedata.normalize("NFC", s.lower())
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(NUM.get(w, w) for w in s.split())


def audio_base(data, first_rel):
    archive = data / "real.zip"
    if archive.is_file():
        dst = TMP / "audio"
        dst.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dst)
        if not (dst / first_rel).is_file():
            raise RuntimeError(f"real.zip extracted, but {first_rel} missing")
        return dst
    candidates = [data] + [p for p in data.rglob("*") if p.is_dir()]
    found = next((p for p in candidates if (p / first_rel).is_file()), None)
    if found is None:
        raise RuntimeError(f"audio path {first_rel} not found under {data}")
    return found


def prompt_from_training(rows):
    names = Counter()
    for row in rows:
        if row["fold"] == 0:
            continue
        words = re.findall(r"[^\W_]+", row["text"], flags=re.UNICODE)
        for a, b in zip(words, words[1:]):
            if a[:1].isupper() and b[:1].isupper() and len(a) > 2 and len(b) > 2:
                names[f"{a} {b}"] += 1
    picked = [name for name, n in names.most_common() if n >= 2][:20]
    return "Tên riêng và từ vựng trong các yêu cầu: " + ", ".join(picked) + ".", picked


def build_ct2(adapter):
    import torch
    from ctranslate2.converters import TransformersConverter
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration

    base = Path(snapshot_download(BASE_ID, local_dir=str(TMP / "base_hf")))
    model = WhisperForConditionalGeneration.from_pretrained(str(base), torch_dtype=torch.float32)
    before = model.model.decoder.layers[0].self_attn.q_proj.weight.detach().clone()
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    delta = (model.model.decoder.layers[0].self_attn.q_proj.weight - before).abs().mean().item()
    if delta <= 0:
        raise RuntimeError("P2 LoRA merge produced no change")
    merged = TMP / "merged_hf"
    model.save_pretrained(str(merged), safe_serialization=True)
    for name in COPY:
        if (base / name).is_file():
            shutil.copy(base / name, merged / name)
    del model
    dst = TMP / "p2_ct2"
    TransformersConverter(str(merged), copy_files=[n for n in COPY if (merged / n).is_file()]).convert(str(dst), quantization="float16", force=True)
    log("P2 CT2 ready", dst, "merge delta", delta)
    return dst


def worker(gpu, setting, config, ct2, base, rows, prompt):
    import soundfile as sf
    from faster_whisper import WhisperModel

    model = WhisperModel(str(ct2), device="cuda", device_index=gpu, compute_type="float16")
    path = OUT / f"{setting}.jsonl"
    done = {r["audio"] for r in read_jsonl(path)} if path.is_file() else set()
    with path.open("a", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            if row["audio"] in done:
                continue
            audio, sr = sf.read(str(base / row["audio"]), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                raise RuntimeError(f"unexpected sample rate {sr}: {row['audio']}")
            opts = dict(config)
            if setting == "prompt":
                opts["initial_prompt"] = prompt
            segments, _ = model.transcribe(audio, language="vi", condition_on_previous_text=False, **opts)
            parts = list(segments)
            hyp = " ".join(part.text.strip() for part in parts).strip()
            f.write(json.dumps({"audio": row["audio"], "hyp": hyp, "avg_logprob": min((p.avg_logprob for p in parts), default=0.0)}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 100 == 0:
                log(setting, i, "/", len(rows))
    log(setting, "complete", len(read_jsonl(path)))


def score(rows, outputs):
    import jiwer

    by_audio = {r["audio"]: r["hyp"] for r in outputs}
    if len(by_audio) != len(rows):
        raise RuntimeError(f"only {len(by_audio)} of {len(rows)} clips decoded")
    edits = words = exact = high = 0
    for row in rows:
        ref, hyp = norm(row["text"]), norm(by_audio[row["audio"]])
        result = jiwer.process_words(ref, hyp)
        edits += result.substitutions + result.deletions + result.insertions
        words += len(ref.split())
        exact += result.wer == 0
        high += result.wer >= 0.5
    return {"clips": len(rows), "words": words, "edits": edits, "wer": edits / words, "exact_clips": exact, "wer_ge_50pct": high}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    inp = Path("/kaggle/input")
    real_paths = list(inp.rglob("real.jsonl"))
    adapter_paths = [p for p in inp.rglob("adapter_config.json") if "lifemate-phowhisper-p2-main-adapter" in str(p)]
    log("real.jsonl candidates", [str(p) for p in real_paths])
    log("adapter candidates", [str(p) for p in adapter_paths])
    if len(real_paths) != 1 or len(adapter_paths) != 1:
        raise RuntimeError("expected exactly one real.jsonl and one P2 adapter")
    real = read_jsonl(real_paths[0])
    fold0 = [r for r in real if r["fold"] == 0]
    if len(fold0) != 1009:
        raise RuntimeError(f"fold0 changed: {len(fold0)}")
    base = audio_base(real_paths[0].parent, fold0[0]["audio"])
    missing = [r["audio"] for r in fold0 if not (base / r["audio"]).is_file()]
    if missing:
        raise RuntimeError(f"missing fold0 audio: {missing[:5]} ({len(missing)} total)")
    prompt, names = prompt_from_training(real)
    log("audio base", base, "prompt", prompt)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.1", "peft==0.17.1", "ctranslate2==4.8.4", "faster-whisper==1.2.1", "jiwer==4.0.0", "soundfile"], check=True)
    ct2 = build_ct2(adapter_paths[0].parent)
    env = dict(os.environ)
    import nvidia.cublas, nvidia.cudnn
    env["LD_LIBRARY_PATH"] = ":".join([str(Path(module.__path__[0]) / "lib") for module in (nvidia.cublas, nvidia.cudnn)] + [env.get("LD_LIBRARY_PATH", "")])
    settings = list(SETTINGS.items())
    for offset in range(0, len(settings), 2):
        jobs = []
        for gpu, (setting, config) in enumerate(settings[offset:offset + 2]):
            args = [sys.executable, __file__, "worker", str(gpu), setting, json.dumps(config), str(ct2), str(base), str(real_paths[0]), prompt]
            jobs.append(subprocess.Popen(args, env=env))
        exits = [job.wait() for job in jobs]
        if any(exits):
            raise RuntimeError(f"decode workers failed: {exits}")
    metrics = {setting: score(fold0, read_jsonl(OUT / f"{setting}.jsonl")) for setting in SETTINGS}
    metrics["older_quyenuit24_p2_reference"] = {"clips": 1009, "words": 14531, "edits": 2087, "wer": 2087 / 14531}
    metrics["prompt_names_train_only"] = names
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    log("metrics", json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        _, _, gpu, setting, config, ct2, base, real_path, prompt = sys.argv
        rows = [r for r in read_jsonl(real_path) if r["fold"] == 0]
        worker(int(gpu), setting, json.loads(config), Path(ct2), Path(base), rows, prompt)
    else:
        main()
