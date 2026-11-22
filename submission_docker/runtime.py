"""LifeMate inference runtime used by predict.py and predict_notebook.ipynb (fully offline).

Pipeline per sample: PhoWhisper-large fine-tuned (CTranslate2 float16, beam 5, deterministic) on the
final clip and every history user clip -> e5 app filter (top-7 installed apps + contacts + sys) -> Qwen3-4B + REC LoRA
served by a local vLLM (OpenAI API on 127.0.0.1, temperature 0, seed 0) in the lifemate agent loop
(one zaic_tools ToolEnv per sample, ID guard, at most 6 steps).
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CODE = Path(__file__).resolve().parent
MODELS = CODE / "saved_models"
LLM_DIR = MODELS / "qwen3-4b"
ADAPTER = MODELS / "rec_adapter"
ASR_DIR = MODELS / "phowhisper-ft-ct2"
E5_DIR = MODELS / "app_embedder_e5"
PORT = 8000
LORA = "traj_rec"
STEP_CAP = 6
NORMALISE = True  # REC was trained on normalised text
sys.path.insert(0, str(CODE))


def seed_everything(seed: int = 42) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_split(root: Path) -> str:
    return next(p.name for p in sorted(root.iterdir()) if (p / f"{p.name}_tools_catalog.json").exists())


class Runtime:
    def __init__(self, root: Path = Path("/data")):
        self.root = root
        self.split = find_split(root)
        os.environ["LIFEMATE_DATA_ROOT"] = str(root)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        t0 = time.time()
        self.server = self._start_vllm()
        from faster_whisper import WhisperModel
        self.asr = WhisperModel(str(ASR_DIR), device="cuda", compute_type="float16")
        from lifemate.app_filter import AppFilter
        self.filter = AppFilter(E5_DIR, split=self.split, root=root, device="cpu")
        from transformers import AutoTokenizer
        from lifemate.llm import LLM
        from lifemate.sft import history_think_template
        template = history_think_template(AutoTokenizer.from_pretrained(str(LLM_DIR)).chat_template)
        self.llm = LLM(model=LORA, base_url=f"http://127.0.0.1:{PORT}/v1", timeout=120, chat_template=template)
        self.load_s = time.time() - t0
        print(f"load model: {self.load_s:.1f} s", flush=True)

    def _start_vllm(self) -> subprocess.Popen:
        log = open("/tmp/vllm.log", "w")
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(LLM_DIR),
               "--served-model-name", "qwen3-4b", "--host", "127.0.0.1", "--port", str(PORT),
               "--max-model-len", "16384", "--gpu-memory-utilization", "0.70", "--dtype", "half", "--seed", "0",
               "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
               "--enable-lora", "--max-lora-rank", "16", "--max-loras", "1", "--lora-modules", f"{LORA}={ADAPTER}"]
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        while True:
            if p.poll() is not None:
                raise SystemExit("vLLM died:\n" + Path("/tmp/vllm.log").read_text()[-3000:])
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5)
                return p
            except OSError:
                time.sleep(3)

    def transcribe(self, rel: str) -> str:
        segs, _ = self.asr.transcribe(str(self.root / self.split / rel), language="vi", beam_size=5,
                                      condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segs).strip()

    def load_cases(self) -> list[dict]:
        from lifemate.data import load_samples
        return load_samples(self.split, self.root)

    def infer(self, item: dict) -> tuple[list[dict], str]:
        """ASR -> app filter -> agent on one fresh ToolEnv (created inside run_sample). Returns (tool_calls, state_hash)."""
        from lifemate.agent import run_sample
        clips = [item["audio"]] + [t["audio"] for t in item.get("history") or [] if t.get("role") == "user" and t.get("audio")]
        asr = {c: self.transcribe(c) for c in dict.fromkeys(clips)}
        text = asr[item["audio"]]
        keep = self.filter.keep(text, list(item["context"]["installed_apps"]))
        row = run_sample(item, self.split, text, self.llm, asr, STEP_CAP, keep_apps=keep, id_guard=True,
                         normalise=NORMALISE)
        return row["tool_calls"], row["state_hash"]

    def close(self) -> None:
        self.server.terminate()
