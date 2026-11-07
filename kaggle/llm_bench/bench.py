"""Kaggle job: is Qwen3-4B (4-bit) faster through vLLM or through Hugging Face transformers?

Runs on a Kaggle GPU kernel (push with `kaggle kernels push -p kaggle/llm_bench --accelerator NvidiaTeslaT4`).
No competition data leaves the workstation: the prompts are invented, built to the same shape as LifeMate prompts
(Vietnamese system rules + ~45 tool JSON schemas + a short request, sized to the ~5k-token Stage 2.3 median) and rendered with
Qwen3's own chat template (tools, enable_thinking=False). Each engine runs in its own subprocess so GPU memory is
released between engines. Writes /kaggle/working/results.json and prints a summary.

Measured per engine: load time; sequential latency of one agent step (greedy, max 160 new tokens) on 12 different
prompts, which is what the 30 s per-sample budget depends on; the same prompts again with one tool result appended
(step 2 of an agent loop: prefix caching helps here); and batched throughput over all 12 prompts.
"""
from __future__ import annotations

import json
import os
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

OUT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".")
MAX_NEW = 160
# v2 on a T4 (no FlashAttention 2 before compute capability 8.0): vLLM fell back to FlexAttention and ran ~26 s per
# step; HF bnb4 ~22 s; HF fp16 ran out of memory. v3 retries vLLM with the Triton attention backend, which supports T4.
ENGINES = {
    "vllm_awq_triton": {"model": "Qwen/Qwen3-4B-AWQ", "kind": "vllm", "quant": None, "env": {"VLLM_ATTENTION_BACKEND": "TRITON_ATTN_VLLM_V1"}},
    "vllm_bnb4_triton": {"model": "Qwen/Qwen3-4B", "kind": "vllm", "quant": "bitsandbytes", "env": {"VLLM_ATTENTION_BACKEND": "TRITON_ATTN_VLLM_V1"}},
    "hf_bnb4": {"model": "Qwen/Qwen3-4B", "kind": "hf", "quant": "bnb4"},
}

APPS = {
    "smart_home": ["get_status", "light_set", "fan_set", "climate_set", "scene_run"], "alarm": ["list", "create", "update", "delete"],
    "reminders": ["list", "create", "complete"], "contacts": ["search", "get"], "phone": ["call"], "sms": ["send", "read_recent"],
    "banking": ["get_balance", "transfer", "list_beneficiaries"], "ewallet": ["get_balance", "transfer", "topup_mobile"],
    "calendar": ["list_events", "create_event", "update_event", "cancel_event"], "ride": ["get_quotes", "book", "cancel"],
    "food": ["list_restaurants", "get_menu", "place_order"], "music": ["list_tracks", "play", "control"],
    "weather": ["get_forecast"], "notes": ["create", "list"], "email": ["create_draft", "send"],
}
REQUESTS = ["tắt đèn phòng khách giúp anh", "nếu mai mưa thì dời lịch chạy bộ muộn một tiếng", "chuyển cho mẹ năm trăm nghìn",
            "đặt xe bốn chỗ từ công ty về nhà", "nhắc tôi họp lúc ba giờ chiều mai", "gọi cho lan nhé",
            "bật điều hoà phòng ngủ hai sáu độ", "mở nhạc thư giãn", "xem số dư tài khoản", "gửi tin nhắn cho bố là con về muộn",
            "đặt hai suất cơm tấm", "tạo ghi chú mua sữa"]


def tools() -> list[dict]:
    out = []
    for app, names in APPS.items():
        for n in names:
            out.append({"type": "function", "function": {
                "name": f"{app}.{n}", "description": f"Tool {n} của app {app}: thao tác trên dữ liệu người dùng, trả về kết quả dạng JSON",
                "parameters": {"type": "object", "properties": {
                    "id": {"type": "string", "description": f"ID trong {app}, lấy từ tool đọc"},
                    "value": {"type": "string"}, "time": {"type": "string", "format": "date-time"}},
                    "required": ["id"]}}})
    return out


SYSTEM = ("Bạn là LifeMate, trợ lý giọng nói điều khiển các app đã cài qua tool calling. Chỉ gọi tool trong danh sách. "
          "Thiếu tham số bắt buộc thì gọi sys.ask_missing_info. Không app nào làm được thì không gọi tool. "
          "Giờ hiện tại: 2026-11-{day:02d}T08:{minute:02d}:00+07:00 (Asia/Ho_Chi_Minh).")


def prompts(tok) -> tuple[list[str], list[str]]:
    t = tools()
    step1, step2 = [], []
    for i, r in enumerate(REQUESTS):
        msgs = [{"role": "system", "content": SYSTEM.format(day=1 + i, minute=i)}, {"role": "user", "content": r}]
        step1.append(tok.apply_chat_template(msgs, tools=t, add_generation_prompt=True, tokenize=False, enable_thinking=False))
        msgs2 = msgs + [{"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "contacts.search", "arguments": {"id": "c_01"}}}]},
                        {"role": "tool", "content": json.dumps({"ok": True, "contacts": [{"id": f"c_{k:02d}", "name": f"Người {k}", "phone": f"09000000{k:02d}"} for k in range(12)]}, ensure_ascii=False)}]
        step2.append(tok.apply_chat_template(msgs2, tools=t, add_generation_prompt=True, tokenize=False, enable_thinking=False))
    return step1, step2


def summary(lat: list[float]) -> dict:
    s = sorted(lat)
    return {"mean_s": round(st.mean(s), 3), "p50_s": round(s[len(s) // 2], 3), "max_s": round(s[-1], 3)}


def run_vllm(cfg: dict) -> dict:
    from vllm import LLM, SamplingParams
    t0 = time.time()
    kw = dict(model=cfg["model"], max_model_len=16384, gpu_memory_utilization=0.88, dtype="half", seed=0)
    if cfg["quant"]:
        kw["quantization"] = cfg["quant"]
    llm = LLM(**kw)
    load = time.time() - t0
    tok = llm.get_tokenizer()
    p1, p2 = prompts(tok)
    sp = SamplingParams(temperature=0, max_tokens=MAX_NEW, seed=0)
    llm.generate([p1[0]], sp, use_tqdm=False)  # warm-up (CUDA graphs, kernels), not timed
    res = {}
    for name, ps in (("step1", p1), ("step2", p2)):
        lat, ntok, outs = [], 0, []
        for p in ps:
            t = time.time(); o = llm.generate([p], sp, use_tqdm=False)[0]; lat.append(time.time() - t)
            ntok += len(o.outputs[0].token_ids); outs.append(o.outputs[0].text)
        res[name] = {**summary(lat), "out_tok": ntok, "tok_per_s": round(ntok / sum(lat), 1), "tool_call_outputs": sum("<tool_call>" in x for x in outs), "sample": outs[0][:300]}
    t = time.time(); os_ = llm.generate(p1, sp, use_tqdm=False); bt = time.time() - t
    res["batch12"] = {"total_s": round(bt, 3), "out_tok": sum(len(o.outputs[0].token_ids) for o in os_), "tok_per_s": round(sum(len(o.outputs[0].token_ids) for o in os_) / bt, 1)}
    res["prompt_tokens"] = [len(tok(p)["input_ids"]) for p in (p1[0], p2[0])]
    return {"load_s": round(load, 1), **res}


def run_hf(cfg: dict) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    kw = dict(dtype=torch.float16, device_map="cuda:0", attn_implementation="sdpa")
    if cfg["quant"] == "bnb4":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], **kw).eval()
    load = time.time() - t0
    p1, p2 = prompts(tok)

    def gen(texts):
        tok.padding_side = "left"
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda:0")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False)
        new = out[:, enc["input_ids"].shape[1]:]
        return [tok.decode(x, skip_special_tokens=False) for x in new], [int((x != tok.pad_token_id).sum()) for x in new]
    gen([p1[0]])
    res = {}
    for name, ps in (("step1", p1), ("step2", p2)):
        lat, ntok, outs = [], 0, []
        for p in ps:
            torch.cuda.synchronize(); t = time.time(); txt, n = gen([p]); torch.cuda.synchronize(); lat.append(time.time() - t)
            ntok += n[0]; outs.append(txt[0])
        res[name] = {**summary(lat), "out_tok": ntok, "tok_per_s": round(ntok / sum(lat), 1), "tool_call_outputs": sum("<tool_call>" in x for x in outs), "sample": outs[0][:300]}
    try:
        torch.cuda.synchronize(); t = time.time(); txt, n = gen(p1); torch.cuda.synchronize(); bt = time.time() - t
        res["batch12"] = {"total_s": round(bt, 3), "out_tok": sum(n), "tok_per_s": round(sum(n) / bt, 1)}
    except torch.OutOfMemoryError as e:
        res["batch12"] = {"error": f"OOM: {str(e)[:120]}"}
    res["prompt_tokens"] = [len(tok(p)["input_ids"]) for p in (p1[0], p2[0])]
    res["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return {"load_s": round(load, 1), **res}


def child(name: str) -> None:
    cfg = ENGINES[name]
    import torch
    if torch.cuda.get_device_capability(0) < (8, 0):  # Triton only where FlashAttention 2 is missing (T4); L4/A30 keep the default
        os.environ.update(cfg.get("env", {}))
    try:
        r = run_vllm(cfg) if cfg["kind"] == "vllm" else run_hf(cfg)
        r["ok"] = True
    except Exception as e:  # report the failure as a result: "does not run on this GPU" is an answer too
        import traceback
        r = {"ok": False, "error": f"{type(e).__name__}: {e}"[:600], "trace": traceback.format_exc()[-1500:]}
    r.update(engine=name, **cfg)
    with open(OUT / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    t0 = time.time()
    subprocess.run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv", shell=True)
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    subprocess.run(pip + ["vllm==0.10.2"], check=False)
    subprocess.run(pip + ["transformers==4.57.1"], check=False)  # vLLM 0.10.2 breaks on transformers 5.x (tokenizer API)
    subprocess.run(pip + ["bitsandbytes>=0.46.1", "accelerate"], check=False)
    install_s = time.time() - t0
    (OUT / "results.jsonl").unlink(missing_ok=True)
    for name in ENGINES:
        t = time.time()
        p = subprocess.run([sys.executable, __file__, "--child", name], timeout=1800)
        print(f"{name}: exit {p.returncode} in {time.time() - t:.0f}s", flush=True)
    rows = [json.loads(l) for l in open(OUT / "results.jsonl", encoding="utf-8")]
    env = {"install_s": round(install_s, 1)}
    try:
        import torch
        env.update(torch=torch.__version__, gpu=torch.cuda.get_device_name(0))
    except Exception as e:
        env["torch_error"] = str(e)
    for mod in ("vllm", "transformers", "bitsandbytes"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception as e:
            env[mod] = f"missing: {e}"
    json.dump({"env": env, "rows": rows}, open(OUT / "results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(env))
    for r in rows:
        if r["ok"]:
            print(f"{r['engine']:10s} load {r['load_s']:6.1f}s | step1 p50 {r['step1']['p50_s']:.2f}s max {r['step1']['max_s']:.2f}s "
                  f"{r['step1']['tok_per_s']} tok/s | step2 p50 {r['step2']['p50_s']:.2f}s | batch12 {r['batch12']} | prompt tok {r['prompt_tokens']}")
        else:
            print(f"{r['engine']:10s} FAILED {r['error']}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        child(sys.argv[2])
    else:
        main()
