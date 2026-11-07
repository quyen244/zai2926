"""Kaggle job: where do the ~20 s per agent step of vLLM + Qwen3-4B-AWQ (Triton attention) on a T4 go?

Push: `kaggle kernels push -p kaggle/vllm_awq --accelerator NvidiaTeslaT4`. Self-contained (Kaggle runs one file).
Prompts are invented, same shape as LifeMate prompts (~4.8k tokens step 1, ~5.2k step 2), rendered with Qwen3's chat
template (tools, enable_thinking=False); no competition data leaves the workstation.

llm_bench v4 (2026-11-04, same GPU and engine) measured step 1 p50 21.0 s for ~58 output tokens (3 tok/s), yet 12 prompts
batched took 22.3 s in total. A per-request cost that does not grow with batch size points at the decode step itself
(each step ~0.35 s instead of the ~10 ms T4 memory bandwidth allows) or at a one-off cost per new prompt shape. The probes
separate these:
  prefill    long prompt, max_tokens=1                 -> prefill time alone
  decode_short / decode_long   64 forced tokens (ignore_eos) after a ~40 / ~4.8k token prompt -> ms per decode step,
                               and whether it depends on context length (attention) or not (weights / overhead)
  pass_a     12 step-1 prompts, one at a time          -> what v4 measured
  pass_b     12 new step-1 prompts (other date, so no prefix-cache hit past the template head), one at a time
             -> pass_a >> pass_b means a warm-up / compile cost, pass_a ~ pass_b means steady-state latency
  pass_a_again  pass_a prompts again                   -> effect of a full prefix-cache hit
  step2      12 step-2 prompts (one tool result appended), one at a time
  batch12    the 12 pass_a prompts in one generate()
Writes /kaggle/working/results.json.
"""
from __future__ import annotations

import json
import os
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

# T4 is compute capability 7.5: no FlashAttention 2, and vLLM's default fallback (FlexAttention) is ~4x slower in batch
# (llm_bench v2 vs v4). Must be set before vllm is imported.
os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN_VLLM_V1"

OUT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".")
MODEL = "Qwen/Qwen3-4B-AWQ"
MAX_NEW = 160

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
SYSTEM = ("Bạn là LifeMate, trợ lý giọng nói điều khiển các app đã cài qua tool calling. Chỉ gọi tool trong danh sách. "
          "Thiếu tham số bắt buộc thì gọi sys.ask_missing_info. Không app nào làm được thì không gọi tool. "
          "Giờ hiện tại: 2026-{month:02d}-{day:02d}T08:{minute:02d}:00+07:00 (Asia/Ho_Chi_Minh).")


def tools() -> list[dict]:
    return [{"type": "function", "function": {
        "name": f"{app}.{n}", "description": f"Tool {n} của app {app}: thao tác trên dữ liệu người dùng, trả về kết quả dạng JSON",
        "parameters": {"type": "object", "properties": {
            "id": {"type": "string", "description": f"ID trong {app}, lấy từ tool đọc"},
            "value": {"type": "string"}, "time": {"type": "string", "format": "date-time"}}, "required": ["id"]}}}
        for app, names in APPS.items() for n in names]


def render(tok, msgs: list[dict], with_tools: bool = True) -> str:
    return tok.apply_chat_template(msgs, tools=tools() if with_tools else None, add_generation_prompt=True, tokenize=False, enable_thinking=False)


def step1(tok, month: int) -> list[str]:
    return [render(tok, [{"role": "system", "content": SYSTEM.format(month=month, day=1 + i, minute=i)}, {"role": "user", "content": r}])
            for i, r in enumerate(REQUESTS)]


def step2(tok) -> list[str]:
    result = json.dumps({"ok": True, "contacts": [{"id": f"c_{k:02d}", "name": f"Người {k}", "phone": f"09000000{k:02d}"} for k in range(12)]}, ensure_ascii=False)
    return [render(tok, [{"role": "system", "content": SYSTEM.format(month=11, day=1 + i, minute=i)}, {"role": "user", "content": r},
                         {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "contacts.search", "arguments": {"id": "c_01"}}}]},
                         {"role": "tool", "content": result}]) for i, r in enumerate(REQUESTS)]


def stats(lat: list[float]) -> dict:
    s = sorted(lat)
    return {"mean_s": round(st.mean(s), 3), "p50_s": round(s[len(s) // 2], 3), "min_s": round(s[0], 3), "max_s": round(s[-1], 3),
            "each_s": [round(x, 2) for x in lat]}


def main() -> None:
    t0 = time.time()
    subprocess.run("nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv", shell=True)
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    subprocess.run(pip + ["vllm==0.10.2"], check=False)
    subprocess.run(pip + ["transformers==4.57.1"], check=False)  # vLLM 0.10.2 breaks on transformers 5.x (tokenizer API)
    install_s = time.time() - t0

    import torch
    import vllm
    from vllm import LLM, SamplingParams
    env = {"install_s": round(install_s, 1), "gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
           "torch": torch.__version__, "vllm": vllm.__version__, "attention_backend": os.environ["VLLM_ATTENTION_BACKEND"]}
    print(json.dumps(env), flush=True)

    t = time.time()
    llm = LLM(model=MODEL, max_model_len=16384, gpu_memory_utilization=0.88, dtype="half", seed=0)
    res: dict = {"env": env, "load_s": round(time.time() - t, 1)}
    try:
        res["quantization"] = str(llm.llm_engine.model_config.quantization)
    except Exception as e:
        res["quantization"] = f"? {e}"
    tok = llm.get_tokenizer()
    greedy = SamplingParams(temperature=0, max_tokens=MAX_NEW, seed=0)
    one = SamplingParams(temperature=0, max_tokens=1, seed=0)
    forced = SamplingParams(temperature=0, max_tokens=64, min_tokens=64, ignore_eos=True, seed=0)

    def timed(prompts: list[str], sp) -> tuple[float, list]:
        t = time.time()
        o = llm.generate(prompts, sp, use_tqdm=False)
        return time.time() - t, o

    a, b = step1(tok, 11), step1(tok, 10)
    s2 = step2(tok)
    short = render(tok, [{"role": "user", "content": "kể một câu chuyện ngắn về con mèo"}], with_tools=False)
    res["prompt_tokens"] = {"step1": len(tok(a[0])["input_ids"]), "step2": len(tok(s2[0])["input_ids"]), "short": len(tok(short)["input_ids"])}

    first, _ = timed([a[0]], greedy)  # the very first request after load, reported, then excluded from the passes
    res["first_request_s"] = round(first, 2)

    res["prefill"] = stats([timed([p], one)[0] for p in b[:3]])
    for name, p in (("decode_short", short), ("decode_long", b[3])):
        p1 = timed([p], one)[0]
        lat = [timed([p], forced)[0] for _ in range(3)]
        res[name] = {"prefill_s": round(p1, 3), "64tok_s": stats(lat), "ms_per_decode_step": round(1000 * (min(lat) - p1) / 63, 1)}

    for name, ps in (("pass_a", a), ("pass_b", b), ("pass_a_again", a), ("step2", s2)):
        lat, outs = [], []
        for p in ps:
            dt, o = timed([p], greedy)
            lat.append(dt); outs.append(o[0].outputs[0])
        ntok = sum(len(x.token_ids) for x in outs)
        res[name] = {**stats(lat), "out_tok": ntok, "tok_per_s": round(ntok / sum(lat), 1),
                     "tool_call_outputs": sum("<tool_call>" in x.text for x in outs), "sample": outs[0].text[:300]}
        print(name, json.dumps({k: v for k, v in res[name].items() if k != "sample"}), flush=True)

    bt, bo = timed(a, greedy)
    n = sum(len(o.outputs[0].token_ids) for o in bo)
    res["batch12"] = {"total_s": round(bt, 3), "out_tok": n, "tok_per_s": round(n / bt, 1)}
    res["total_job_s"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT / "results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("pass_a", "pass_b", "pass_a_again", "step2")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
