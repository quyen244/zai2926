"""Write a readable trace of one sample through the real pipeline: input audio and its text, every processing node
(normaliser, date hints, prompt builder), every LLM call (the full request and the raw reply), how each reply is parsed,
validated and executed in the ToolEnv, and the final submission line (scored against gold on train).

  python scripts/trace_sample.py --id train_0001                                   # LLM replaced by the gold calls
  python scripts/trace_sample.py --id train_0001 --text gold                       # reference transcript instead of ASR
  python scripts/trace_sample.py --id train_0018 train_0095 --llm cache --text gold   # several samples, one file each
  python scripts/trace_sample.py --split public_test --id public_0001 --llm api --base-url http://HOST:8000/v1 --model Qwen/Qwen3-4B-AWQ

--llm cache replays the replies a Kaggle job recorded (work/kaggle/cache/<model>.<text>.*.jsonl, newest match, or
--cache FILE): no model runs, and the trace shows exactly what that run did, checked against the cached result.
  python scripts/trace_sample.py --id train_0006 --llm cache                       # B0 run of the last Kaggle job
  python scripts/trace_sample.py --id train_0006 --llm cache --text asr            # B1 run
Default view per LLM call: INPUT PROMPT (the text after the Qwen3 chat template) and OUTPUT; --request adds the full
JSON body. The prompt needs the tokenizer on disk (a few MB, no weights; without it the JSON body is printed instead):
  .venv/Scripts/python.exe -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-AWQ')"
The trace ends with predicted vs gold calls, one per line with their results (the show_gt.py layout).
--brief gives the old compact view.
--llm gold (train only) answers with label.tool_calls, one per turn, so the trace shows the pipeline without a model.
--llm api calls any OpenAI-compatible server (vLLM, Ollama) through lifemate.llm.LLM; without --base-url it uses the
OpenAI settings in .env (dev only, see .claude/CLAUDE.md). ASR text comes from the cached work/asr/phowhisper/<split>.jsonl
(no model runs here). The trace holds real per-sample data: it is written under work/traces/ (git-ignored).
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lifemate.agent import run_sample  # noqa: E402
from lifemate.data import PROJECT, audio_path, fuzzy_params, load_samples  # noqa: E402
from lifemate.gold import build_gold  # noqa: E402
from lifemate.llm import tools_body  # noqa: E402
from lifemate.normalize import normalize_text, resolve_dates  # noqa: E402
from lifemate.prompt import build_user_text  # noqa: E402
from lifemate.scoring import score_sample  # noqa: E402

RULE = "=" * 100


class GoldScript:
    """Stands in for lifemate.llm.LLM: same chat_tools() interface, replies with the gold calls one per turn."""

    def __init__(self, calls: list[dict]):
        self.calls, self.i = list(calls), 0

    def chat_tools(self, system, messages, tools, **_):
        if self.i >= len(self.calls):
            return {"role": "assistant", "content": "Đã xử lý xong.", "tool_calls": []}
        c = self.calls[self.i]
        self.i += 1
        return {"role": "assistant", "content": None, "tool_calls": [{"id": f"gold_{self.i}", "type": "function", "function": {
            "name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}]}


class CachedReplies:
    """Stands in for lifemate.llm.LLM: returns the replies recorded by a Kaggle job, in order."""

    def __init__(self, replies: list[dict]):
        self.replies, self.i = list(replies), 0

    def chat_tools(self, system, messages, tools, **_):
        if self.i >= len(self.replies):
            return {"role": "assistant", "content": "(hết reply đã cache)", "tool_calls": []}
        self.i += 1
        return self.replies[self.i - 1]


def make_renderer(name: str):
    """Qwen3 chat template, applied the way vLLM does (tool-call arguments parsed back to objects, enable_thinking from
    chat_template_kwargs). Only from files already on disk: returns None when the tokenizer is not there."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception:
        return None

    def render(body: dict) -> tuple[str, int]:
        msgs = []
        for m in body["messages"]:
            if m.get("tool_calls"):
                m = {**m, "tool_calls": [{**c, "function": {**c["function"], "arguments": json.loads(c["function"]["arguments"] or "{}")}} for c in m["tool_calls"]]}
            msgs.append(m)
        text = tok.apply_chat_template(msgs, tools=body["tools"], add_generation_prompt=True, tokenize=False,
                                       **body.get("chat_template_kwargs", {}))
        return text, len(tok(text)["input_ids"])
    return render


def dump(x) -> str:
    return json.dumps(x, ensure_ascii=False, indent=2)


def duration(path: Path) -> str:
    try:
        with wave.open(str(path)) as w:
            return f"{w.getnframes() / w.getframerate():.2f} s, {w.getframerate()} Hz, {w.getnchannels()} kênh"
    except (OSError, wave.Error) as e:
        return f"không đọc được ({e})"


class Tracer:
    """full (default): every LLM call prints the input prompt the model actually reads (Qwen3 chat template, needs the
    tokenizer on disk) and its raw output. --request also prints the exact JSON body POSTed to /chat/completions (printed
    anyway when the tokenizer is missing). brief: system/tools once, then only the new messages."""

    def __init__(self, brief: bool, model: str, openai: bool, render=None, request: bool = False):
        self.lines: list[str] = []
        self.brief, self.model, self.openai, self.render, self.request = brief, model, openai, render, request
        self.done: list[tuple[str, dict, dict]] = []  # (tool, args, result) of every env.call, for the final summary
        self.sent = 0  # brief mode: messages already printed in an earlier LLM request
        self.t = time.perf_counter()

    def h(self, title: str) -> None:
        self.lines += ["", RULE, title, RULE]

    def w(self, *parts: str) -> None:
        self.lines += list(parts)

    def __call__(self, event: str, d: dict) -> None:
        ms = f"[+{1000 * (time.perf_counter() - self.t):.0f} ms]"
        if event == "prompt":
            self.h(f"NODE build_prompt  {ms}  (lifemate/prompt.py)")
            self.w(f"system: {len(d['system'])} ký tự | tools: {len(d['tools'])} | messages: {len(d['messages'])}")
        elif event == "llm_request" and not self.brief:
            body = tools_body(self.model, self.openai, d["system"], d["messages"], d["tools"])
            if self.request or self.render is None:  # no tokenizer: the JSON body is the only view of the input
                self.h(f"LLM CALL #{d['step'] + 1}: REQUEST - JSON body gửi tới /chat/completions, đầy đủ  {ms}")
                if self.render is None:
                    self.w("(chưa có tokenizer Qwen3 trên máy nên in JSON body thay cho prompt; xem --tokenizer trong docstring)")
                self.w(f"({len(body['messages'])} message, {len(body['tools'])} tool, {len(json.dumps(body, ensure_ascii=False))} ký tự JSON)", dump(body))
            if self.render is not None:
                text, n_tok = self.render(body)
                self.h(f"LLM CALL #{d['step'] + 1}: INPUT PROMPT sau chat template (chuỗi model thật sự đọc)  {ms}")
                self.w(f"({n_tok} token)", text)
        elif event == "llm_request":
            self.h(f"LLM CALL #{d['step'] + 1}: REQUEST (rút gọn)  {ms}")
            if d["step"] == 0:
                self.w("--- system ---", d["system"], f"--- tools ({len(d['tools'])}) ---",
                       *[f"  {t['function']['name']}: {t['function']['description']}" for t in d["tools"]], "--- messages ---")
            else:
                self.w(f"(system và tools giống CALL #1; {len(d['messages'])} message, các message cũ đã in ở trên)")
            for k, m in enumerate(d["messages"][self.sent:], self.sent):
                self.w(f"[{k}] {m['role']}: " + (m["content"] if isinstance(m.get("content"), str) and not m.get("tool_calls") else dump({x: m[x] for x in m if x != "role"})))
            self.sent = len(d["messages"])
        elif event == "llm_reply":
            self.h(f"LLM CALL #{d['step'] + 1}: OUTPUT  {ms}")
            self.w(dump(d["reply"]))
            if not d["reply"].get("tool_calls"):
                self.w("-> không có tool call: vòng lặp dừng")
        elif event == "parse":
            self.w("", f"NODE parse arguments  {ms}", f"  tool: {d['name']}", f"  raw: {d['raw']}", f"  parsed: {json.dumps(d['args'], ensure_ascii=False)}")
        elif event == "validate":
            self.w(f"NODE env.validate (không ghi log)  {ms}", f"  {json.dumps(d['result'], ensure_ascii=False)}")
        elif event == "call":
            self.done.append((d["name"], d["args"], d["result"]))
            self.w(f"NODE env.call -> ToolEnv  {ms}", f"  {d['name']}({json.dumps(d['args'], ensure_ascii=False)})", f"  result: {json.dumps(d['result'], ensure_ascii=False)}",
                   f"  log của ToolEnv hiện có {d['logged']} call")
            self.sent += 2  # brief mode: the assistant call and its tool result are already printed here
        elif event == "final":
            self.h(f"OUTPUT: dòng submission  {ms}")
            self.w(f"id: {d['id']} | {len(d['tool_calls'])} call | state_hash {d['state_hash']}")
            self.w(*[f"    -> {c['name']} {c['arguments']}" for c in d["tool_calls"]] or ["    -> (không gọi tool nào)"])


@functools.cache
def gold_train() -> dict:
    """build_gold replays every train sample: do it once per run, not once per traced id."""
    return build_gold("train")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--id", nargs="+", required=True, help="one or more sample ids; one trace file each")
    ap.add_argument("--text", choices=["asr", "gold"], default="asr", help="final-turn text: cached ASR or reference transcript")
    ap.add_argument("--llm", choices=["gold", "api", "cache"], default="gold")
    ap.add_argument("--cache", type=Path, help="--llm cache: result-cache file (default: newest work/kaggle/cache/*.<text>.*.jsonl)")
    ap.add_argument("--base-url"); ap.add_argument("--model")
    ap.add_argument("--validate-first", action="store_true")
    ap.add_argument("--brief", "--brief-tools", dest="brief", action="store_true",
                    help="compact view (tools one line each, only new messages per call) instead of the full request")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B-AWQ", help="HF id or local dir of the tokenizer used to render the prompt text")
    ap.add_argument("--request", action="store_true", help="also print the full JSON body of every LLM request")
    ap.add_argument("--out", type=Path, help="output file (only with a single --id)")
    a = ap.parse_args()
    if a.out and len(a.id) > 1:
        ap.error("--out needs a single --id")
    if (a.llm == "gold" or a.text == "gold") and a.split != "train":
        raise SystemExit("--llm gold and --text gold need train (only train has labels and transcripts)")

    samples = {s["id"]: s for s in load_samples(a.split)}
    asr_rows = {}
    asr_file = PROJECT / "work" / "asr" / "phowhisper" / f"{a.split}.jsonl"
    if asr_file.is_file():
        asr_rows = {json.loads(l)["audio"]: json.loads(l) for l in asr_file.read_text(encoding="utf-8").splitlines() if l.strip()}
    cache_file, cache_rows = None, {}
    if a.llm == "cache":
        files = [a.cache] if a.cache else sorted((PROJECT / "work" / "kaggle" / "cache").glob(f"*.{a.text}.*.jsonl"), key=lambda f: f.stat().st_mtime)
        if not files:
            raise SystemExit(f"no cache file for --text {a.text}")
        cache_file = files[-1]
        cache_rows = {json.loads(l)["id"]: json.loads(l) for l in cache_file.read_text(encoding="utf-8").splitlines() if l.strip()}
        print(f"cache: {cache_file.name} ({len(cache_rows)} samples)")
    for sid in a.id:
        if sid not in samples:
            print(f"{sid}: not in {a.split}, skipped")
        elif a.llm == "cache" and sid not in cache_rows:
            print(f"{sid}: not in {cache_file.name}, skipped")
        elif a.text == "asr" and samples[sid]["audio"] not in asr_rows:
            print(f"{sid}: no cached ASR in {asr_file}, skipped")
        else:
            trace_one(a, sid, samples[sid], asr_rows, cache_file, cache_rows.get(sid))


def trace_one(a, sid: str, sample: dict, asr_rows: dict, cache_file: Path | None, cached: dict | None) -> None:
    asr = {k: v["hyp"] for k, v in asr_rows.items()}
    user_text = sample["audio_text"] if a.text == "gold" else asr[sample["audio"]]
    now = sample["context"]["current_time"]

    # what the request body names as model: the Kaggle vLLM model for cache/gold replays (same body the job sent)
    tr = Tracer(a.brief, "Qwen/Qwen3-4B-AWQ", False, request=a.request)
    tr.w(f"TRACE {sid} | split {a.split} | text {a.text} | llm {a.llm}{' ' + (a.model or '') if a.llm == 'api' else ''} | validate_first {a.validate_first}",
         f"tạo lúc {time.strftime('%Y-%m-%d %H:%M:%S')}")
    tr.h("INPUT")
    ctx = sample["context"]
    tr.w(f"current_time: {now} ({ctx['timezone']}) | location: {ctx['location']}", f"user_profile: {json.dumps(ctx['user_profile'], ensure_ascii=False)}",
         f"installed_apps ({len(ctx['installed_apps'])}): {', '.join(ctx['installed_apps'])}")
    turns = [(f"history[{k}]", h) for k, h in enumerate(sample["history"])] + [("LƯỢT CUỐI (được chấm)", {"role": "user", "audio": sample["audio"], "audio_text": sample.get("audio_text")})]
    for label, t in turns:
        tr.w("")
        if t["role"] == "user":
            row = asr_rows.get(t["audio"], {})
            tr.w(f"{label} user audio: {t['audio']} ({duration(audio_path(a.split, t['audio']))})",
                 f"  nội dung audio (reference): {t.get('audio_text') or '(không có ở split này)'}",
                 f"  ASR PhoWhisper (cache):     {row.get('hyp', '(không có)')}",
                 f"    giải mã {row.get('sec_decode', '?')} s, avg_logprob {row.get('avg_logprob', '?')}, redo {row.get('redo', '?')}")
        else:
            tr.w(f"{label} assistant: {t.get('text', '')}")
            for c, r in zip(t.get("tool_calls") or [], t.get("tool_results") or []):
                tr.w(f"  đã gọi {c['name']}({json.dumps(c['arguments'], ensure_ascii=False)}) -> {json.dumps(r, ensure_ascii=False)}")
    tr.h("NODE normalize_text + resolve_dates  (lifemate/normalize.py, mỗi lượt user)")
    for label, t in turns:
        if t["role"] != "user":
            continue
        raw = (user_text if label.startswith("LƯỢT") else asr.get(t["audio"], t.get("audio_text"))) or ""
        tr.w(f"{label}", f"  vào:     {raw}", f"  ra:      {normalize_text(raw)}",
             f"  ngày:    {json.dumps(resolve_dates(normalize_text(raw), now), ensure_ascii=False)}",
             f"  text gửi LLM: {build_user_text(raw, now)!r}")

    if a.llm == "cache":
        tr.w("", f"LLM: phát lại {len(cached['replies'])} reply đã cache từ {cache_file.name} (lúc chạy trên Kaggle mất {cached.get('sec', '?')} s)")
        llm = CachedReplies(cached["replies"])
    elif a.llm == "gold":
        llm = GoldScript(sample["label"]["tool_calls"])
    else:
        from lifemate.llm import LLM
        llm = LLM(model=a.model, base_url=a.base_url)
        tr.model, tr.openai = llm.model, llm.openai
    if not a.brief:
        tr.render = make_renderer(a.tokenizer)
    keep = None  # a cache file tagged ".topK." was run with the tool filter: replay it with the same tool list
    parts = cache_file.name.split(".") if cache_file else []
    if len(parts) == 6 and parts[3].startswith("top"):
        f = PROJECT / "work" / "retrieval" / f"app_filter.{parts[3]}.json"
        keep = set(json.loads(f.read_text(encoding="utf-8"))["keep"][a.text][sid])
        tr.w(f"tool filter ({parts[3]}, {f.name}): chỉ đưa tool của {', '.join(sorted(keep))} + sys")
    row = run_sample(sample, a.split, user_text, llm, asr=(asr or None) if a.text == "asr" else None, validate_first=a.validate_first, trace=tr,
                     keep_apps=keep)

    if cached is not None:
        same = row["tool_calls"] == cached["tool_calls"] and row["state_hash"] == cached["state_hash"]
        tr.w("", f"phát lại khớp kết quả đã cache trên Kaggle: {'CÓ' if same else 'KHÔNG (code ở máy đã khác code lúc chạy)'}")
    if a.split == "train":
        gold = gold_train()[sid]
        sc = score_sample(row["tool_calls"], row["state_hash"], sample["label"]["tool_calls"], gold["state_hash"], fuzzy_params("train"))
        tr.h("SCORE so với gold (lifemate/scoring.py)")
        tr.w(f"hash khớp: {sc.hash_ok} | M = {sc.matched}, U = {sc.unmatched}, G = {sc.n_gold} | P = {sc.precision:.3f}, R = {sc.recall:.3f} | điểm = {sc.score:.3f}")
        lab = sample["label"]
        tr.h("KẾT QUẢ CUỐI: predicted vs gold (dạng show_gt.py)")
        for title, calls in ((f"predicted: {len(tr.done)} call | điểm {sc.score:.3f}, hash khớp {sc.hash_ok}", tr.done),
                             (f"gold: {len(lab['tool_calls'])} call", [(c["name"], c["arguments"], r) for c, r in zip(lab["tool_calls"], lab["tool_results"])])):
            tr.w(f"- {title}")
            if not calls:
                tr.w("    -> (không gọi tool nào)")
            for name, args, res in calls:
                tr.w(f"    -> {name} {args}", f"         = {json.dumps(res, ensure_ascii=False)[:300]}")
    llm_tag = f"cache-{parts[0]}{'-' + parts[3] if len(parts) == 6 else ''}" if a.llm == "cache" else a.llm  # model in the name: Qwen and QVikhr traces do not overwrite each other
    out = a.out or PROJECT / "work" / "traces" / f"{sid}.{a.text}.{llm_tag}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(tr.lines) + "\n", encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
