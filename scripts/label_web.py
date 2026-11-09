"""Local web page to hand-label train samples for plan rule R3 (docs/plan_format.md): is the request conditional, and
which gold [read] call checks the condition. Runs on this machine only; labels stay in work/ (git-ignored).

  python scripts/label_web.py                 # http://127.0.0.1:8765
  python scripts/label_web.py --port 8800

Queues (default first): "bị loại ở tầng 1" = the 8B said conditional, the request has NO condition word, the gold
calls a [read] tool; "qua tầng 1" = the samples the first Qwen3-8B pass called conditional AND whose request has a condition word AND whose gold calls a [read] tool
(thì, nếu, ...), in a fixed shuffled order (seed 0) so a partial pass is still a fair sample; "tất cả" = every train
sample. The 8B label is hidden until you ask for it, so it does not anchor your answer.
Output: work/sft/manual_labels.jsonl, one row per sample:
  {"id", "conditional": bool | null (null = không chắc), "check_call": int | null, "condition": str, "note": str, "t": unix time}
Keys: 1-9 pick the k-th call as check (it must be [read]), N not conditional, U unsure, C copy the selected request
text as the condition, Enter save + next, arrow left/right move, H show/hide the 8B label.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lifemate.data import PROJECT, load_catalog, load_samples  # noqa: E402
from plan_data import plan_segments  # noqa: E402

OUT = PROJECT / "work" / "sft" / "manual_labels.jsonl"
PASS1 = PROJECT / "work" / "kaggle" / "plan_check" / "labels.jsonl"
MARKER = re.compile(r"\b(thì|nếu|hễ|lỡ|trường hợp|miễn là|còn không|không thì|kẻo)\b", re.I)

SAMPLES = {s["id"]: s for s in load_samples("train")}
KINDS = {t["name"]: t["kind"] for x in load_catalog("train")["apps"] for t in x["tools"]}
P1 = {(r := json.loads(l))["id"]: r["reply"] for l in PASS1.read_text(encoding="utf-8").splitlines() if l.strip()} if PASS1.is_file() else {}


def load_manual() -> dict[str, dict]:
    if not OUT.is_file():
        return {}
    return {(r := json.loads(l))["id"]: r for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()}


MANUAL = load_manual()
def kind(name: str) -> str:
    return "ask" if name == "sys.ask_missing_info" else KINDS.get(name, "read")


def has_read(i: str) -> bool:
    """A replan needs a call to wait on: at least one gold call must be a [read] tool (a condition answered by the
    history or by the user's own words has none, and is not R3)."""
    return any(kind(c["name"]) == "read" for c in SAMPLES[i]["label"]["tool_calls"])


SUSPECT = sorted(i for i, r in P1.items() if r.get("conditional") and MARKER.search(SAMPLES[i]["audio_text"]) and has_read(i))
random.Random(0).shuffle(SUSPECT)
# dropped by tier 1: the 8B said conditional but the request has no condition word; only those with a [read] call
# (without one there is nothing to replan on). Labelled by hand to measure what tier 1 throws away.
DROPPED = sorted(i for i, r in P1.items() if r.get("conditional") and not MARKER.search(SAMPLES[i]["audio_text"]) and has_read(i))
random.Random(0).shuffle(DROPPED)
ALL = sorted(SAMPLES)


def j(x) -> str:
    return json.dumps(x, ensure_ascii=False)


def sample_view(i: str) -> dict:
    s = SAMPLES[i]
    hist = []
    for h in s["history"]:
        if h["role"] == "user":
            hist.append({"role": "user", "text": h.get("audio_text", "")})
        else:
            calls = [{"name": c["name"], "args": j(c["arguments"]), "result": j(r)} for c, r in zip(h.get("tool_calls") or [], h.get("tool_results") or [])]
            hist.append({"role": "assistant", "text": h.get("text", ""), "calls": calls})
    calls = [{"k": k, "name": c["name"], "kind": kind(c["name"]), "args": j(c["arguments"]), "result": j(r)}
             for k, (c, r) in enumerate(zip(s["label"]["tool_calls"], s["label"]["tool_results"]), 1)]
    return {"id": i, "time": s["context"]["current_time"], "request": s["audio_text"], "history": hist, "calls": calls,
            "pass1": P1.get(i), "manual": MANUAL.get(i)}


def save(row: dict) -> dict:
    i = row["id"]
    s = SAMPLES[i]
    cond, k = row.get("conditional"), row.get("check_call")
    if cond is True:
        calls = s["label"]["tool_calls"]
        if not isinstance(k, int) or not 1 <= k <= len(calls) or kind(calls[k - 1]["name"]) != "read":
            return {"ok": False, "error": "bước kiểm tra phải là một call [read] của trajectory"}
    else:
        k = None
    MANUAL[i] = {"id": i, "conditional": cond, "check_call": k, "condition": (row.get("condition") or "").strip(),
                 "note": (row.get("note") or "").strip(), "t": int(time.time())}
    tmp = OUT.with_suffix(".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text("".join(j(MANUAL[x]) + "\n" for x in sorted(MANUAL)), encoding="utf-8")
    tmp.replace(OUT)
    return {"ok": True, "saved": MANUAL[i]}


class H(BaseHTTPRequestHandler):
    def send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def json(self, x, code: int = 200) -> None:
        self.send(code, j(x).encode("utf-8"), "application/json; charset=utf-8")

    def log_message(self, *a) -> None:  # quiet console
        pass

    def do_GET(self) -> None:
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/":
            return self.send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        if u.path == "/api/queue":
            ids = {"dropped": DROPPED, "suspect": SUSPECT}.get(q.get("set", "dropped"), ALL)
            return self.json({"ids": ids, "done": [i for i in ids if i in MANUAL]})
        if u.path == "/api/sample" and q.get("id") in SAMPLES:
            return self.json(sample_view(q["id"]))
        if u.path == "/api/plans" and q.get("id") in SAMPLES:
            k = int(q["k"]) if q.get("k", "").isdigit() else None
            return self.json({str(m): t for m, t in plan_segments(SAMPLES[q["id"]], k).items()})
        self.json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/label":
            return self.json({"error": "not found"}, 404)
        row = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if row.get("id") not in SAMPLES:
            return self.json({"ok": False, "error": "unknown id"}, 400)
        self.json(save(row))


PAGE = r"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gán nhãn điều kiện</title>
<style>
:root { color-scheme: light; --bg:#f4f6f9; --card:#fff; --ink:#1c2530; --muted:#5d6b7b; --line:#d9e0e8;
  --blue:#2f6fd6; --blue-soft:#e5eefc; --green:#1f8456; --green-soft:#e0f3e9; --red:#c2413a; --red-soft:#fbe7e5;
  --amber:#a8711b; --amber-soft:#fbf0dc; }
@media (prefers-color-scheme: dark) { :root { color-scheme: dark; --bg:#12181f; --card:#1a222c; --ink:#e3e9f1; --muted:#97a5b5;
  --line:#2c3846; --blue:#7aa7f0; --blue-soft:#1d2b42; --green:#6fcf9f; --green-soft:#17321f; --red:#f08a82; --red-soft:#3a2223;
  --amber:#e7b25f; --amber-soft:#35291a; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font: 15px/1.5 "Segoe UI", system-ui, sans-serif; }
header { position: sticky; top:0; z-index:2; background:var(--card); border-bottom:1px solid var(--line);
  display:flex; gap:12px; align-items:center; padding:10px 16px; flex-wrap: wrap; }
header b { font-size:16px; }
.bar { flex:1; min-width:120px; height:8px; background:var(--line); border-radius:4px; overflow:hidden; }
.bar i { display:block; height:100%; background:var(--green); width:0; }
main { max-width: 1100px; margin: 0 auto; padding: 16px; display:grid; gap:14px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 16px; }
h2 { margin:0 0 8px; font-size:13px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }
.req { font-size:19px; line-height:1.55; user-select:text; }
.mono { font-family: Consolas, "Cascadia Mono", monospace; font-size:13.5px; white-space:pre-wrap; word-break:break-word; }
.call { display:grid; grid-template-columns: 34px 1fr auto; gap:4px 10px; padding:8px 6px; border-radius:8px; align-items:start; }
.call + .call { border-top:1px dashed var(--line); }
.call.sel { background: var(--green-soft); outline: 2px solid var(--green); }
.k { font-weight:700; color:var(--muted); text-align:right; padding-top:1px; }
.tag { font-size:11px; font-weight:700; padding:1px 6px; border-radius:5px; margin-right:6px; }
.tag.read { background:var(--blue-soft); color:var(--blue); } .tag.write { background:var(--amber-soft); color:var(--amber); }
.tag.ask, .tag.meta { background:var(--line); color:var(--muted); }
.res { color:var(--muted); grid-column: 2 / 4; }
button { font: inherit; border:1px solid var(--line); background:var(--card); color:var(--ink); border-radius:8px; padding:6px 12px; cursor:pointer; }
button:hover { border-color: var(--blue); }
button.on { background:var(--blue); color:#fff; border-color:var(--blue); }
button.yes.on { background:var(--green); border-color:var(--green); } button.no.on { background:var(--red); border-color:var(--red); }
button.unsure.on { background:var(--amber); border-color:var(--amber); }
.row { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
input[type=text] { font: inherit; flex:1; min-width: 220px; padding:6px 10px; border:1px solid var(--line); border-radius:8px; background:var(--bg); color:var(--ink); }
.hist { color:var(--muted); font-size:14px; }
.hist div { margin: 2px 0; }
.plan { font-family: Consolas, monospace; font-size:13.5px; padding:4px 0; }
.muted { color:var(--muted); } .err { color:var(--red); font-weight:600; } .ok { color:var(--green); font-weight:600; }
kbd { border:1px solid var(--line); border-bottom-width:2px; border-radius:4px; padding:0 5px; font-size:12px; }
select { font: inherit; padding:5px 8px; border-radius:8px; border:1px solid var(--line); background:var(--card); color:var(--ink); }
</style></head><body>
<header>
  <b id="sid">...</b>
  <select id="set"><option value="dropped">Bị loại ở tầng 1 (8B có, không từ nối, có [read])</option><option value="suspect">Qua tầng 1 (8B có + từ nối + có [read])</option><option value="all">Tất cả train</option></select>
  <button id="prev">&larr;</button><span id="pos" class="muted"></span><button id="next">&rarr;</button>
  <button id="todo">Mẫu chưa gán tiếp theo</button>
  <div class="bar"><i id="barfill"></i></div><span id="progress" class="muted"></span>
</header>
<main>
  <section class="card"><h2>Câu user (bôi đen rồi bấm <kbd>C</kbd> để lấy làm điều kiện)</h2><div id="req" class="req"></div>
    <div id="time" class="muted" style="margin-top:4px"></div></section>
  <section class="card" id="histcard"><h2>Lịch sử hội thoại</h2><div id="hist" class="hist"></div></section>
  <section class="card"><h2>Trajectory gold (bấm số <kbd>1</kbd>-<kbd>9</kbd> hoặc nút để chọn bước kiểm tra, phải là [read])</h2><div id="calls"></div></section>
  <section class="card"><h2>Nhãn của bạn</h2>
    <div class="row">
      <button class="yes" id="byes">Có điều kiện</button>
      <button class="no" id="bno">Không (<kbd>N</kbd>)</button>
      <button class="unsure" id="bunsure">Không chắc (<kbd>U</kbd>)</button>
      <span id="kshow" class="muted"></span>
    </div>
    <div class="row" style="margin-top:10px"><input type="text" id="cond" placeholder="Cụm điều kiện trong câu (tuỳ chọn, ví dụ: nếu mai mưa)"></div>
    <div class="row" style="margin-top:8px"><input type="text" id="note" placeholder="Ghi chú (tuỳ chọn)"></div>
    <div class="row" style="margin-top:10px"><button class="on" id="save">Lưu + mẫu tiếp (<kbd>Enter</kbd>)</button><span id="msg"></span></div>
  </section>
  <section class="card"><h2>Plan mà bản A sẽ học với nhãn này</h2><div id="plans"></div></section>
  <section class="card"><h2>Nhãn Qwen3-8B lượt 1 (<kbd>H</kbd> để hiện/ẩn, ẩn mặc định cho khỏi bị dẫn dắt)</h2><div id="p1" class="mono muted">(đang ẩn)</div></section>
</main>
<script>
let ids = [], done = new Set(), pos = 0, cur = null, st = {cond: undefined, k: null}, showP1 = false;
const $ = (x) => document.getElementById(x);
const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;"}[c]));

async function loadQueue() {
  const r = await (await fetch("/api/queue?set=" + $("set").value)).json();
  ids = r.ids; done = new Set(r.done);
  const h = location.hash.slice(1);
  pos = ids.indexOf(h) >= 0 ? ids.indexOf(h) : Math.max(0, ids.findIndex((i) => !done.has(i)));
  await show();
}
function progress() {
  $("progress").textContent = `${done.size}/${ids.length} đã gán`;
  $("barfill").style.width = (100 * done.size / Math.max(1, ids.length)) + "%";
  $("pos").textContent = `${pos + 1}/${ids.length}`;
}
async function show() {
  if (!ids.length) return;
  const id = ids[pos];
  location.hash = id;
  cur = await (await fetch("/api/sample?id=" + id)).json();
  $("sid").textContent = id + (done.has(id) ? "  ✓" : "");
  $("req").textContent = cur.request;
  $("time").textContent = "Thời điểm: " + cur.time;
  $("histcard").style.display = cur.history.length ? "" : "none";
  $("hist").innerHTML = cur.history.map((h) => h.role === "user" ? `<div><b>user:</b> ${esc(h.text)}</div>` :
    `<div>${(h.calls || []).map((c) => `<div class="mono">    -> ${esc(c.name)} ${esc(c.args)}<br>         = ${esc(c.result.slice(0, 300))}</div>`).join("")}${h.text ? `<div><b>assistant:</b> ${esc(h.text)}</div>` : ""}</div>`).join("");
  const m = cur.manual;
  st = m ? {cond: m.conditional, k: m.check_call} : {cond: undefined, k: null};
  $("cond").value = m ? m.condition : ""; $("note").value = m ? m.note : ""; $("msg").textContent = "";
  renderCalls(); renderState(); renderP1(); progress();
}
function renderCalls() {
  $("calls").innerHTML = cur.calls.length ? cur.calls.map((c) => `<div class="call ${st.cond === true && st.k === c.k ? "sel" : ""}">
      <div class="k">${c.k}</div>
      <div class="mono"><span class="tag ${c.kind}">${c.kind}</span>-> ${esc(c.name)} ${esc(c.args)}</div>
      <div>${c.kind === "read" ? `<button data-k="${c.k}">Bước kiểm tra</button>` : ""}</div>
      <div class="mono res">   = ${esc(c.result.slice(0, 600))}</div></div>`).join("") : `<div class="muted">(không gọi tool nào)</div>`;
  document.querySelectorAll("#calls button[data-k]").forEach((b) => b.onclick = () => pick(+b.dataset.k));
}
function renderState() {
  $("byes").classList.toggle("on", st.cond === true);
  $("bno").classList.toggle("on", st.cond === false);
  $("bunsure").classList.toggle("on", st.cond === null);
  $("kshow").textContent = st.cond === true ? `bước kiểm tra: call ${st.k}` : "";
  renderCalls(); plans();
}
async function plans() {
  const k = st.cond === true ? st.k : "";
  const r = await (await fetch(`/api/plans?id=${cur.id}&k=${k}`)).json();
  $("plans").innerHTML = Object.entries(r).map(([m, t]) => `<div class="plan"><span class="muted">message ${m}:</span> ${esc(t)}</div>`).join("");
}
function renderP1() {
  const p = cur.pass1;
  $("p1").textContent = showP1 ? (p ? JSON.stringify(p) : "(không có)") : "(đang ẩn)";
}
function pick(k) {
  const c = cur.calls.find((x) => x.k === k);
  if (!c) return;
  if (c.kind !== "read") { $("msg").innerHTML = `<span class="err">call ${k} là [${c.kind}], không phải [read]</span>`; return; }
  st = {cond: true, k}; $("msg").textContent = ""; renderState();
}
async function save() {
  if (st.cond === undefined) { $("msg").innerHTML = `<span class="err">chưa chọn nhãn</span>`; return; }
  if (st.cond === true && !st.k) { $("msg").innerHTML = `<span class="err">chọn bước kiểm tra (call [read])</span>`; return; }
  const r = await (await fetch("/api/label", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({id: cur.id, conditional: st.cond, check_call: st.k, condition: $("cond").value, note: $("note").value})})).json();
  if (!r.ok) { $("msg").innerHTML = `<span class="err">${esc(r.error)}</span>`; return; }
  done.add(cur.id);
  if (pos < ids.length - 1) { pos++; await show(); } else { progress(); $("msg").innerHTML = `<span class="ok">đã lưu (mẫu cuối)</span>`; }
}
$("byes").onclick = () => { const r = cur.calls.filter((c) => c.kind === "read"); if (r.length) pick(st.k && st.cond === true ? st.k : r[r.length - 1].k); else $("msg").innerHTML = `<span class="err">không có call [read] nào</span>`; };
$("bno").onclick = () => { st = {cond: false, k: null}; renderState(); };
$("bunsure").onclick = () => { st = {cond: null, k: null}; renderState(); };
$("save").onclick = save;
$("prev").onclick = () => { if (pos > 0) { pos--; show(); } };
$("next").onclick = () => { if (pos < ids.length - 1) { pos++; show(); } };
$("todo").onclick = () => { const n = ids.findIndex((i, j) => j > pos && !done.has(i)); const f = n >= 0 ? n : ids.findIndex((i) => !done.has(i)); if (f >= 0) { pos = f; show(); } };
$("set").onchange = () => { location.hash = ""; loadQueue(); };
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") { if (e.key === "Enter") { e.preventDefault(); save(); } return; }
  if (/^[1-9]$/.test(e.key)) pick(+e.key);
  else if (e.key === "n" || e.key === "N") $("bno").click();
  else if (e.key === "u" || e.key === "U") $("bunsure").click();
  else if (e.key === "c" || e.key === "C") { const s = String(getSelection()).trim(); if (s) $("cond").value = s; }
  else if (e.key === "h" || e.key === "H") { showP1 = !showP1; renderP1(); }
  else if (e.key === "Enter") { e.preventDefault(); save(); }
  else if (e.key === "ArrowLeft") $("prev").click();
  else if (e.key === "ArrowRight") $("next").click();
});
loadQueue();
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    print(f"queue bị loại tầng 1: {len(DROPPED)} | qua tầng 1: {len(SUSPECT)} | all: {len(ALL)} | đã gán: {len(MANUAL)} -> {OUT}")
    print(f"http://127.0.0.1:{a.port}")
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
