"""Stage 1.1: build a local listening sheet (work/listening/index.html) for a structured listening pass.

120 train clips (taken from the ASR bench set, stratified by taxonomy so dialect, money, names, self-repair and long
clips are all present) + 50 public_test clips. Each row: audio player, reference transcript (train), every ASR hypothesis
found in work/asr/*.jsonl, and fields to fill (region, noise, speed, clarity, reference wrong?, hard words, notes).
Answers are kept in the browser's localStorage and exported with the "Export CSV" button.
Open the file directly in a browser (it plays the wav files from data/ by relative path).
Run: python scripts/stage1_listening.py
"""
from __future__ import annotations

import html
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import soundfile as sf  # noqa: E402

from lifemate.data import PROJECT, audio_path, load_samples  # noqa: E402

OUT = PROJECT / "work" / "listening"
PRIORITY = ["dialect_central", "dialect_south", "money", "self_repair", "conditional", "english_word", "history_reference", "negation"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(11)
    train = {s["id"]: s for s in load_samples("train")}
    tags_path = PROJECT / "work" / "tags.jsonl"
    if not tags_path.is_file():
        raise SystemExit(f"{tags_path} not found: run `python scripts/stage1_taxonomy.py` first (work/ is machine-local, not synced by git)")
    tags = {json.loads(line)["id"]: json.loads(line) for line in tags_path.read_text(encoding="utf-8").splitlines()}
    bench_path = PROJECT / "work" / "asr" / "bench_ids.json"
    pool = json.loads(bench_path.read_text(encoding="utf-8")) if bench_path.is_file() else sorted(train)
    if len(pool) < 120:
        print(f"warning: pool is only {len(pool)} ids ({bench_path} looks stale/partial); picking from it anyway")
    hyps: dict[str, dict[str, str]] = {}  # sample id -> {model tag: hyp} (old asr_bench.py files, id-keyed)
    for f in sorted((PROJECT / "work" / "asr").glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            hyps.setdefault(r["id"], {})[f.stem] = r["hyp"]
    phowhisper: dict[str, dict[str, str]] = {}  # split -> {audio path: hyp} (the real Stage 2.1 production output)
    for split in ("train", "public_test"):
        f = PROJECT / "work" / "asr" / "phowhisper" / f"{split}.jsonl"
        if f.is_file():
            phowhisper[split] = {json.loads(line)["audio"]: json.loads(line)["hyp"] for line in f.read_text(encoding="utf-8").splitlines() if line.strip()}

    chosen: list[str] = []
    for tag in PRIORITY:  # 10 per priority tag, then fill with random bench clips
        cands = [i for i in pool if tag in tags[i]["linguistic"] and i not in chosen]
        rng.shuffle(cands)
        chosen += cands[:10]
    rest = [i for i in pool if i not in chosen]
    rng.shuffle(rest)
    chosen += rest[:120 - len(chosen)]

    pub = load_samples("public_test")
    rng.shuffle(pub)
    rows = [("train", train[i], tags[i]) for i in sorted(chosen)] + [("public_test", s, None) for s in sorted(pub[:50], key=lambda s: s["id"])]

    items = []
    for split, s, t in rows:
        p = audio_path(split, s["audio"])
        with sf.SoundFile(p) as f:
            dur = f.frames / f.samplerate
        rel = Path("../../data/train+public_test") / split / s["audio"]
        row_hyps = dict(hyps.get(s["id"], {}))
        ph = phowhisper.get(split, {}).get(s["audio"])
        if ph is not None:
            row_hyps["phowhisper"] = ph
        hyp_html = "".join(f"<div class='hyp'><b>{html.escape(m)}</b> {html.escape(h)}</div>" for m, h in sorted(row_hyps.items()))
        ref = html.escape(s.get("audio_text", "")) or "<i>public_test: no transcript, write what you hear in notes</i>"
        tag_txt = " ".join((t or {}).get("linguistic", []))
        items.append(f"""
<tr data-id="{s['id']}">
  <td class="id">{s['id']}<br><small>{dur:.1f}s · {len(s['history'])} hist</small><br><small>{tag_txt}</small></td>
  <td><audio controls preload="none" src="{rel.as_posix()}"></audio><div class="ref">{ref}</div>{hyp_html}</td>
  <td class="form">
    <label>Region <select name="region"><option></option><option>Bắc</option><option>Trung</option><option>Nam</option><option>?</option></select></label>
    <label>Noise <select name="noise"><option></option><option>0 clean</option><option>1 some</option><option>2 heavy</option></select></label>
    <label>Volume <select name="volume"><option></option><option>quiet</option><option>normal</option><option>loud</option></select></label>
    <label>Speed <select name="speed"><option></option><option>slow</option><option>normal</option><option>fast</option></select></label>
    <label>Clarity <select name="clarity"><option></option><option>easy</option><option>hard</option><option>unintelligible</option></select></label>
    <label>Reference wrong? <select name="ref_wrong"><option></option><option>no</option><option>yes</option></select></label>
    <input name="hard_words" placeholder="hard words / misheard spans">
    <input name="notes" placeholder="notes (xem convention ở header)">
  </td>
</tr>""")

    page = f"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><title>LifeMate listening sheet</title>
<style>
body{{font:14px/1.5 system-ui,sans-serif;margin:16px;color:#18202e;background:#f5f7fa}}
table{{border-collapse:collapse;width:100%;background:#fff}} td{{border-bottom:1px solid #d9dfe8;padding:8px;vertical-align:top}}
td.id{{width:140px;font-family:monospace}} td.form{{width:300px}} .form label{{display:block;margin-bottom:3px}}
.form select{{float:right}} .form input{{width:100%;box-sizing:border-box;margin-top:4px}}
.ref{{margin:6px 0;font-weight:600}} .hyp{{color:#5b6577}} .hyp b{{font-family:monospace;font-weight:600;color:#1f5fd1}}
header{{position:sticky;top:0;background:#f5f7fa;padding:8px 0 4px;display:flex;flex-direction:column;gap:4px}}
header .row{{display:flex;gap:12px;align-items:center}}
tr.done td{{background:#eef6f1}} audio{{width:100%}}
</style></head><body>
<header><div class="row"><b>LifeMate listening sheet</b> <span id="count"></span> <button id="export">Export CSV</button>
<small>Answers are saved in this browser automatically.</small></div>
<small>notes convention: <b>[ellipsis]</b> ref thêm từ không nói ra &middot; <b>[mishear]</b> nghe nhầm thật &middot; <b>[code_switch]</b> ASR lẫn từ tiếng Anh/tên riêng</small>
<small><b>Clarity</b> = cảm nhận tổng thể nghe lần đầu (easy/hard/unintelligible), gộp cả noise/volume/speed/phát âm &middot; <b>hard words</b> = từ khó nghe trong chính audio (không so với ref hay ASR) -&gt; dùng làm từ điển hotword cho ASR sau này</small></header>
<table>{''.join(items)}</table>
<script>
const KEY='lifemate-listening-v1';
let db={{}}; try{{db=JSON.parse(localStorage.getItem(KEY)||'{{}}')}}catch(e){{}}
const rows=[...document.querySelectorAll('tr[data-id]')];
function save(){{try{{localStorage.setItem(KEY,JSON.stringify(db))}}catch(e){{}}; mark()}}
function mark(){{let n=0;rows.forEach(r=>{{const d=db[r.dataset.id]||{{}};const ok=d.region&&d.clarity;r.classList.toggle('done',!!ok);if(ok)n++}});document.getElementById('count').textContent=n+'/'+rows.length+' done'}}
rows.forEach(r=>{{const id=r.dataset.id;r.querySelectorAll('select,input').forEach(el=>{{
  if(db[id]&&db[id][el.name]) el.value=db[id][el.name];
  el.addEventListener('change',()=>{{db[id]=db[id]||{{}};db[id][el.name]=el.value;save()}});}})}});
mark();
document.getElementById('export').onclick=()=>{{
  const cols=['region','noise','volume','speed','clarity','ref_wrong','hard_words','notes'];
  const q=v=>'"'+String(v||'').replace(/"/g,'""')+'"';
  const lines=['id,'+cols.join(',')].concat(rows.map(r=>[r.dataset.id].concat(cols.map(c=>(db[r.dataset.id]||{{}})[c])).map(q).join(',')));
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['\\ufeff'+lines.join('\\n')],{{type:'text/csv'}}));a.download='listening.csv';a.click();
}};
</script></body></html>"""
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print(f"{len(rows)} clips -> {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
