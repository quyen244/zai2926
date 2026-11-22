"""Create cached, meaning-preserving text variants for synthetic ASR speech."""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from paraphrase import ask, parse, parse_objs  # noqa: E402
from lifemate.normalize import normalize_text  # noqa: E402

TRAIN = ROOT / "data/train+public_test/train/train.jsonl"
OUT = ROOT / "work/synth/asr_rollouts_v2.jsonl"
MODEL = "openai/gpt-oss-120b"
JUDGE_MODEL = "google/gemini-3-flash-preview"
SYSTEM = """Bạn viết lại câu nói tiếng Việt dùng để ra lệnh cho trợ lý giọng nói.
Tạo đúng {k} cách nói mới, tự nhiên, khác nhau rõ rệt nhưng giữ nguyên ý định và mọi chi tiết của câu gốc.
Không đổi tên người/địa điểm, số, tiền, ngày giờ, vật thể, hành động, phủ định, điều kiện hoặc thứ tự các việc.
Không thêm chi tiết và không trả lời yêu cầu. Chỉ trả về JSON array các chuỗi tiếng Việt."""
FEWSHOT = """Ví dụ dễ:
Câu gốc: Cái còn lại thì mai mưa dời muộn thêm nửa tiếng cho tôi ngủ thêm, hông mưa thì cứ giữ nguyên nha.
Câu viết lại đạt: Nếu mai mưa thì dời những lịch còn lại trễ thêm nửa tiếng để tôi ngủ thêm; nếu không mưa thì giữ nguyên nhé.

Ví dụ khó:
Câu gốc: Nếu ví đủ thì trả tiền truyền hình bằng ví và nhắn SMS cho Tuấn là đã đóng, còn không đủ thì trả bằng ngân hàng và nhắc tôi tám giờ tối nay nạp ví nha.
Câu viết lại đạt: Ví đủ tiền thì thanh toán tiền truyền hình bằng ví rồi nhắn SMS cho Tuấn báo đã đóng. Nếu thiếu tiền trong ví thì trả qua ngân hàng, đồng thời nhắc tôi nạp ví lúc tám giờ tối nay.

Hai ví dụ chỉ minh hoạ cách giữ điều kiện, hành động và chi tiết. Không lấy lại nội dung của ví dụ cho câu mới."""
JUDGE_SYSTEM = """Bạn kiểm định các câu paraphrase tiếng Việt dùng làm transcript huấn luyện ASR.
Chỉ đánh dấu đạt nếu cách hiểu tự nhiên của câu mới giữ nguyên chính xác câu gốc.
So từng câu về: việc cần làm, phủ định, điều kiện, số/tên/địa điểm/thời gian, đối tượng và vai ngữ nghĩa (ai làm, làm với ai/cái gì, ai nhận hoặc được lợi), thứ tự các hành động.
Nếu câu mơ hồ hơn, đổi vai người gọi/người được gọi, đổi người nhận/người gửi, bỏ hoặc thêm hành động, hoặc không chắc chắn thì đánh dấu false.
Trả về JSON array đúng thứ tự, mỗi phần tử là {\"ok\": true/false, \"reason\": \"lý do ngắn\"}."""


def sources() -> list[dict]:
    rows = []
    for line in TRAIN.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sample = json.loads(line)
        if sample.get("audio") and sample.get("audio_text"):
            rows.append({"id": sample["id"], "audio": sample["audio"], "text": sample["audio_text"]})
        for i, turn in enumerate(sample.get("history", [])):
            if turn.get("audio") and turn.get("audio_text"):
                rows.append({"id": f"{sample['id']}_u{i+1}", "audio": turn["audio"], "text": turn["audio_text"]})
    unique = {r["audio"]: r for r in rows}
    return sorted(unique.values(), key=lambda r: r["id"])


def norm(s: str) -> str:
    return re.sub(r"\W+", " ", normalize_text(s).lower()).strip()


def gen(row: dict, k: int) -> dict:
    original_nums = sorted(re.findall(r"\d+", norm(row["text"])))
    kept, seen, errs, total_sec = [], {norm(row["text"])}, [], 0.0
    for _ in range(3):
        # `ask` uses the shared Kaggle Model Proxy auth helper; failed rows can resume from cache.
        ask_n = k + 2
        res = ask(MODEL, SYSTEM.format(k=ask_n), FEWSHOT + "\n\n" + json.dumps({"source": row["text"], "already_generated": kept, "return": f"{ask_n} rewrites"}, ensure_ascii=False), "low")
        total_sec += res.get("sec", 0)
        if res.get("error"):
            errs.append(res["error"])
        candidates = []
        for text in parse(res.get("content", "")):
            nt = norm(text)
            nums = sorted(re.findall(r"\d+", nt))
            ratio = len(text) / max(1, len(row["text"]))
            min_ratio = 0.3 if len(row["text"]) < 24 else 0.45
            if nt in seen or nums != original_nums or not min_ratio <= ratio <= 2.3:
                continue
            candidates.append(text)
        if candidates:
            judged = ask(JUDGE_MODEL, JUDGE_SYSTEM, json.dumps({"source": row["text"], "candidates": candidates}, ensure_ascii=False), "low")
            total_sec += judged.get("sec", 0)
            verdicts = parse_objs(judged.get("content", ""))
            for i, text in enumerate(candidates):
                if i < len(verdicts) and verdicts[i].get("ok") is True:
                    kept.append(text)
                    seen.add(norm(text))
                    if len(kept) >= k:
                        break
        if len(kept) >= k:
            break
    return {**row, "rollouts": kept, "n_requested": k, "error": "; ".join(errs) or None, "sec": round(total_sec, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    rows = sources()
    if args.all:
        selected = rows
    else:
        selected = random.Random(42).sample(rows, min(args.limit, len(rows)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                cache[r["id"]] = r
    todo = [r for r in selected if len(cache.get(r["id"], {}).get("rollouts", [])) < args.k]
    print(f"sources={len(rows)} selected={len(selected)} cached={len(selected)-len(todo)} pending={len(todo)} workers={args.workers}", flush=True)
    with OUT.open("a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(gen, r, args.k): r for r in todo}
        for n, future in enumerate(as_completed(futures), 1):
            r = future.result()
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            cache[r["id"]] = r
            if n % 25 == 0 or n == len(todo):
                print(f"completed={n}/{len(todo)} latest={r['id']} kept={len(r['rollouts'])}", flush=True)
    missing = [r["id"] for r in selected if len(cache.get(r["id"], {}).get("rollouts", [])) < args.k]
    print(f"done={len(selected)-len(missing)}/{len(selected)} missing={len(missing)} output={OUT}", flush=True)


if __name__ == "__main__":
    main()
